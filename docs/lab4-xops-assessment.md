## Lab 4: XOps Assessment

This covers the CI/CD pipeline for the churn model. The evidence files are in `docs/` and are named below where I use them.

## Pipeline Stage Mapping

I used AWS CodePipeline and CodeBuild, since the Lab 1 to 3 infrastructure is already on AWS and the CodeBuild role can reach the data bucket and the Model Registry without extra credentials. The CodePipeline has three stages, and the five phases sit inside them.

| Phase | CodePipeline stage | Where it happens |
|---|---|---|
| Source | `Source` | CodeConnections to GitHub, push to `main` (trigger type `WebhookV2`) |
| Test | `Build` | `buildspec.yml` `pre_build`: `pytest` on `test_data.py` and `test_features.py` |
| Build | `Build` | `build` phase: upserts and runs the SageMaker Pipeline, whose `TrainChurnModel` step runs my `train()` on managed spot |
| Evaluate | `Build` | end of `build`: `pytest` on `test_model.py` and `test_fairness.py` against the metrics that run emitted |
| Register | `Build` | the `RegisterChurnModel` step in the SageMaker Pipeline, as `PendingManualApproval` |
| (human gate) | `ManualApproval` | SNS-notified approval action |

I did not make one stage per phase. A stage is a place to hand artifacts off or put a person in, and the model never leaves the SageMaker Pipeline, so extra stages would only mean more CodeBuild projects and artifact passing.

Registration happens before the CodeBuild-only tests, so `post_build` sets the package to `Rejected` if those fail. A model that fails the gate inside `train()` never registers at all. A failure in the Test phase stops the build in `PRE_BUILD`, so neither Build nor Register runs (`docs/lab4-failure-demo.txt`).

One thing I should be clear about: pushes did not start the pipeline at first. The pipeline came up as V2 with no push trigger, and then the GitHub app was only authorized, not installed, so GitHub never sent push events. The two failure-demo runs (`78379493` and its revert `cf279eae`) were started with `start-pipeline-execution`, and they built the right commits. After I added a `Triggers` block, installed the app and used a new connection, an empty commit started the final green run (`8e7cafb6`, commit `1e73a9a`) by itself.

## Champion-Challenger

A candidate replaces the champion only if all three of these hold:

1. Precision@top10% is at least 0.50, recall@top10% is at least 0.25, and the lower end of the 95% lift CI over the recency baseline is above 0. These are the gates in `tests/test_model.py`.
2. Its AUC is at least the champion's AUC plus 0.01.
3. Its recall@top10% is at least the champion's minus 0.01, so it does not trade away the customers the retention team actually contacts.

Separately, the regression test fails any candidate whose AUC is more than 0.02 below the champion, and `post_build` rejects it. I picked 0.01 because the biggest improvement in Lab 3 (+0.0063, CI [-0.0011, +0.0137]) was still inside noise, so a smaller gain is not evidence of anything.

The final green run, v5, has AUC 0.7721, exactly equal to champion v3, so under this rule v3 stays champion and v5 stays pending. They match because they are the same code on the same data (`docs/lab4-build-green.log`).

## Retraining Triggers

Both of these are designed, not deployed.

Scheduled: Amazon EventBridge Scheduler rule `cron(0 6 ? * MON *)` (every Monday at 06:00 UTC) calls `StartPipelineExecution` on `northstar-churn-pipeline`. A second schedule runs the Glue feature job on Sunday at 22:00 UTC, so the data is fresh enough to pass the 48-hour freshness test.

Performance-based: a SageMaker Model Monitor model-quality schedule on the production endpoint compares predictions with the labels that arrive later. A CloudWatch alarm fires when the trailing 7-day AUC is below 0.72, which is the champion's 0.7721 minus 0.05, and at least 5,000 customers have labels. The alarm goes to an EventBridge rule that starts the same SageMaker Pipeline. The 5,000 minimum matters because my own tier slices show how noisy a small sample gets: Platinum has only 21 churners in the test set.

## DataOps Maturity

Current level: 2 (on a 0 to 4 scale: 0 manual, 1 scripted, 2 automated and tested, 3 monitored and self-triggering, 4 continuously optimizing)

Evidence: The infrastructure and Glue jobs are Terraform and code (`infrastructure/modules/glue`, `glue-scripts/feature_engineer.py`). The data is tested on every build: `tests/test_data.py` has 28 passing tests in `docs/lab4-build-green.log`, including schema, grain, leakage, freshness and a check that no feature uses post-cutoff data. I also ported the feature logic to `data/feature_engineering.py`, and it matches the Glue output for all 9,999 customers.

Gap to Level 3: Nothing runs the data pipeline. I started the crawler, transform and feature job by hand, and the freshness test failed on 13-day-old features until I reran the job myself. There is also no alerting when a data quality check fails outside CI.

Top priority investment: An EventBridge Scheduler rule that starts a Glue Workflow (crawler, transform, feature job) every night, with a Glue Data Quality ruleset on `processed/` that alarms through SNS.

## MLOps Maturity

Current level: 2

Evidence: Every push can run tests, train on spot, register and gate a model (`buildspec.yml`, `pipeline.yaml`). The failure demo shows a failing test stopping in `PRE_BUILD` with no new package, and the alarm email arriving (`docs/lab4-failure-demo.txt`, `docs/lab4-alarm-email.png`). The champion regression test passed against v3 (`champion v3 auc=0.7721; candidate auc=0.7721`), and the registry carries lineage: data URI, timestamp, commit and metrics (`docs/lab4-model-package.json`). The fairness check flags a 42.3pp recall gap between Bronze (0.471) and Platinum (0.048).

Gap to Level 3: There is no production monitoring or drift detection, and nothing retrains the model on its own. The triggers above are only designs. The fairness flag also has no owner or follow-up. It warns, but nobody is responsible for acting on it.

Top priority investment: SageMaker Model Monitor (data quality and model quality schedules) on the Lab 5 endpoint, with the CloudWatch alarm wired to `StartPipelineExecution` as described above.
