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

Current level: 1 (Repeatable), on the chapter's scale of 0 Manual, 1 Repeatable, 2 Automated, 3 Governed, 4 Optimizing

Evidence: The data pipeline is defined in code and reruns the same way each time: Terraform for the Glue jobs and the Feature Store (`infrastructure/modules/glue`, `infrastructure/modules/feature_store`), `glue-scripts/feature_engineer.py`, and a versioned data bucket. Every build also tests the data. `docs/lab4-build-green.log` shows 28 passing `test_data.py` tests covering schema, grain, leakage, freshness and no post-cutoff data, and `data/feature_engineering.py` matches the Glue output for all 9,999 customers. But Level 2 means pipelines trigger on their own, and mine do not. I started the crawler, the transform and the feature job by hand, and the freshness test failed on 13-day-old features until I reran the job myself. Monitoring is basic: CI checks the data, but nothing watches it between builds.

Gap to Level 2: Nothing triggers the data pipeline, and there is no systematic data quality monitoring outside CI.

Top priority investment: An EventBridge Scheduler rule that starts a Glue Workflow (crawler, transform, feature job) every night, plus a Glue Data Quality ruleset on `processed/` that alarms to an SNS topic. DataOps is the weaker of my two disciplines, so it is the one to invest in first.

## MLOps Maturity

Current level: 2 (Automated)

Evidence: A push to `main` starts the pipeline on its own: tests, spot training, registration and the gates (`buildspec.yml`, `pipeline.yaml`). The final green run was started by a push, not by hand: the execution for commit `1e73a9a` began at 23:35:56 (`docs/lab4-pipeline-runs.txt`), and CodePipeline listed its trigger type as `WebhookV2` when I checked before deleting the stack. Using the chapter's delivery measures on my own runs, a build went from start to a registered package in about 6 minutes (build `cf279eae` started 23:00:43, v4 appeared 23:06:24). The deliberate bad commit never reached the registry, and the alarm went ALARM at 22:59:17 and back to OK at 23:08:17, so I restored in 9 minutes (`docs/lab4-failure-demo.txt`). A few Level 3 pieces exist: the champion regression test passed against v3 (`champion v3 auc=0.7721; candidate auc=0.7721`), packages carry lineage (`docs/lab4-model-package.json`), and the fairness check flags a 42.3pp recall gap between Bronze (0.471) and Platinum (0.048).

Gap to Level 3: Nothing retrains the model on a drift signal. The two triggers in this document are designs only, and with no model deployed yet there is no production signal to monitor. The audit trail is also partial: approval is a manual console action with no policy on who can approve, and the fairness flag has no owner.

Top priority investment: SageMaker Model Monitor (data quality and model quality schedules) on the Lab 5 endpoint, with a CloudWatch alarm that goes through an EventBridge rule to `StartPipelineExecution`, as described under Retraining Triggers.
