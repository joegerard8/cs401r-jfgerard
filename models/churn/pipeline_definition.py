"""SageMaker Pipeline definition for the NorthStar churn model.

CS 401R Lab 4. `pipeline.yaml` has referred to this file since the template was
written; it had never actually existed, so the CodeBuild stage's core action --
`aws sagemaker start-pipeline-execution --pipeline-name northstar-churn-pipeline`
-- had nothing to start and the Lab 4 pipeline could not run end to end.

Two steps:
    Train     XGBoost framework estimator running models/churn/train_sagemaker.py
    Register  RegisterModel, ModelApprovalStatus=PendingManualApproval

The approval status is the point of the lab. This pipeline never approves its
own model: promotion is the human gate in CodePipeline's ManualApproval stage,
and a pipeline that registered as Approved would make that stage decorative.

Upsert the pipeline (idempotent, safe to re-run):
    python models/churn/pipeline_definition.py \
        --role-arn "$(terraform -chdir=infrastructure/environments/dev output -raw ml_engineer_role_arn)" \
        --bucket northstar-dev-data-$(aws sts get-caller-identity --query Account --output text) \
        --upsert

QUOTA. The TrainingStep runs on MANAGED SPOT capacity (use_spot_instances).
The AWS *default* on-demand training quota is 0 for every instance family, so
an on-demand job fails with ResourceLimitExceeded on every brand-new student
account. Spot training has non-zero defaults: `ml.m5.large for spot training
job usage` (L-29688C85) defaults to 4. Switching to spot (2026-09-30) is what
lets Lab 4 run within default quota with no per-student Support case, and it
retired Pre-Lab 4. Keep use_spot_instances=True.
"""

import argparse
import pathlib

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import TrainingStep
from sagemaker.xgboost.estimator import XGBoost

PIPELINE_NAME = "northstar-churn-pipeline"
MODEL_PACKAGE_GROUP = "northstar-churn-models"

# 1.7-1, not 1.5-1 and not latest. Verified 2026-07-31: a modern (3.x) XGBoost
# model will NOT load in the 1.5.2 container in either JSON or UBJ, which is
# what blocks the Graviton path; it does load in 1.7-1. Do not bump this
# without re-running that check.
FRAMEWORK_VERSION = "1.7-1"


def build_pipeline(role_arn: str, bucket: str, region: str = "us-east-1") -> Pipeline:
    boto_session = boto3.Session(region_name=region)
    # PipelineSession, not Session. With a plain Session the estimator's .fit()
    # executes a training job immediately at definition time instead of being
    # captured as a step, so building the pipeline would start a real job and
    # bill for it.
    session = PipelineSession(boto_session=boto_session,
                              default_bucket=bucket)

    # ── Parameters ────────────────────────────────────────────────────────────
    # Note what is NOT a parameter: the training code location.
    #
    # buildspec.yml used to zip models/ and pass a TrainingCodeS3Uri parameter,
    # and the estimator's source_dir cannot consume that. source_dir is resolved
    # at definition time -- the SDK urlparses it -- so handing it a
    # ParameterString fails with "'ParameterString' object has no attribute
    # 'decode'". It also expects a .tar.gz, while the buildspec produced a .zip.
    #
    # Instead the buildspec upserts this pipeline on every build, which packages
    # the checked-out source_dir to S3 as part of the upsert. The execution is
    # still pinned to the triggering commit -- more directly than before, since
    # the code is the working tree CodeBuild cloned rather than a URI that could
    # point anywhere. CommitSha survives as a parameter for traceability.
    commit_sha = ParameterString("CommitSha", default_value="unknown")
    input_data = ParameterString(
        "InputDataS3Uri", default_value=f"s3://{bucket}/features/customers/")
    instance_type = ParameterString("TrainingInstanceType", default_value="ml.m5.large")

    # See the ModelMetrics comment below for why this URI is built by hand and
    # shared with the container rather than read off the training step.
    metrics_uri = Join(on="", values=[
        f"s3://{bucket}/artifacts/evaluations/", commit_sha, "/evaluation_metrics.json"])

    estimator = XGBoost(
        # source_dir is models/churn, not the repo root: the XGBoost 1.7-1
        # container already ships pandas, numpy, scikit-learn, pyarrow and
        # xgboost, so the only things that need to travel are this entry point
        # and churn_model.py, which it imports (your Lab 3 training logic).
        # Packaging the repo root would upload data/ into every training job.
        entry_point="train_sagemaker.py",
        source_dir=str(pathlib.Path(__file__).parent),
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        output_path=f"s3://{bucket}/artifacts/training/",
        # Load-bearing. Left unset, the SDK uploads sourcedir.tar.gz to
        # s3://<bucket>/<pipeline-name>/code/<hash>/ -- a top-level prefix the
        # MLEngineer role cannot read. Its S3 grant is deliberately scoped to
        # artifacts/* and features/*, so the training container failed at
        # startup downloading its own code. artifacts/code/ is also exactly
        # where the old buildspec put the training zip (defect 68).
        code_location=f"s3://{bucket}/artifacts/code",
        sagemaker_session=session,
        environment={"COMMIT_SHA": commit_sha, "METRICS_S3_URI": metrics_uri},
        # The model trains in seconds on 10k rows. A tight cap turns a runaway
        # job into a fast failure instead of a bill.
        max_run=1800,
        # Managed spot, not on-demand. On-demand training quota defaults to 0
        # on a new account; ml.m5.large spot defaults to 4. max_wait is the
        # total budget including time waiting for spot capacity and any
        # restart after an interruption, and must be >= max_run. No
        # checkpointing: the job trains in seconds, so an interrupted job
        # simply restarts from scratch.
        use_spot_instances=True,
        max_wait=3600,
    )

    train_step = TrainingStep(
        name="TrainChurnModel",
        step_args=estimator.fit({
            "training": TrainingInput(s3_data=input_data, content_type="application/x-parquet"),
        }),
    )

    # ModelMetrics points at the metrics object the training job uploaded, so
    # the Model Registry entry carries its own evidence. Without this the
    # registry holds an artifact with no record of why it was promotable --
    # which is exactly how the 2026-08-01 slice numbers were lost.
    #
    # It has to be this deterministic URI, derived the same way on both sides,
    # rather than anything read off the training step. Two things rule the
    # obvious alternatives out:
    #   * step.properties.* are lazy pipeline references, not strings. You
    #     cannot .replace() a path out of them at definition time.
    #   * /opt/ml/output/data is uploaded as output.tar.gz, so there is no bare
    #     evaluation_metrics.json next to the model artifact to point at.
    # So train_sagemaker.py PUTs the metrics to METRICS_S3_URI itself, and
    # Join() builds the identical URI here from the same parameter.
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(s3_uri=metrics_uri,
                                       content_type="application/json")
    )

    register_step = RegisterModel(
        name="RegisterChurnModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[commit_sha, input_data, instance_type],
        steps=[train_step, register_step],
        sagemaker_session=session,
    )


# Create the model package group if it does not exist. Kept out of
# build_pipeline() so that building a definition stays free of side effects.
#
# Until 2026-10-01 this function sat BELOW the `if __name__ == "__main__"`
# block and nothing called it, so a fresh account's first RegisterModel step
# depended on the group already existing from Lab 3. It is now called on
# --upsert. CodeBuild's role already grants CreateModelPackageGroup for this.
def ensure_model_package_group(region: str = "us-east-1") -> None:
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.describe_model_package_group(ModelPackageGroupName=MODEL_PACKAGE_GROUP)
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="NorthStar churn models, CS 401R Lab 4",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-arn", required=True,
                    help="MLEngineer role ARN from the Lab 1/2 Terraform output")
    ap.add_argument("--bucket", required=True,
                    help="northstar-dev-data-<account-id>")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--upsert", action="store_true",
                    help="Create or update the pipeline in SageMaker")
    ap.add_argument("--definition-out", default=None,
                    help="Write the resolved JSON definition here (no AWS mutation)")
    args = ap.parse_args()

    pipeline = build_pipeline(args.role_arn, args.bucket, args.region)

    if args.definition_out:
        with open(args.definition_out, "w") as f:
            f.write(pipeline.definition())
        print(f"Definition written to {args.definition_out}")

    if args.upsert:
        ensure_model_package_group(args.region)
        pipeline.upsert(role_arn=args.role_arn)
        print(f"Upserted pipeline {PIPELINE_NAME}")
    elif not args.definition_out:
        print(pipeline.definition())


if __name__ == "__main__":
    main()

