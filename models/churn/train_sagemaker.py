"""SageMaker training-container entry point for the NorthStar churn model.

CS 401R Lab 4. This is the script the SageMaker Pipeline's TrainingStep runs
inside the XGBoost framework container. You do not edit it. It is the thin
container-side adapter between SageMaker and YOUR training logic, which lives
in `churn_model.py` beside this file.

Why an adapter. Your Lab 3 script pulls its data from the Feature Store over
Athena, because it runs on a laptop with AWS credentials. A training container
must not make service calls for its input -- SageMaker stages the data into
/opt/ml/input/data/<channel> before the container starts. So this script reads
the channel, hands the DataFrame to churn_model.train(), and writes the outputs
where SageMaker, the CI gate, and Lab 5's serving container expect them.

Contract with churn_model.py (checked at startup, see _check_contract):
    FEATURE_COLUMNS   list[str]   model input columns, in order
    LABEL             str         "churn_label"
    train(df, seed=42) -> (xgboost.Booster, dict)
        dict must include precision_top10, recall_top10, and the bootstrap
        lift CI as lift_ci_low/lift_ci_high (or auc_lift_ci_low/_high).

Contract with SageMaker:
    in   /opt/ml/input/data/training/     Parquet or CSV feature rows
    out  /opt/ml/model/xgboost-model      UBJ model, the name the serving container wants
    out  /opt/ml/output/data/model.ubj, evaluation_metrics.json, model_metadata.json
"""

import argparse
import glob
import json
import os
import shutil
import sys

import pandas as pd

# churn_model.py sits beside this file inside source_dir. The training
# container's working directory is not guaranteed to be on sys.path, so add
# this file's own directory explicitly rather than relying on it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import churn_model  # noqa: E402


def _check_contract():
    """Fail in the first seconds of the job, with a message that says what to
    fix, instead of minutes later with an AttributeError from deep inside."""
    missing = [n for n in ("FEATURE_COLUMNS", "LABEL", "train")
               if not hasattr(churn_model, n)]
    if missing:
        raise SystemExit(
            f"churn_model.py is missing {missing}. It must define FEATURE_COLUMNS, "
            f"LABEL and train(df, seed=42) -> (xgboost.Booster, metrics dict). "
            f"See the contract at the top of churn_model.py.")
    return churn_model.FEATURE_COLUMNS, churn_model.LABEL, churn_model.train


FEATURE_COLUMNS, LABEL, train = _check_contract()

SLICE_COLUMN = "loyalty_tier"


def read_channel(channel_dir: str) -> pd.DataFrame:
    """Read every Parquet/CSV part in a channel into one DataFrame.

    Glue writes partitioned output, so a channel is a directory of part files,
    not a single file. Sorting the parts is load-bearing for the same reason
    ORDER BY is load-bearing in the Athena path (defect 45): the row order
    reaching train_test_split(random_state=42) determines the split, and
    glob() returns directory order, which is not stable across machines.
    """
    parquet = sorted(glob.glob(os.path.join(channel_dir, "**", "*.parquet"), recursive=True))
    csv = sorted(glob.glob(os.path.join(channel_dir, "**", "*.csv"), recursive=True))

    if parquet:
        df = pd.concat([pd.read_parquet(p) for p in parquet], ignore_index=True)
    elif csv:
        df = pd.concat([pd.read_csv(c) for c in csv], ignore_index=True)
    else:
        raise FileNotFoundError(
            f"No .parquet or .csv files under {channel_dir}. SageMaker stages the "
            f"training channel here before the container starts; an empty channel "
            f"means the pipeline's TrainingInput S3 URI is wrong or the prefix is empty."
        )

    # The offline store is append-only and the Glue job can be re-run, so the
    # channel may carry more than one record per customer. Deduplicate exactly
    # the way the Athena query does, then impose the same total order.
    if "customer_id" in df.columns:
        if "event_time" in df.columns:
            df = df.sort_values(["customer_id", "event_time"]).drop_duplicates(
                subset=["customer_id"], keep="last")
        else:
            df = df.drop_duplicates(subset=["customer_id"], keep="last")
        df = df.sort_values("customer_id").reset_index(drop=True)

    missing = [c for c in FEATURE_COLUMNS + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Training channel missing required columns: {missing}")
    if SLICE_COLUMN not in df.columns:
        raise ValueError(
            f"Training channel missing '{SLICE_COLUMN}'. The reference trainer "
            f"evaluates per-tier slices and cannot run without it.")

    return df


def normalise_metrics(metrics: dict) -> dict:
    """Emit the metric vocabulary the promotion gate actually reads.

    There are currently two names for the same four numbers. train_reference.py
    says baseline_auc_roc / auc_lift_over_baseline / auc_lift_ci_low|high;
    churn_training_skeleton.py and tests/test_model.py say baseline_auc /
    auc_vs_baseline / lift_ci_low|high. tests/test_model.py is the gate, so its
    names have to be present or every build fails with "lift_ci_low missing"
    on a model that is fine.

    This shim writes both. That is deliberate but it is a workaround, not a
    fix: the real fix is one vocabulary across all three artifacts, which is
    a rename with doc fallout and belongs in its own change.
    """
    out = dict(metrics)
    for canonical, alias in (
        ("baseline_auc_roc", "baseline_auc"),
        ("auc_lift_over_baseline", "auc_vs_baseline"),
        ("auc_lift_ci_low", "lift_ci_low"),
        ("auc_lift_ci_high", "lift_ci_high"),
    ):
        if canonical in out:
            out[alias] = out[canonical]
    return out


def _check_outputs(model, metrics):
    import xgboost as xgb
    if not isinstance(model, xgb.Booster):
        raise SystemExit(
            f"churn_model.train() returned a {type(model).__name__}, not an "
            f"xgboost.Booster. Use xgb.train(...), or return clf.get_booster() "
            f"if you trained an XGBClassifier.")
    if not isinstance(metrics, dict):
        raise SystemExit("churn_model.train() must return (model, metrics_dict).")
    need = ["precision_top10", "recall_top10", "lift_ci_low", "lift_ci_high"]
    absent = [k for k in need if metrics.get(k) is None]
    if absent:
        raise SystemExit(
            f"Metrics from churn_model.train() are missing {absent}. The CI "
            f"gate in tests/test_model.py reads all four. Your Lab 3 "
            f"evaluate_model() already computes them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-channel",
                    default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    ap.add_argument("--output-data-dir",
                    default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    ap.add_argument("--commit-sha", default=os.environ.get("COMMIT_SHA", "unknown"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    print(f"Reading training channel: {args.train_channel}")
    df = read_channel(args.train_channel)
    print(f"  {len(df):,} rows, {df[LABEL].mean():.1%} positive")

    model, metrics = train(df, seed=args.seed)
    metrics = normalise_metrics(metrics)
    _check_outputs(model, metrics)
    metrics["commit_sha"] = args.commit_sha

    # Save format is chosen by FILE EXTENSION, and getting it wrong produces an
    # artifact nothing modern can open. `.xgb` is not a recognised extension, so
    # XGBoost 1.7 -- the version in this training container -- falls back to the
    # legacy BINARY format, which was deprecated in 1.6 and REMOVED in 3.1:
    #     XGBoostError: Failed to load model ... The binary format has been
    #     deprecated in 1.6 and removed in 3.1, use UBJ or JSON instead
    # So an artifact saved as model.xgb here cannot be loaded by the CI gate,
    # by a student on a current xgboost, or by anything else on 3.1+. Locally
    # on 3.2.0 the same call silently defaults to UBJSON, which is why this
    # never showed up until the model was built in the container (defect 71).
    #
    # `.ubj` is explicit and version-stable. The second copy is named
    # `xgboost-model` because that is the filename the SageMaker XGBoost
    # SERVING container looks for -- the same convention train_reference.py and
    # train_local.py already use. Without it the registered model trains and
    # registers fine and then cannot be deployed in Lab 5. Same bytes, two
    # names, because two different consumers each expect their own.
    # NOTHING BUT THE MODEL GOES IN args.model_dir.
    #
    # /opt/ml/model is tarred into model.tar.gz and handed to the serving
    # container, and the XGBoost container's algorithm mode ENUMERATES that
    # directory and tries to load every file in it as a booster. Leaving the
    # metrics JSON beside the model made the endpoint fail to boot:
    #     RuntimeError: Model /opt/ml/model/evaluation_metrics.json cannot be
    #     loaded: Pickle load error=invalid load key, '{'
    # gunicorn's worker then exits 3, /ping 502s, and the endpoint sits in
    # Creating for ~25 minutes before failing. The model trains, registers and
    # gates clean, and is simply not deployable -- which no test upstream of an
    # actual deploy can catch (defect 72).
    #
    # So: the servable artifact only, under the name the container expects.
    # Everything else goes to output_data_dir and S3.
    model_path = os.path.join(args.output_data_dir, "model.ubj")
    model.save_model(model_path)
    shutil.copyfile(model_path, os.path.join(args.model_dir, "xgboost-model"))
    print(f"Servable artifact: {args.model_dir}/xgboost-model (UBJ)")

    # tests/test_model.py requires metadata beside the artifact, and it is
    # right to: a registered model whose feature order is not recorded cannot
    # be safely served later. Hyperparameters are read back out of the trained
    # booster rather than restated here, so this cannot drift from whatever
    # churn_model.py actually used.
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL,
        "slice_column": SLICE_COLUMN,
        "hyperparameters": json.loads(model.save_config()),
        "evaluation_metrics": metrics,
        "commit_sha": args.commit_sha,
        "training_date": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(os.path.join(args.output_data_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Two copies on purpose. /opt/ml/model is tarred into model.tar.gz and
    # becomes the registered artifact; /opt/ml/output/data is uploaded
    # separately and is what the pipeline's RegisterModel step points
    # ModelMetrics at. Writing only the first buries the metrics inside a
    # tarball that the gate would have to unpack.
    with open(os.path.join(args.output_data_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # Third copy, and the one the Model Registry cites. /opt/ml/output/data is
    # uploaded as output.tar.gz, so there is no bare JSON in S3 for
    # ModelMetrics to point at unless the container puts one there itself.
    # pipeline_definition.py builds the identical URI with Join() and hands it
    # over as METRICS_S3_URI -- the two sides must agree or the registered
    # model cites an object that does not exist.
    metrics_uri = os.environ.get("METRICS_S3_URI")
    if metrics_uri:
        import boto3
        bucket, _, key = metrics_uri[len("s3://"):].partition("/")
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key,
                      Body=json.dumps(metrics, indent=2, sort_keys=True).encode(),
                      ContentType="application/json")
        # model_metadata.json rides along to the same prefix. It used to travel
        # inside model.tar.gz, which is precisely what broke serving; CI still
        # needs it, so S3 is where it goes.
        s3.put_object(Bucket=bucket,
                      Key=key.replace("evaluation_metrics.json", "model_metadata.json"),
                      Body=json.dumps(metadata, indent=2, sort_keys=True).encode(),
                      ContentType="application/json")
        print(f"Metrics + metadata uploaded to {os.path.dirname(metrics_uri)}/")
    else:
        print("METRICS_S3_URI unset — skipping upload (fine outside a pipeline run)")

    print(json.dumps({k: v for k, v in metrics.items()
                      if not isinstance(v, dict)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
