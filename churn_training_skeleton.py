"""
NorthStar Retail — Churn Prediction Training Script (Skeleton)
CS 401R Lab 3 Starter Kit

This script is the SageMaker XGBoost training entry point for the NorthStar
churn prediction model. It reads features from the SageMaker Feature Store,
trains an XGBoost classifier, tracks the run in a SageMaker MLflow App, and
registers the model in the Model Registry.

Experiment tracking needs two packages that are NOT in the training container:

    pip install mlflow sagemaker-mlflow

`sagemaker-mlflow` is the SigV4 auth plugin for arn:aws:sagemaker:... tracking
URIs. Install `mlflow` alone and you get a connection error that never mentions
credentials. See track_run() for the cost warning about the OTHER MLflow
product -- read it before you create anything.

Usage (local testing):
    python churn_training_skeleton.py \
        --feature-group-name northstar-dev-customer-features \
        --artifacts-bucket northstar-dev-data-<account> \
        --max-depth 6 --eta 0.1 --num-round 200 \
        --mlflow-arn arn:aws:sagemaker:us-east-1:<account>:mlflow-app/app-XXXX \
        --run-name xgb-depth6-eta0.1

Usage (SageMaker training job — parameters passed via hyperparameter dict):
    Configured via the SageMaker Python SDK Estimator.
"""

import argparse
import json
import os
import pickle
import tarfile
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd

# SageMaker imports — available in SageMaker training containers.
#
# The two failure modes below are different problems and used to produce the
# same message. sagemaker 3.x REMOVED the feature_store package, so on a 3.x
# install the SDK imports fine and only the second line fails. Reporting that
# as "sagemaker not available" sends you looking for a missing install that is
# not missing. Pin sagemaker<3.0.0 (see requirements.txt, defect 55).
try:
    import sagemaker
except ImportError:
    sagemaker = None
    SAGEMAKER_AVAILABLE = False
    print("WARNING: sagemaker not installed — running in local test mode")
else:
    try:
        from sagemaker.feature_store.feature_group import FeatureGroup
        from sagemaker.session import Session
        SAGEMAKER_AVAILABLE = True
    except ImportError:
        SAGEMAKER_AVAILABLE = False
        print("WARNING: sagemaker is installed but has no feature_store module. "
              "This is sagemaker 3.x, which dropped it. Run "
              "`pip install 'sagemaker>=2.200.0,<3.0.0'` — the Feature Store "
              "path in this script cannot work on 3.x.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    raise ImportError("xgboost is required: pip install xgboost")

from sklearn.metrics import (auc, classification_report, confusion_matrix,
                              precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split

# ── Argument Parsing ───────────────────────────────────────────────────────────
# SageMaker passes hyperparameters as CLI arguments.

def parse_args():
    parser = argparse.ArgumentParser(description="NorthStar churn prediction training")

    # Data configuration
    # Names come from Lab 2's Terraform: `terraform output -raw feature_group_name`
    # and `terraform output -raw s3_bucket_name`. Do not invent new ones.
    parser.add_argument("--feature-group-name", type=str,
                        default="northstar-dev-customer-features",
                        help="SageMaker Feature Store feature group name (Lab 2 output)")
    # These dates bound the INGEST batch, not the feature window. Lab 2 stamps
    # event_time with the wall-clock time its Glue job ran (time.time()), not
    # with the feature cutoff T = 2026-04-01 -- T was already enforced when the
    # features were computed. A window that ends before your Lab 2 run (the old
    # default here was 2026-06-01) matches zero rows. These defaults match the
    # reference implementation and cover any Lab 2 run during the semester.
    parser.add_argument("--training-start-date", type=str, default="2026-01-01",
                        help="Earliest Lab 2 ingest date to include (YYYY-MM-DD)")
    parser.add_argument("--training-end-date", type=str, default="2027-01-01",
                        help="Latest Lab 2 ingest date to include (YYYY-MM-DD)")
    parser.add_argument("--artifacts-bucket", type=str, required=True,
                        help="Your Lab 1 data bucket, northstar-dev-data-<account>. "
                             "Athena results and model artifacts go under its artifacts/ "
                             "prefix, the only prefix your MLEngineer role can write")
    parser.add_argument("--local-data-path", type=str, default=None,
                        help="Path to local feature CSV for development testing (bypasses Feature Store)")

    # XGBoost hyperparameters
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--num-round", type=int, default=200)
    parser.add_argument("--min-child-weight", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--scale-pos-weight", type=float, default=None,
                        help=(
                            "XGBoost positive-class weight. Leave unset and it is computed "
                            "from YOUR training split as negatives/positives, which is what "
                            "you want. The reference 10,000-customer dataset is 22.0%% "
                            "churners, giving (1 - 0.220) / 0.220 = 3.545. Passing a value "
                            "tuned for a different class balance will quietly skew your "
                            "precision/recall trade-off, so only override this deliberately."
                        ))

    # Feature-set controls. Task 1 requires results with and without
    # churn_risk_score if it is used, and Task 3 asks which features carried
    # signal and which were decorative -- answered by dropping groups of
    # features and measuring the change. Both are logged to MLflow.
    parser.add_argument("--include-risk-score", action="store_true",
                        help="Add churn_risk_score (a recency heuristic) as an input")
    parser.add_argument("--drop-features", type=str, default="",
                        help="Comma-separated FEATURE_COLUMNS to leave out (ablation). "
                             "days_since_last_purchase cannot be dropped: the recency "
                             "baseline needs it.")

    # Experiment tracking (Task 1, 5 points)
    parser.add_argument("--mlflow-arn", type=str,
                        default=os.environ.get("MLFLOW_APP_ARN"),
                        help=(
                            "ARN of your SageMaker MLflow APP, e.g. "
                            "arn:aws:sagemaker:us-east-1:<account>:mlflow-app/app-XXXX. "
                            "Create it once with `aws sagemaker create-mlflow-app`. "
                            "This is NOT an MLflow Tracking Server -- see the warning "
                            "on track_run() below. Omit to skip tracking."
                        ))
    parser.add_argument("--mlflow-experiment", type=str, default="northstar-churn",
                        help="MLflow experiment name; runs group under this")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run. Make it descriptive -- 'run3' tells "
                             "you nothing three weeks later.")
    parser.add_argument("--run-description", type=str, default="",
                        help="The question this run answers (stored as the MLflow run description)")

    # Model Registry. Only the FINAL model is registered (Task 1); exploratory
    # runs are tracked in MLflow but kept out of the group Lab 5 deploys from.
    parser.add_argument("--register", action="store_true",
                        help="Register this run in the Model Registry as PendingManualApproval")
    parser.add_argument("--inference-image", type=str,
                        default="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
                        help="Serving container recorded in the model package "
                             "(SageMaker built-in XGBoost, us-east-1)")

    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--output-data-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))

    return parser.parse_args()


# ── Feature Store Loading ──────────────────────────────────────────────────────

# These are exactly the features your Lab 2 pipeline wrote to the Feature
# Group. If a name here does not match your feature group, the Athena query
# below will fail - check `aws sagemaker describe-feature-group` first.
#
# Note what is NOT in this list: churn_label (the target) and customer_id
# (an identifier, not a signal). Including either as an input is leakage.
FEATURE_COLUMNS = [
    # Recency and tenure
    "days_since_last_purchase",
    "customer_tenure_days",
    # Frequency
    "purchase_frequency_30d",
    "purchase_frequency_90d",
    "purchase_frequency_180d",
    # Monetary
    "avg_order_value",
    "total_spend_90d",
    "total_lifetime_value",
    "avg_basket_size_6m",
    # Behavioural - these are the features that catch churners who still
    # look active on recency alone. Drop them and your model collapses
    # toward the recency-only baseline.
    "category_diversity_score",
    "online_to_store_ratio",
]

# churn_risk_score is available in the feature group but deliberately excluded
# here: it is a pure recency heuristic and doubles as the baseline you must
# beat. If you add it, report your metrics both with and without it.
BASELINE_COLUMN = "churn_risk_score"

# The recency-only baseline model uses just this one feature. Task 1 requires
# you to train it and show your full model beats it by a margin whose 95%
# confidence interval excludes zero. There is no fixed AUC-lift threshold: a
# ">= 0.03 lift" gate lived here until 2026-08-02 and was removed because 0.03
# is smaller than the metric's own run-to-run standard deviation, so it failed
# on 21% of splits regardless of model quality. See EVAL_THRESHOLDS below.
BASELINE_FEATURE = "days_since_last_purchase"

LABEL_COLUMN = "churn_label"

# Not a feature. This is the column Task 1's slice evaluation groups by, so it
# has to come back from the query even though it never enters the model. Keep it
# out of FEATURE_COLUMNS: loyalty_tier correlates with spend, and feeding it in
# both leaks and makes the per-tier fairness check circular.
SLICE_COLUMN = "loyalty_tier"


def resolve_feature_columns(args) -> list[str]:
    """This run's model inputs: FEATURE_COLUMNS adjusted by --drop-features
    and --include-risk-score."""
    drop = [c.strip() for c in args.drop_features.split(",") if c.strip()]
    unknown = [c for c in drop if c not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"--drop-features names unknown columns: {unknown}")
    if BASELINE_FEATURE in drop:
        raise ValueError(f"{BASELINE_FEATURE} cannot be dropped: the recency baseline needs it")
    cols = [c for c in FEATURE_COLUMNS if c not in drop]
    if args.include_risk_score:
        cols.append(BASELINE_COLUMN)
    return cols


def load_features_from_feature_store(feature_group_name: str,
                                     start_date: str,
                                     end_date: str,
                                     artifacts_bucket: str) -> pd.DataFrame:
    """
    Load features from the SageMaker Feature Store offline store using Athena.

    Args:
        feature_group_name: your Lab 2 feature group
        start_date, end_date: ISO dates (YYYY-MM-DD) bounding the event_time range
        artifacts_bucket: S3 bucket that Athena writes its result set to. Athena
            will not run a query without an output location, so this has to be
            threaded in from --artifacts-bucket rather than assumed.

    Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-use-with-studio.html
    """
    if not SAGEMAKER_AVAILABLE:
        raise RuntimeError("SageMaker SDK not available - use --local-data-path for local testing")

    # event_time was written as Fractional (epoch seconds) by the Lab 2 job, so
    # the date bounds must be converted before comparison. Passing ISO strings
    # straight into the query fails with TYPE_MISMATCH rather than returning
    # rows, so at least the failure is loud.
    start_epoch = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    end_epoch = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()

    session = Session()
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=session)

    # Two things in the query below are easy to get wrong, so read them:
    #
    # 1. event_time is Fractional - Unix epoch SECONDS, not an ISO string.
    #    Comparing it to '2026-04-01' fails the query outright with:
    #      TYPE_MISMATCH: Cannot check if double is BETWEEN varchar(10)...
    #    Convert your date bounds to epoch seconds before substituting them.
    #
    # 2. The offline store is append-only. Re-running the Lab 2 feature job
    #    writes a SECOND record for every customer, and a naive SELECT then
    #    returns duplicate customers with stale feature values. Deduplicate
    #    with a window function keyed on customer_id, keeping the most recent
    #    event_time. Filtering on a global MAX(write_time) does NOT work -
    #    it keeps only the customers written in the final microsecond.
    #
    #    The offline store also soft-deletes: rows carry is_deleted, and
    #    deleted records must be excluded.
    #
    # 3. The outer SELECT must ORDER BY customer_id, and this is the one people
    #    skip because it looks cosmetic. It is not. Athena parallelises the scan
    #    across the offline store's Parquet objects and returns rows in whatever
    #    order the splits happen to finish, which varies from run to run.
    #    train_test_split(random_state=42) is deterministic only for a GIVEN row
    #    order -- so without ORDER BY, the identical data produces a different
    #    train/test split, and therefore different metrics, on every single run.
    #
    #    This was measured, not theorised: four runs on byte-identical data
    #    produced AUC between 0.7276 and 0.7431, and a Platinum-slice AUC
    #    between 0.430 and 0.700. An entire "the model is worse than random on
    #    your best customers" finding turned out to be an artefact of row order.
    #
    #    The rn = 1 filter guarantees customer_id is unique here, so ordering on
    #    it is a TOTAL order and the pipeline becomes reproducible.
    #
    #    If your metrics move between runs and your data did not, this is why.

    #    loyalty_tier is selected but is NOT in FEATURE_COLUMNS, and that is
    #    deliberate. It is not an input to the model - it is the column Task 1's
    #    slice evaluation groups by. Keep it out of the feature matrix (see
    #    preprocess_features, which selects FEATURE_COLUMNS explicitly) and
    #    carry it alongside X and y through the split so each row's prediction
    #    can still be attributed to a tier:
    #
    #        X_train, X_val, y_train, y_val, tier_train, tier_val = \
    #            train_test_split(X, y, df[SLICE_COLUMN], test_size=0.30,
    #                             random_state=42, stratify=y)
    #
    #    Split X and y without it and you cannot line the tiers back up.

    query = feature_group.athena_query()
    query_string = f"""
        WITH ranked AS (
            SELECT
                customer_id,
                {", ".join(FEATURE_COLUMNS)},
                {LABEL_COLUMN},
                {SLICE_COLUMN},
                event_time,
                ROW_NUMBER() OVER (
                    PARTITION BY customer_id
                    ORDER BY event_time DESC, write_time DESC
                ) AS rn
            FROM "{query.table_name}"
            WHERE event_time BETWEEN {start_epoch} AND {end_epoch}
              AND NOT is_deleted
        )
        SELECT
            customer_id,
            {", ".join(FEATURE_COLUMNS)},
            {LABEL_COLUMN},
            {SLICE_COLUMN}
        FROM ranked
        WHERE rn = 1
        ORDER BY customer_id
    """

    query.run(query_string=query_string,
              output_location=f"s3://{artifacts_bucket}/artifacts/athena-results/")
    query.wait()
    query_id = query.get_query_execution()['QueryExecution']['QueryExecutionId']
    print(f"Athena QueryExecutionId: {query_id}")
    df = query.as_dataframe()
    df.attrs["athena_query_id"] = query_id   # logged to MLflow in track_run()
    print(f"  {len(df):,} rows, {df['customer_id'].nunique():,} unique customers")
    return df


def load_features_local(path: str) -> pd.DataFrame:
    """Load features from a local CSV file (development mode)."""
    print(f"Loading features from local file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Local data missing required columns: {missing}")
    # Not fatal - the model trains fine without it - but Task 1's slice
    # evaluation cannot be done at all, so fail the task rather than the run.
    if SLICE_COLUMN not in df.columns:
        print(f"WARNING: '{SLICE_COLUMN}' not in local data — Task 1 slice "
              f"evaluation will not be possible with this file")
    return df


# ── Feature Preprocessing ──────────────────────────────────────────────────────

def preprocess_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean and prepare features for XGBoost training.
    Returns (X, y) arrays.
    """
    print(f"Preprocessing {len(df):,} samples...")

    # Drop rows where label is missing
    df = df.dropna(subset=[LABEL_COLUMN])

    # Missing-value strategy: sentinel -1.
    #
    # In practice this branch is inert: Lab 2's producer-side quality gates
    # assert zero nulls in every feature before anything reaches the Feature
    # Store, and the count printed below confirms it on every run. If a null
    # ever did arrive, -1 is a safe sentinel for THIS feature set because
    # every feature is non-negative by construction (days, counts, dollars,
    # ratios in [0, 1]), so -1 can never collide with a real value and a tree
    # can isolate it with a single split. Median imputation was rejected
    # because it would hide missingness the model might find informative.
    n_null = int(df[FEATURE_COLUMNS].isna().sum().sum())
    print(f"  Null feature values: {n_null} (filled with sentinel -1)")
    X = df[FEATURE_COLUMNS].fillna(-1).values  # Sentinel -1 for missing
    y = df[LABEL_COLUMN].values.astype(int)

    print(f"  Positive class rate: {y.mean():.1%}")
    print(f"  Feature matrix shape: {X.shape}")

    return X, y


# ── Model Training ─────────────────────────────────────────────────────────────

def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                args) -> xgb.Booster:
    """Train the XGBoost churn model."""

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS)

    # Class imbalance, computed from the training split rather than hardcoded.
    # On the reference dataset this lands at 3.545.
    spw = (args.scale_pos_weight if args.scale_pos_weight is not None
           else float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))

    # Record the RESOLVED value so track_run() can log what was actually used.
    # Logging args.scale_pos_weight instead would log None on every run that
    # let it compute itself -- which is most of them -- and your MLflow
    # comparison would be missing the one parameter most likely to explain a
    # precision/recall difference between runs.
    args.resolved_scale_pos_weight = spw

    params = {
        # eval_metric ORDER MATTERS. XGBoost early-stops on the LAST metric in
        # this list, not the first. This used to read ["auc", "logloss"], which
        # silently made logloss the early-stopping criterion: under
        # scale_pos_weight the reweighted logloss keeps improving long after
        # ranking quality peaks, so training ran the full 200 rounds instead of
        # stopping near round 31 and validation AUC came out 0.7603 instead of
        # 0.7822. Measured across 50 splits, that cost the model the promotion
        # gate on 28% of them -- a student doing everything right was told their
        # features did not beat recency. Keep "auc" last (defect 49).
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": args.max_depth,
        "eta": args.eta,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": spw,
        "seed": 42,
        "verbosity": 1,
    }

    print(f"\nTraining XGBoost with params: {json.dumps(params, indent=2)}")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=20,
        verbose_eval=25,
    )

    # Same pattern as resolved_scale_pos_weight: record where early stopping
    # landed so the MLflow comparison shows it.
    args.best_iteration = model.best_iteration
    print(f"Best iteration: {model.best_iteration} (validation AUC {model.best_score:.4f})")

    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

EVAL_THRESHOLDS = {
    "precision_top10":   0.50,   # Minimum precision @ top 10% scored customers
    "recall_top10":      0.25,   # Minimum recall @ top 10% scored customers
    # Note the recall ceiling: with ~22% positives, targeting the top 10% of
    # customers caps achievable recall near 0.45. Hitting 0.25 means capturing
    # roughly half of what is reachable within that contact budget.
}

# There is deliberately NO absolute AUC threshold. One used to live here
# (auc_roc >= 0.72) and it was removed on 2026-08-02: measured across 200
# random train/test splits of the same data, the reference model fell below
# 0.72 on 58% of them. A gate the reference clears by luck of the shuffle
# grades your random seed, not your model. Report AUC; do not gate on it.
#
# The baseline gate is an INTERVAL, not a threshold: the 95% CI on
# (your AUC - recency-only baseline AUC) must exclude zero.
# BASELINE_FEATURE is defined once, near FEATURE_COLUMNS above.
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42


def train_recency_baseline(X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray) -> np.ndarray:
    """Train the recency-only baseline and score the validation split.

    The baseline is a model over `days_since_last_purchase` ALONE, trained on
    the same training rows and scored on the same validation rows as the full
    model. That is what makes the comparison fair.

    It is deliberately NOT a constant predictor. A constant prediction has an
    AUC of exactly 0.5 by construction, so "beating" it proves only that your
    model is better than a coin flip. The question Lab 3 asks is whether your
    feature engineering beats the rule the business already has for free:
    "contact whoever has not purchased in a while."
    """
    i = FEATURE_COLUMNS.index(BASELINE_FEATURE)
    dtr = xgb.DMatrix(X_train[:, [i]], label=y_train, feature_names=[BASELINE_FEATURE])
    dva = xgb.DMatrix(X_val[:, [i]], feature_names=[BASELINE_FEATURE])
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 4, "eta": 0.1, "subsample": 0.9,
        "colsample_bytree": 0.9, "scale_pos_weight": spw, "seed": 42,
    }
    return xgb.train(params, dtr, num_boost_round=200).predict(dva)


def bootstrap_lift_ci(y_true: np.ndarray,
                      proba: np.ndarray,
                      baseline_proba: np.ndarray,
                      n: int = BOOTSTRAP_N,
                      seed: int = BOOTSTRAP_SEED,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for (model AUC - baseline AUC), by resampling the val set.

    PAIRED: each replicate resamples row indices once and scores BOTH models on
    those same rows, so the interval is on the difference and the correlation
    between the two models is preserved. Resampling them independently breaks
    the pairing and inflates the interval, which would make a real improvement
    look inconclusive.

    Replicates whose resample happens to be single-class are skipped, because
    AUC is undefined there.

    The seed is fixed so the gate is reproducible: the same split must always
    produce the same verdict.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    idx = np.arange(len(y_true))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        ys = y_true[s]
        if len(np.unique(ys)) < 2:
            continue
        diffs.append(roc_auc_score(ys, proba[s]) - roc_auc_score(ys, baseline_proba[s]))
    if len(diffs) < n // 2:
        raise ValueError(
            "Too few usable bootstrap replicates to form a confidence interval. "
            "Your validation set is too small or too imbalanced to support this gate."
        )
    d = np.sort(diffs)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def evaluate_model(model: xgb.Booster,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   baseline_proba: np.ndarray) -> dict:
    """
    Evaluate the trained model against required thresholds.
    Returns a metrics dict. Raises ValueError if thresholds are not met.
    """
    dval = xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS)
    y_pred_proba = model.predict(dval)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    auc_roc = roc_auc_score(y_val, y_pred_proba)

    # Precision and recall @ top 10%
    top10_threshold = np.percentile(y_pred_proba, 90)
    top10_mask = y_pred_proba >= top10_threshold
    precision_top10 = y_val[top10_mask].mean() if top10_mask.sum() > 0 else 0.0
    recall_top10 = y_val[top10_mask].sum() / y_val.sum() if y_val.sum() > 0 else 0.0

    # Baseline: a model trained on days_since_last_purchase ALONE.
    #
    # This used to be `np.full_like(y_pred_proba, y_val.mean())` - a constant
    # prediction, whose AUC is exactly 0.5 by definition. Beating a coin flip
    # is not evidence that your feature engineering did anything, and it is not
    # what Lab 3 asks for. The comparison that matters is against recency,
    # because recency is the rule the business already has for free.
    #
    baseline_auc = roc_auc_score(y_val, baseline_proba)
    auc_vs_baseline = auc_roc - baseline_auc

    lift_ci_low, lift_ci_high = bootstrap_lift_ci(
        y_val, y_pred_proba, baseline_proba)

    metrics = {
        "auc_roc": round(float(auc_roc), 4),
        "precision_top10": round(float(precision_top10), 4),
        "recall_top10": round(float(recall_top10), 4),
        "auc_vs_baseline": round(float(auc_vs_baseline), 4),
        "baseline_auc": round(float(baseline_auc), 4),
        "lift_ci_low": round(float(lift_ci_low), 4),
        "lift_ci_high": round(float(lift_ci_high), 4),
        "positive_rate_val": round(float(y_val.mean()), 4),
        "n_val_samples": int(len(y_val)),
        "eval_timestamp": datetime.utcnow().isoformat(),
    }

    print("\n── Evaluation Results ──────────────────────────────")
    failures = []
    for metric, threshold in EVAL_THRESHOLDS.items():
        value = metrics[metric]
        status = "✓ PASS" if value >= threshold else "✗ FAIL"
        print(f"  {metric:25s}: {value:.4f}  (threshold: {threshold:.4f})  {status}")
        if value < threshold:
            failures.append(f"{metric}={value:.4f} < {threshold}")

    # The baseline gate: the interval must exclude zero.
    ci_status = "✓ PASS" if lift_ci_low > 0 else "✗ FAIL"
    print(f"  {'auc_lift 95% CI':25s}: [{lift_ci_low:.4f}, {lift_ci_high:.4f}]"
          f"  (must exclude 0)  {ci_status}")
    if lift_ci_low <= 0:
        failures.append(
            f"auc_lift 95% CI [{lift_ci_low:.4f}, {lift_ci_high:.4f}] includes "
            f"zero - no evidence the model beats the recency-only baseline")

    if failures:
        raise ValueError(
            f"Model failed evaluation thresholds: {'; '.join(failures)}. "
            f"Do not promote to Model Registry."
        )

    print("  All thresholds passed ✓")

    # Slice evaluation lives in evaluate_slices() below. main() calls it with
    # the tier array split alongside X and y, which this function does not see.

    return metrics


# ── Slice Evaluation ───────────────────────────────────────────────────────────

# A tier needs at least this many positives in the test set before its AUC or
# recall supports a claim. Lab 3 documents a Platinum slice with ~2 positives
# whose AUC swung between 0.00 and 1.00 on the random split alone; the
# bootstrap CI reported beside each AUC shows the same thing quantitatively.
MIN_SLICE_POSITIVES = 30


def evaluate_slices(model: xgb.Booster,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    slice_column: pd.Series) -> dict:
    """
    Evaluate model performance by slice (e.g., loyalty_tier, age_band).
    Flag slices where recall@10% drops more than 10pp below aggregate.

    Recall is measured at the GLOBAL top-10% cutoff: the retention budget is
    one list of the 10% highest-risk customers, so the question per tier is
    "what share of this tier's churners make that list?"
    """
    slice_column = pd.Series(np.asarray(slice_column))
    dval = xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS)
    y_pred_proba = model.predict(dval)
    flagged = y_pred_proba >= np.percentile(y_pred_proba, 90)
    aggregate_recall = flagged[y_val == 1].mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    slice_results = {}
    for slice_value in ["Bronze", "Silver", "Gold", "Platinum"]:
        mask = (slice_column == slice_value).values
        y_s, p_s, f_s = y_val[mask], y_pred_proba[mask], flagged[mask]
        positives = int(y_s.sum())
        auc_s = roc_auc_score(y_s, p_s) if 0 < positives < len(y_s) else float("nan")
        boot = []
        for _ in range(1000):
            s = rng.integers(0, len(y_s), len(y_s))
            if 0 < y_s[s].sum() < len(s):
                boot.append(roc_auc_score(y_s[s], p_s[s]))
        recall_s = f_s[y_s == 1].mean() if positives else float("nan")
        slice_results[slice_value] = {
            "n": int(mask.sum()),
            "positives": positives,
            "churn_rate": round(float(y_s.mean()), 4),
            "auc": round(float(auc_s), 4),
            "auc_ci_low": round(float(np.quantile(boot, 0.025)), 4) if boot else float("nan"),
            "auc_ci_high": round(float(np.quantile(boot, 0.975)), 4) if boot else float("nan"),
            "recall_top10": round(float(recall_s), 4),
            "share_flagged": round(float(f_s.mean()), 4),
            # Flag if recall < aggregate_recall - 0.10
            "recall_gap_flag": bool(recall_s < aggregate_recall - 0.10),
            "too_small_to_conclude": positives < MIN_SLICE_POSITIVES,
        }

    print(f"\n── Slice Evaluation (loyalty_tier; recall at global top-10% cutoff, "
          f"aggregate {aggregate_recall:.4f}) ──")
    print(pd.DataFrame(slice_results).T.to_string())
    for tier, r in slice_results.items():
        if r["too_small_to_conclude"]:
            print(f"  ⚠ {tier}: only {r['positives']} positives -- too few to conclude from")
        if r["recall_gap_flag"]:
            print(f"  ⚠ {tier}: recall {r['recall_top10']:.4f} is more than 10pp below aggregate")

    return slice_results


def save_evidence_plots(model: xgb.Booster,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        output_dir: str) -> None:
    """Confusion matrix at the chosen threshold and feature importance (Task 1 evidence).

    The chosen threshold is the top-10% cutoff, the operating point the
    retention team uses (and the one precision/recall@10% are measured at).
    Importance is by gain, the average loss reduction per split.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_pred_proba = model.predict(xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS))
    threshold = float(np.percentile(y_pred_proba, 90))
    cm = confusion_matrix(y_val, (y_pred_proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["retain (0)", "churn (1)"])
    ax.set_yticks([0, 1], ["retain (0)", "churn (1)"])
    ax.set_xlabel("Predicted (top 10% flagged as churn)")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix, test set n={len(y_val):,}\n"
                 f"threshold = top-10% cutoff (p ≥ {threshold:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion-matrix.png"), dpi=150)
    plt.close(fig)

    imp = pd.Series(model.get_score(importance_type="gain")).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color="#2a6f97")
    ax.set_xlabel("Average gain per split")
    ax.set_title("XGBoost feature importance (gain)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature-importance.png"), dpi=150)
    plt.close(fig)

    print(f"\nConfusion matrix at top-10% threshold {threshold:.4f}: "
          f"TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")
    print("Feature importance (gain): " +
          ", ".join(f"{k}={v:.2f}" for k, v in imp.sort_values(ascending=False).items()))


# ── Experiment Tracking (MLflow App) ───────────────────────────────────────────

def track_run(args, metrics: dict, scale_pos_weight: float) -> None:
    """
    Log this training run to your SageMaker MLflow App.

    Worth 5 points in Task 1: >=3 runs, each with logged params AND metrics,
    retrievable via mlflow.search_runs.

    ###########################################################################
    #  THERE ARE TWO MLflow PRODUCTS ON SAGEMAKER. USE THE APP.               #
    #                                                                         #
    #    CreateMlflowApp            serverless   NO ADDITIONAL CHARGE   <-- ok #
    #    CreateMlflowTrackingServer $0.60/hour   until you delete it    <-- NO #
    #                                                                         #
    #  $0.60/hr breaches the entire $10 course budget in 16.7 hours and costs #
    #  about $43 over a weekend -- more per hour than any endpoint in this    #
    #  course. It is not an endpoint, so "did I delete my endpoints?" will    #
    #  not find it. Most tutorials describe the Tracking Server because it    #
    #  shipped first. If anything asks you to choose a size (Small/Medium),   #
    #  you are on the wrong product.                                          #
    ###########################################################################

    Two things that fail in ways the error does not explain:

      1. `create-mlflow-app` postdates many installed AWS CLIs. An old CLI
         says "Invalid choice: 'create-mlflow-app'", which reads like a typo
         rather than a version problem. Check `aws --version` and upgrade.

      2. You need BOTH `mlflow` and `sagemaker-mlflow` installed. The second
         is the SigV4 auth plugin for arn:aws:sagemaker:... tracking URIs.
         Without it you get a connection error that never mentions credentials.
    """
    if not args.mlflow_arn:
        print("\n(no --mlflow-arn given; skipping experiment tracking)")
        return

    import mlflow

    mlflow.set_tracking_uri(args.mlflow_arn)   # the App ARN, verbatim
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=args.run_name, description=args.run_description) as run:
        mlflow.log_params({
            "max_depth": args.max_depth,
            "eta": args.eta,
            "num_round": args.num_round,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "xgboost_version": xgb.__version__,   # see below -- this matters
            # The feature set is the other axis this experiment varies.
            "n_features": len(FEATURE_COLUMNS),
            "include_risk_score": args.include_risk_score,
            "dropped_features": args.drop_features or "none",
            "best_iteration": getattr(args, "best_iteration", None),
            "athena_query_id": getattr(args, "athena_query_id", None) or "local",
            "registered": args.register,
        })
        mlflow.log_metrics({
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        })
        mlflow.set_tag("features", ",".join(FEATURE_COLUMNS))

    print(f"\n✓ MLflow run logged: {args.run_name} (run_id {run.info.run_id})")

    # LOG THE XGBOOST VERSION AS A PARAM. Lab 3 documents that identical data
    # and an identical split produce different metrics on XGBoost 3.2.0 vs 1.7
    # (the SageMaker training container is 1.7). If you train some runs locally
    # and some through SageMaker and do not record the version, you will end up
    # comparing two things that were never comparable and cannot tell why.
    #
    # THREE RUNS IS THE FLOOR, NOT THE GOAL. Three runs with identical
    # hyperparameters and different seeds is the same experiment three times;
    # it earns 2 of 5. Vary something you can defend, and be able to say what
    # the variation did to the metrics.


# ── Model Saving & Registry ────────────────────────────────────────────────────

def save_and_register_model(model: xgb.Booster,
                             metrics: dict,
                             args) -> None:
    """
    Save the model artifact and register it in SageMaker Model Registry.
    Status is set to PendingManualApproval — human approval required before deployment.
    """
    os.makedirs(args.model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(args.model_dir, "model.xgb")
    model.save_model(model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Save feature metadata alongside model
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "hyperparameters": {
            "max_depth": args.max_depth,
            "eta": args.eta,
            "num_round": args.num_round,
        },
        "evaluation_metrics": metrics,
        "training_date": datetime.utcnow().isoformat(),
    }
    with open(os.path.join(args.model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Only the final model is registered (--register). Exploratory runs stay
    # in MLflow and out of the group Lab 5 deploys from.
    if not args.register:
        print("  (exploratory run: not registered; pass --register for the final model)")
        return

    # The model package group must be "northstar-churn-models" -- Lab 5 deploys
    # from that group by name, so any other name breaks Lab 5.
    group = "northstar-churn-models"
    sm_client = boto3.client("sagemaker")
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group)
    except sm_client.exceptions.ClientError:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription="NorthStar churn prediction (XGBoost on Lab 2 Feature Store features)")

    # The SageMaker XGBoost serving container loads a file named
    # "xgboost-model" from model.tar.gz, so the artifact is packaged under that
    # name. artifacts/ is the only prefix the MLEngineer role can write.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tar_path = os.path.join(args.model_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="xgboost-model")
    key = f"artifacts/models/churn/{args.run_name or 'run'}-{stamp}/model.tar.gz"
    boto3.client("s3").upload_file(tar_path, args.artifacts_bucket, key)
    print(f"  uploaded s3://{args.artifacts_bucket}/{key}")

    # Include the evaluation metrics in the model card / description.
    # Status must be "PendingManualApproval" — never "Approved" from this script.
    response = sm_client.create_model_package(
        ModelPackageGroupName=group,
        ModelPackageDescription=(
            f"Churn model trained {datetime.utcnow().date()} (run {args.run_name}, "
            f"XGBoost {xgb.__version__}). AUC {metrics['auc_roc']}, "
            f"P@10% {metrics['precision_top10']}, R@10% {metrics['recall_top10']}, "
            f"lift over recency {metrics['auc_vs_baseline']} "
            f"(95% CI [{metrics['lift_ci_low']}, {metrics['lift_ci_high']}]). "
            f"Features: {', '.join(FEATURE_COLUMNS)}."
        )[:1024],
        InferenceSpecification={
            "Containers": [{"Image": args.inference_image,
                            "ModelDataUrl": f"s3://{args.artifacts_bucket}/{key}"}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties={
            k: str(v) for k, v in metrics.items()
        },
    )
    print(f"✓ Registered {response['ModelPackageArn']} as PendingManualApproval")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global FEATURE_COLUMNS
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    print("=" * 60)
    print("NorthStar Churn Prediction — Training Script")
    print("=" * 60)

    # This run's inputs. Resolved before anything reads FEATURE_COLUMNS, so the
    # query, the DMatrix feature names and the saved metadata all agree.
    FEATURE_COLUMNS = resolve_feature_columns(args)
    print(f"Model inputs ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")

    # 1. Load features
    if args.local_data_path:
        df = load_features_local(args.local_data_path)
    else:
        df = load_features_from_feature_store(
            args.feature_group_name,
            args.training_start_date,
            args.training_end_date,
            args.artifacts_bucket,
        )
    args.athena_query_id = df.attrs.get("athena_query_id")

    # 2. Preprocess
    X, y = preprocess_features(df)

    # 3. Train/validation split (temporal split is better — split by customer_id here for simplicity)
    # test_size=0.30 matches the reference run and every published figure in
    # Lab 3 (6,999 train / 3,000 test on the 10k dataset). The bootstrap lift CI
    # is calibrated on a test set that size; a 0.20 split leaves 2,000 rows and
    # widens the interval enough to matter.
    #
    # The tier rides through the split as a third array so evaluate_slices()
    # can attribute each validation row to its tier; it never enters X.
    X_train, X_val, y_train, y_val, tier_train, tier_val = train_test_split(
        X, y, df[SLICE_COLUMN], test_size=0.30, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} | Validation: {len(X_val):,}")

    # 4. Train
    model = train_model(X_train, y_train, X_val, y_val, args)

    # 5. Evaluate
    baseline_proba = train_recency_baseline(X_train, y_train, X_val)
    metrics = evaluate_model(model, X_val, y_val, baseline_proba)
    slices = evaluate_slices(model, X_val, y_val, tier_val)
    pd.DataFrame(slices).T.rename_axis("tier").reset_index().to_csv(
        os.path.join(args.output_data_dir, "slice-evaluation.csv"), index=False)
    save_evidence_plots(model, X_val, y_val, args.output_data_dir)

    # 6. Save and register
    save_and_register_model(model, metrics, args)

    # 7. Track this run in your MLflow App (Task 1, 5 pts)
    track_run(args, metrics, getattr(args, "resolved_scale_pos_weight", None))

    # 8. Write metrics to output. SageMaker also collects these as job output;
    #    that is job lineage, NOT experiment tracking. It records that a job
    #    ran, not what you varied or why. Step 7 is the graded one.
    metrics_path = os.path.join(args.output_data_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete. AUC-ROC: {metrics['auc_roc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
