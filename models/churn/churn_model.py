"""Your Lab 3 churn model, packaged for the Lab 4 pipeline.

The pipeline's training job runs train_sagemaker.py (beside this file), which
imports this module and calls train(). Everything the job needs from this file
goes through three names:

    FEATURE_COLUMNS   list[str]   model input columns, in order
    LABEL             str         "churn_label"
    train(df, seed=42) -> (xgboost.Booster, dict)

`df` is the full labeled feature table: every column of the Lab 2 output,
including `loyalty_tier` for slice evaluation. It is already loaded; nothing
here queries Athena or the Feature Store.

The metrics dict carries the four numbers the CI gate in tests/test_model.py
reads (precision_top10, recall_top10, lift_ci_low, lift_ci_high), plus the rest
of what Lab 3 reported, and a `slices` dict with per-tier results that
tests/test_fairness.py reads.

The training container is the SageMaker XGBoost 1.7-1 image: it ships pandas,
numpy, scikit-learn, xgboost and boto3, but not mlflow or sagemaker. So there
are no such imports here. MLflow tracking stays in churn_training_skeleton.py.

Only models/churn/ is uploaded to the training job, so the Lab 3 functions are
copied in below rather than imported from churn_training_skeleton.py.

Test it locally before pushing:

    python models/churn/train_sagemaker.py \
        --train-channel <dir containing the features Parquet> \
        --model-dir /tmp/model --output-data-dir /tmp/out
"""

# Keeps Lab 3 annotations like `-> tuple[np.ndarray, np.ndarray]` from being
# evaluated, whatever Python the training container runs.
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Must match the columns the Lab 3 model was trained on, in the same order.
FEATURE_COLUMNS = [
    "days_since_last_purchase",
    "customer_tenure_days",
    "purchase_frequency_30d",
    "purchase_frequency_90d",
    "purchase_frequency_180d",
    "avg_order_value",
    "total_spend_90d",
    "total_lifetime_value",
    "avg_basket_size_6m",
    "category_diversity_score",
    "online_to_store_ratio",
]

LABEL = "churn_label"

# The recency-only baseline is a model over this one column.
BASELINE_FEATURE = "days_since_last_purchase"

# Not a model input (it correlates with spend and would make the fairness check
# circular). Only used to attribute each validation row to a tier.
SLICE_COLUMN = "loyalty_tier"
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]

# Hyperparameters of the Lab 3 graded run (docs/lab3-evidence/train-final.log).
HYPERPARAMS = {
    "max_depth": 3,
    "eta": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
NUM_ROUND = 200
EARLY_STOPPING_ROUNDS = 20

# There is deliberately no absolute AUC threshold (removed in Lab 3: it gated
# the random seed). The baseline gate is the lift CI, checked below.
EVAL_THRESHOLDS = {
    "precision_top10": 0.50,
    "recall_top10": 0.25,
}
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

# A tier needs this many validation positives before its AUC or recall supports
# a claim (Lab 3: Platinum has about 21).
MIN_SLICE_POSITIVES = 30
# A tier is flagged when its recall is this far below the aggregate.
RECALL_GAP_FLAG = 0.10


# -- Preprocessing -------------------------------------------------------------

def preprocess_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y). Missing values get the sentinel -1.

    Every feature is non-negative by construction, so -1 cannot collide with a
    real value and a tree can isolate it with one split. In practice the Lab 2
    producer gates assert zero nulls, so the fill is inert.
    """
    df = df.dropna(subset=[LABEL])
    n_null = int(df[FEATURE_COLUMNS].isna().sum().sum())
    print(f"Preprocessing {len(df):,} samples; null feature values: {n_null} (filled with -1)")
    X = df[FEATURE_COLUMNS].fillna(-1).to_numpy(dtype=float)
    y = df[LABEL].to_numpy().astype(int)
    print(f"  Positive class rate: {y.mean():.1%}; feature matrix {X.shape}")
    return X, y


# -- Training ------------------------------------------------------------------

def train_model(X_train, y_train, X_val, y_val, seed: int = 42) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS)

    # Class imbalance from the training split (3.545 on the reference data).
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    params = {
        "objective": "binary:logistic",
        # ORDER MATTERS: XGBoost early-stops on the LAST metric, so auc goes
        # last (Lab 3 defect 49).
        "eval_metric": ["logloss", "auc"],
        "scale_pos_weight": spw,
        "seed": seed,
        "verbosity": 1,
        **HYPERPARAMS,
    }
    print(f"Training XGBoost with params: {json.dumps(params)}")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_ROUND,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=25,
    )
    print(f"Best iteration: {model.best_iteration} (validation AUC {model.best_score:.4f})")
    return model


def train_recency_baseline(X_train, y_train, X_val, seed: int = 42) -> np.ndarray:
    """Recency-only model: same training rows, scored on the same validation rows.

    Not a constant predictor (that has AUC 0.5 by construction). The question is
    whether the features beat the rule the business already has for free.
    """
    i = FEATURE_COLUMNS.index(BASELINE_FEATURE)
    dtr = xgb.DMatrix(X_train[:, [i]], label=y_train, feature_names=[BASELINE_FEATURE])
    dva = xgb.DMatrix(X_val[:, [i]], feature_names=[BASELINE_FEATURE])
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 4, "eta": 0.1, "subsample": 0.9,
        "colsample_bytree": 0.9, "scale_pos_weight": spw, "seed": seed,
    }
    return xgb.train(params, dtr, num_boost_round=200).predict(dva)


# -- Evaluation ----------------------------------------------------------------

def bootstrap_lift_ci(y_true, proba, baseline_proba, n: int = BOOTSTRAP_N,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05):
    """Paired percentile CI for (model AUC - baseline AUC) over the validation set.

    Each replicate resamples row indices once and scores both models on those
    rows, so the interval is on the difference. The seed is fixed so the same
    split always gets the same verdict.
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
        raise ValueError("Too few usable bootstrap replicates to form a confidence interval.")
    d = np.sort(diffs)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def evaluate_model(model: xgb.Booster, X_val, y_val, baseline_proba) -> dict:
    """Metrics dict. Raises ValueError if a gate is missed.

    The raise fails the training step, so a model that misses a gate is never
    registered.
    """
    y_pred = model.predict(xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS))

    auc_roc = roc_auc_score(y_val, y_pred)
    top10 = y_pred >= np.percentile(y_pred, 90)
    precision_top10 = y_val[top10].mean() if top10.sum() > 0 else 0.0
    recall_top10 = y_val[top10].sum() / y_val.sum() if y_val.sum() > 0 else 0.0

    baseline_auc = roc_auc_score(y_val, baseline_proba)
    lift_lo, lift_hi = bootstrap_lift_ci(y_val, y_pred, baseline_proba)

    metrics = {
        "auc_roc": round(float(auc_roc), 4),
        "precision_top10": round(float(precision_top10), 4),
        "recall_top10": round(float(recall_top10), 4),
        "auc_vs_baseline": round(float(auc_roc - baseline_auc), 4),
        "baseline_auc": round(float(baseline_auc), 4),
        "lift_ci_low": round(float(lift_lo), 4),
        "lift_ci_high": round(float(lift_hi), 4),
        "positive_rate_val": round(float(y_val.mean()), 4),
        "n_val_samples": int(len(y_val)),
        "eval_timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    }

    print("\n-- Evaluation results --")
    failures = []
    for name, threshold in EVAL_THRESHOLDS.items():
        value = metrics[name]
        print(f"  {name:20s} {value:.4f} (threshold {threshold:.2f}) "
              f"{'PASS' if value >= threshold else 'FAIL'}")
        if value < threshold:
            failures.append(f"{name}={value:.4f} < {threshold}")
    print(f"  {'auc_lift 95% CI':20s} [{lift_lo:.4f}, {lift_hi:.4f}] (must exclude 0) "
          f"{'PASS' if lift_lo > 0 else 'FAIL'}")
    if lift_lo <= 0:
        failures.append(f"auc_lift 95% CI [{lift_lo:.4f}, {lift_hi:.4f}] includes zero - "
                        f"no evidence the model beats the recency-only baseline")
    if failures:
        raise ValueError(f"Model failed evaluation thresholds: {'; '.join(failures)}. "
                         f"Do not promote to Model Registry.")
    return metrics


def evaluate_slices(model: xgb.Booster, X_val, y_val, slice_column):
    """Per-tier results and the aggregate recall they are compared against.

    Recall is at the GLOBAL top-10% cutoff: the retention budget is one list of
    the 10% highest-risk customers, so the question per tier is what share of
    that tier's churners make the list.
    """
    slice_column = pd.Series(np.asarray(slice_column, dtype=object))
    y_pred = model.predict(xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS))
    flagged = y_pred >= np.percentile(y_pred, 90)
    aggregate_recall = float(flagged[y_val == 1].mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    out = {}
    for tier in TIERS:
        mask = (slice_column == tier).to_numpy()
        y_s, p_s, f_s = y_val[mask], y_pred[mask], flagged[mask]
        positives = int(y_s.sum())
        auc_s = roc_auc_score(y_s, p_s) if 0 < positives < len(y_s) else float("nan")
        boot = []
        for _ in range(1000):
            s = rng.integers(0, len(y_s), len(y_s))
            if 0 < y_s[s].sum() < len(s):
                boot.append(roc_auc_score(y_s[s], p_s[s]))
        recall_s = float(f_s[y_s == 1].mean()) if positives else float("nan")
        out[tier] = {
            "n": int(mask.sum()),
            "positives": positives,
            "churn_rate": round(float(y_s.mean()), 4) if len(y_s) else float("nan"),
            "auc": round(float(auc_s), 4),
            "auc_ci_low": round(float(np.quantile(boot, 0.025)), 4) if boot else float("nan"),
            "auc_ci_high": round(float(np.quantile(boot, 0.975)), 4) if boot else float("nan"),
            "recall_top10": round(recall_s, 4),
            "share_flagged": round(float(f_s.mean()), 4) if len(f_s) else float("nan"),
            "recall_gap_flag": bool(recall_s < aggregate_recall - RECALL_GAP_FLAG),
            "too_small_to_conclude": positives < MIN_SLICE_POSITIVES,
        }
    print(f"\n-- Slice evaluation ({SLICE_COLUMN}; aggregate recall {aggregate_recall:.4f}) --")
    print(pd.DataFrame(out).T.to_string())
    return out, aggregate_recall


def train(df: pd.DataFrame, seed: int = 42):
    """Train the churn model on `df` and return (booster, metrics).

    Same steps as Lab 3's main(): preprocess, stratified 70/30 split with the
    tier carried through, XGBoost, recency baseline, gated evaluation, slices.
    """
    X, y = preprocess_features(df)
    tier = df.dropna(subset=[LABEL])[SLICE_COLUMN]

    X_train, X_val, y_train, y_val, _, tier_val = train_test_split(
        X, y, tier, test_size=0.30, random_state=seed, stratify=y)
    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,}")

    model = train_model(X_train, y_train, X_val, y_val, seed=seed)
    baseline_proba = train_recency_baseline(X_train, y_train, X_val, seed=seed)
    metrics = evaluate_model(model, X_val, y_val, baseline_proba)

    slices, aggregate_recall = evaluate_slices(model, X_val, y_val, tier_val)
    metrics["slices"] = slices
    metrics["aggregate_recall_top10"] = round(aggregate_recall, 4)
    return model, metrics
