"""Paired comparison of the Lab 3 experiment runs against the anchor run.

Each run's saved model is re-scored on the identical test split (same Athena
query, ORDER BY customer_id, random_state=42), and the AUC difference to the
anchor gets a paired bootstrap CI: rows are resampled once per replicate and
both models are scored on them, exactly as the recency-lift CI is computed.
A difference whose CI straddles zero is noise, however it looks as a point.

Usage: python scripts/compare_lab3_runs.py --artifacts-bucket <bucket>
Writes docs/lab3-evidence/runs/comparison.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import churn_training_skeleton as cts  # noqa: E402

RUNS = ["full11-d6-eta0.1", "full11-d3-eta0.03", "full11-plus-riskscore", "no-monetary", "pruned-7"]
ANCHOR = RUNS[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-bucket", required=True)
    p.add_argument("--feature-group-name", default="northstar-dev-customer-features")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    # The loader selects FEATURE_COLUMNS; widen it so the risk-score run's
    # extra input comes back too. Row set and order are unaffected.
    cts.FEATURE_COLUMNS = cts.FEATURE_COLUMNS + [cts.BASELINE_COLUMN]
    df = cts.load_features_from_feature_store(args.feature_group_name, "2026-01-01", "2027-01-01",
                                              args.artifacts_bucket)
    df = df.dropna(subset=[cts.LABEL_COLUMN]).reset_index(drop=True)
    _, test_idx = train_test_split(np.arange(len(df)), test_size=0.30, random_state=42,
                                   stratify=df[cts.LABEL_COLUMN].values)
    test = df.iloc[test_idx]
    y = test[cts.LABEL_COLUMN].values.astype(int)

    proba = {}
    for run in RUNS:
        meta = json.load(open(f"model/{run}/model_metadata.json"))
        cols = meta["feature_columns"]
        booster = xgb.Booster()
        booster.load_model(f"model/{run}/model.xgb")
        proba[run] = booster.predict(xgb.DMatrix(test[cols].astype(float).values, feature_names=cols))
        logged = meta["evaluation_metrics"]["auc_roc"]
        assert abs(roc_auc_score(y, proba[run]) - logged) < 1e-4, f"{run}: re-scored AUC != logged {logged}"

    rng = np.random.default_rng(42)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(args.n_boot)]
    rows = []
    for run in RUNS:
        m = json.load(open(f"docs/lab3-evidence/runs/{run}/evaluation_metrics.json"))
        diffs = [roc_auc_score(y[s], proba[run][s]) - roc_auc_score(y[s], proba[ANCHOR][s]) for s in boots]
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        rows.append({
            "run": run,
            "auc_roc": m["auc_roc"],
            "precision_top10": m["precision_top10"],
            "recall_top10": m["recall_top10"],
            "baseline_auc": m["baseline_auc"],
            "lift_vs_recency": m["auc_vs_baseline"],
            "lift_ci_low": m["lift_ci_low"],
            "lift_ci_high": m["lift_ci_high"],
            "delta_auc_vs_anchor": round(roc_auc_score(y, proba[run]) - roc_auc_score(y, proba[ANCHOR]), 4),
            "delta_ci_low": round(float(lo), 4),
            "delta_ci_high": round(float(hi), 4),
            "delta_significant": run != ANCHOR and (lo > 0 or hi < 0),
        })
    out = pd.DataFrame(rows)
    path = "docs/lab3-evidence/runs/comparison.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
