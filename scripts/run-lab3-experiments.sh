#!/usr/bin/env bash
# run-lab3-experiments.sh
# The five exploratory churn runs behind docs/lab3-evaluation-report.md.
# Each run answers one question; see the report for the hypotheses and results.
#
# Every run issues its own Athena query against the Feature Store offline
# store, so all five train on identical data and the identical 70/30 split.
# Runs are tracked in the MLflow App; none is registered (the graded run is
# a separate invocation with --register, see README).
#
# Usage: MLFLOW_APP_ARN=arn:aws:sagemaker:... bash scripts/run-lab3-experiments.sh

set -euo pipefail

TF_DIR="infrastructure/environments/dev"
BUCKET="${BUCKET:-$(terraform -chdir="${TF_DIR}" output -raw s3_bucket_name)}"
FEATURE_GROUP="${FEATURE_GROUP:-$(terraform -chdir="${TF_DIR}" output -raw feature_group_name)}"
: "${MLFLOW_APP_ARN:?set MLFLOW_APP_ARN to your MLflow App ARN}"
export PYTHONIOENCODING=utf-8

run () {
  local name="$1" desc="$2"; shift 2
  local out="docs/lab3-evidence/runs/${name}"
  mkdir -p "${out}"
  echo "==> ${name}"
  python churn_training_skeleton.py \
    --feature-group-name "${FEATURE_GROUP}" \
    --artifacts-bucket "${BUCKET}" \
    --mlflow-arn "${MLFLOW_APP_ARN}" \
    --run-name "${name}" \
    --run-description "${desc}" \
    --model-dir "model/${name}" \
    --output-data-dir "${out}" \
    "$@" 2>&1 | tee "${out}/train.log"
}

run full11-d6-eta0.1 \
  "Anchor: skeleton defaults on all 11 Lab 2 features. Does the feature set beat the recency-only baseline?" \
  --max-depth 6 --eta 0.1 --num-round 200

run full11-d3-eta0.03 \
  "Capacity: depth 6 early-stopped at iteration ~10, suggesting fast overfitting (~24 training churners per leaf). Do shallower, slower-learning trees generalise better?" \
  --max-depth 3 --eta 0.03 --num-round 1000

run full11-plus-riskscore \
  "Leakage/redundancy check required by Lab 3: churn_risk_score has Spearman +1.00 with recency. Does a monotone recency transform add information to a tree model?" \
  --max-depth 6 --eta 0.1 --num-round 200 --include-risk-score

run no-monetary \
  "Ablation: drop avg_order_value, total_spend_90d, total_lifetime_value. Among recent buyers these out-rank recency (AOV univariate AUC 0.660 vs 0.631). Is the trade-down signal real?" \
  --max-depth 6 --eta 0.1 --num-round 200 \
  --drop-features avg_order_value,total_spend_90d,total_lifetime_value

run pruned-7 \
  "Ablation: drop the four weakest features (tenure, basket size, online ratio, category diversity; univariate AUC 0.51-0.57 among recent buyers). Are they decorative?" \
  --max-depth 6 --eta 0.1 --num-round 200 \
  --drop-features customer_tenure_days,avg_basket_size_6m,online_to_store_ratio,category_diversity_score

echo "==> done; results in docs/lab3-evidence/runs/ and MLflow experiment northstar-churn"
