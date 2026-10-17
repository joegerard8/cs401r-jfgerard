#!/usr/bin/env bash
# collect-lab3-evidence.sh
# Exports the Lab 3 evidence that lives only in AWS (Athena query, MLflow runs,
# Model Registry) into docs/lab3-evidence/ and checks every required file.
# The TA cannot see your AWS account -- if it is not in the repo, it was not done.
#
# Run AFTER your final training run, BEFORE teardown:
#   bash scripts/collect-lab3-evidence.sh --track B        # or --track C
# Options: --mlflow-app-name NAME  --experiment NAME  --model-package-group NAME
set -uo pipefail
mkdir -p docs/lab3-evidence
python3 "$(dirname "$0")/collect_lab3_evidence.py" "$@" 2>&1 | tee docs/lab3-evidence/collect-output.txt
exit "${PIPESTATUS[0]}"
