#!/usr/bin/env bash
# verify-lab1.sh
# Runs the Lab 1 rubric checks automatically after terraform apply.
# Run from repo root: bash scripts/verify-lab1.sh
#
# What it checks (Lab 1 — simplified single-AZ, public-subnet):
#   - S3 bucket exists with all 4 prefixes
#   - MLEngineer IAM role exists + policy simulation
#   - VPC exists + public subnet exists
#   - SageMaker Domain is InService
#   - Module structure, remote state, repo quality
#   - Every committed evidence file except this script's own output
#     (docs/lab1-verify-output.txt is what you are producing right now)
#
# Not checked (deferred to Lab 2):
#   - Lifecycle rules (added once real data flows)
#   - DataEngineer / ModelMonitor roles (added in Lab 2)
#   - Private subnets / NAT Gateway (added in Lab 2)

set -euo pipefail

# Colour only when writing to a terminal. You are asked to commit this
# script's output, and a file full of raw \033[0;32m escapes is unreadable for
# whoever grades it.
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  NC='\033[0m'
else
  RED='' ; GREEN='' ; YELLOW='' ; NC=''
fi

PASS=0
FAIL=0

check() {
  local label="$1"
  local result="$2"
  if [ "$result" = "PASS" ]; then
    echo -e "  ${GREEN}✓${NC} ${label}"
    ((PASS++)) || true
  else
    echo -e "  ${RED}✗${NC} ${label} — ${result}"
    ((FAIL++)) || true
  fi
}

echo ""
echo "========================================"
echo " NorthStar Lab 1 — Verification Script"
echo "========================================"
echo ""

# ── Pull Terraform outputs ──────────────────────────────────────────────────
cd infrastructure/environments/dev
BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null) \
  || { echo "ERROR: Run 'terraform apply' in infrastructure/environments/dev first."; exit 1; }
ML_ROLE=$(terraform output -raw ml_engineer_role_arn)
VPC_ID=$(terraform output -raw vpc_id)
PUBLIC_SUBNET_ID=$(terraform output -raw public_subnet_id)
DOMAIN_ID=$(terraform output -raw sagemaker_domain_id)
cd ../../..

AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# ── Part A: AWS Environment ─────────────────────────────────────────────────
echo "── Part A: AWS Environment (35 pts total in rubric) ─────────────────────"

# A2 — Network
aws ec2 describe-vpcs --vpc-ids "${VPC_ID}" > /dev/null 2>&1 \
  && check "A2 VPC exists: ${VPC_ID}" "PASS" \
  || check "A2 VPC exists" "VPC NOT FOUND"

SUBNET_EXISTS=$(aws ec2 describe-subnets \
  --subnet-ids "${PUBLIC_SUBNET_ID}" \
  --query 'length(Subnets)' --output text 2>/dev/null || echo "0")
[ "${SUBNET_EXISTS}" -ge 1 ] \
  && check "A2 Public subnet exists: ${PUBLIC_SUBNET_ID}" "PASS" \
  || check "A2 Public subnet exists" "NOT FOUND"

# Confirm no private subnets (we shouldn't have any in Lab 1)
PRIVATE_COUNT=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=${VPC_ID}" "Name=tag:Tier,Values=private" \
  --query 'length(Subnets)' --output text 2>/dev/null || echo "0")
[ "${PRIVATE_COUNT}" -eq 0 ] \
  && check "A2 No private subnets (correct for Lab 1)" "PASS" \
  || check "A2 No private subnets" "FOUND ${PRIVATE_COUNT} — should be 0 in Lab 1"

# A3 — Storage
aws s3api head-bucket --bucket "${BUCKET}" >/dev/null 2>&1 \
  && check "A3 S3 bucket exists: ${BUCKET}" "PASS" \
  || check "A3 S3 bucket exists" "BUCKET NOT FOUND"

for prefix in raw processed features artifacts; do
  aws s3 ls "s3://${BUCKET}/${prefix}/" > /dev/null 2>&1 \
    && check "A3 S3 prefix exists: ${prefix}/" "PASS" \
    || check "A3 S3 prefix exists: ${prefix}/" "PREFIX MISSING"
done

# A3 — IAM (MLEngineer only in Lab 1)
ML_ROLE_NAME=$(echo "${ML_ROLE}" | cut -d'/' -f2)
aws iam get-role --role-name "${ML_ROLE_NAME}" > /dev/null 2>&1 \
  && check "A3 IAM role exists: ${ML_ROLE_NAME}" "PASS" \
  || check "A3 IAM role exists: ${ML_ROLE_NAME}" "ROLE NOT FOUND"

# IAM simulation: MLEngineer CAN create a training job
SIM=$(aws iam simulate-principal-policy \
  --policy-source-arn "${ML_ROLE}" \
  --action-names "sagemaker:CreateTrainingJob" \
  --query 'EvaluationResults[0].EvalDecision' --output text 2>/dev/null || echo "ERROR")
[ "${SIM}" = "allowed" ] \
  && check "A3 IAM sim: MLEngineer can CreateTrainingJob" "PASS" \
  || check "A3 IAM sim: MLEngineer can CreateTrainingJob" "GOT: ${SIM}"

# IAM simulation: MLEngineer CANNOT write to raw/ (no DataEngineer in Lab 1)
SIM=$(aws iam simulate-principal-policy \
  --policy-source-arn "${ML_ROLE}" \
  --action-names "s3:PutObject" \
  --resource-arns "arn:aws:s3:::${BUCKET}/raw/test.csv" \
  --query 'EvaluationResults[0].EvalDecision' --output text 2>/dev/null || echo "ERROR")
[ "${SIM}" = "implicitDeny" ] || [ "${SIM}" = "explicitDeny" ] \
  && check "A3 IAM sim: MLEngineer DENIED write to raw/" "PASS" \
  || check "A3 IAM sim: MLEngineer DENIED write to raw/" "GOT: ${SIM} (expected deny)"

# A4 — SageMaker Domain
DOMAIN_STATUS=$(aws sagemaker describe-domain --domain-id "${DOMAIN_ID}" \
  --query 'Status' --output text 2>/dev/null || echo "MISSING")
[ "${DOMAIN_STATUS}" = "InService" ] \
  && check "A4 SageMaker Domain InService: ${DOMAIN_ID}" "PASS" \
  || check "A4 SageMaker Domain InService" "STATUS: ${DOMAIN_STATUS}"

# ── Part B: Terraform Module Structure ─────────────────────────────────────
echo ""
echo "── Part B: Terraform IaC (Lab 1b) ──────────────────────────────────────"

# B1 — Module directories
for mod in vpc iam sagemaker storage; do
  [ -f "infrastructure/modules/${mod}/main.tf" ] \
    && check "B1 Module exists: ${mod}/" "PASS" \
    || check "B1 Module exists: ${mod}/" "MISSING main.tf"
done

# B4 — No hardcoded project name in module .tf files
# `|| true` matters: under `set -o pipefail` a grep with no matches exits 1,
# which is exactly the passing case, and without it the script would stop here.
LITERALS=$( { grep -r '"northstar' infrastructure/modules/ --include="*.tf" \
  | grep -v "var\." | grep -v "#" || true; } | wc -l | tr -d ' ')
[ "${LITERALS}" -eq 0 ] \
  && check "B4 No hardcoded 'northstar' literals in modules/" "PASS" \
  || check "B4 No hardcoded 'northstar' literals in modules/" "FOUND ${LITERALS} literal(s)"

# B3 — Remote state backend
grep -q "dynamodb_table" infrastructure/environments/dev/backend.tf 2>/dev/null \
  && check "B3 Remote state: S3 backend + DynamoDB lock" "PASS" \
  || check "B3 Remote state" "Missing dynamodb_table in backend.tf"

# B2 — Apply output captured
[ -f "docs/lab1b-apply-output.txt" ] \
  && check "B2 terraform apply output saved" "PASS" \
  || check "B2 terraform apply output saved" "docs/lab1b-apply-output.txt MISSING"

grep -q "Apply complete" docs/lab1b-apply-output.txt 2>/dev/null \
  && check "B2 apply output shows Apply complete" "PASS" \
  || check "B2 apply output shows Apply complete" "not found in docs/lab1b-apply-output.txt"

# B5 — LocalStack output captured
[ -f "docs/lab1b-localstack-output.txt" ] \
  && check "B5 LocalStack validation output saved" "PASS" \
  || check "B5 LocalStack validation output saved" "docs/lab1b-localstack-output.txt MISSING"

# ── Shared Deliverables ─────────────────────────────────────────────────────
echo ""
echo "── Shared Deliverables ──────────────────────────────────────────────────"

[ -f ".gitignore" ] \
  && check "S .gitignore present" "PASS" \
  || check "S .gitignore present" "MISSING"

grep -q "\.tfvars" .gitignore 2>/dev/null \
  && check "S .gitignore covers *.tfvars" "PASS" \
  || check "S .gitignore covers *.tfvars" "MISSING — students may commit secrets"

grep -q "\.terraform" .gitignore 2>/dev/null \
  && check "S .gitignore covers .terraform/" "PASS" \
  || check "S .gitignore covers .terraform/" "MISSING"

# Same scan the TA runs. A bare `git log -S "AKIA"` is wrong here: the word
# appears in this script, in aws-account-setup.md, and (as AWS's documented
# placeholder AKIAIOSFODNN7EXAMPLE) in every LocalStack output file.
if bash scripts/check-secrets.sh >/dev/null 2>&1; then
  check "S No AWS credentials in git history (scripts/check-secrets.sh)" "PASS"
else
  check "S No AWS credentials in git history (scripts/check-secrets.sh)" "FINDINGS — run: bash scripts/check-secrets.sh"
fi

[ -f "docs/lab1-architecture-diagram.png" ] \
  && check "A1 Architecture diagram submitted" "PASS" \
  || check "A1 Architecture diagram submitted" "docs/lab1-architecture-diagram.png MISSING"

[ -f "docs/lab1-studio-shutdown.png" ] \
  && check "S Studio shutdown screenshot submitted" "PASS" \
  || check "S Studio shutdown screenshot submitted" "docs/lab1-studio-shutdown.png MISSING"

[ -f "docs/lab1-adr.md" ] \
  && check "S ADR document submitted" "PASS" \
  || check "S ADR document submitted" "docs/lab1-adr.md MISSING"

[ -f "docs/lab1-cost-estimate.md" ] \
  && check "S Cost estimate document submitted" "PASS" \
  || check "S Cost estimate document submitted" "docs/lab1-cost-estimate.md MISSING"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
printf " Results: ${GREEN}%d passed${NC}  ${RED}%d failed${NC}\n" "${PASS}" "${FAIL}"
echo "========================================"
echo ""
if [ "${FAIL}" -gt 0 ]; then
  echo -e "${YELLOW}Fix the failing checks above and re-run: bash scripts/verify-lab1.sh${NC}"
  echo ""
fi
