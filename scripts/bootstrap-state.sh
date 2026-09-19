#!/usr/bin/env bash
# bootstrap-state.sh
# One-time setup: creates the S3 bucket and DynamoDB table for Terraform remote state.
# Run this before `terraform init`. Safe to run again: every step checks for an
# existing resource first and reports [SKIP], so a re-run is a no-op rather than
# an error. (This header used to claim the opposite.)
#
# Usage: bash scripts/bootstrap-state.sh
# Prerequisites: AWS CLI configured with valid credentials and sufficient IAM permissions.

set -euo pipefail

AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
STATE_BUCKET="northstar-tfstate-${AWS_ACCOUNT_ID}"
LOCK_TABLE="northstar-tfstate-lock"

echo "==> Bootstrap: Terraform remote state"
echo "    Account ID : ${AWS_ACCOUNT_ID}"
echo "    Region     : ${AWS_REGION}"
echo "    State bucket: ${STATE_BUCKET}"
echo "    Lock table : ${LOCK_TABLE}"
echo ""

# ── S3 state bucket ────────────────────────────────────────────────────────────
if aws s3api head-bucket --bucket "${STATE_BUCKET}" 2>/dev/null; then
  echo "[SKIP] S3 bucket already exists: ${STATE_BUCKET}"
else
  echo "[CREATE] S3 bucket: ${STATE_BUCKET}"
  if [ "${AWS_REGION}" = "us-east-1" ]; then
    aws s3api create-bucket \
      --bucket "${STATE_BUCKET}" \
      --region "${AWS_REGION}"
  else
    aws s3api create-bucket \
      --bucket "${STATE_BUCKET}" \
      --region "${AWS_REGION}" \
      --create-bucket-configuration LocationConstraint="${AWS_REGION}"
  fi
fi

echo "[SET] Versioning on state bucket"
aws s3api put-bucket-versioning \
  --bucket "${STATE_BUCKET}" \
  --versioning-configuration Status=Enabled

echo "[SET] Block public access on state bucket"
aws s3api put-public-access-block \
  --bucket "${STATE_BUCKET}" \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

echo "[SET] Server-side encryption on state bucket"
aws s3api put-bucket-encryption \
  --bucket "${STATE_BUCKET}" \
  --server-side-encryption-configuration '{
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
  }'

# ── DynamoDB lock table ────────────────────────────────────────────────────────
if aws dynamodb describe-table --table-name "${LOCK_TABLE}" --region "${AWS_REGION}" 2>/dev/null; then
  echo "[SKIP] DynamoDB table already exists: ${LOCK_TABLE}"
else
  echo "[CREATE] DynamoDB lock table: ${LOCK_TABLE}"
  aws dynamodb create-table \
    --table-name "${LOCK_TABLE}" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "${AWS_REGION}"

  echo "[WAIT] Waiting for table to become active..."
  aws dynamodb wait table-exists --table-name "${LOCK_TABLE}" --region "${AWS_REGION}"
fi

# ── Patch backend.tf with real account ID ─────────────────────────────────────
BACKEND_FILE="infrastructure/environments/dev/backend.tf"
if [ ! -f "${BACKEND_FILE}" ] && [ -f "${BACKEND_FILE}.example" ]; then
  echo "[COPY] ${BACKEND_FILE}.example -> ${BACKEND_FILE}"
  cp "${BACKEND_FILE}.example" "${BACKEND_FILE}"
fi
if grep -q "YOUR_ACCOUNT_ID" "${BACKEND_FILE}" 2>/dev/null; then
  echo "[PATCH] Updating ${BACKEND_FILE} with account ID"
  sed -i.bak "s/YOUR_ACCOUNT_ID/${AWS_ACCOUNT_ID}/g" "${BACKEND_FILE}"
  rm -f "${BACKEND_FILE}.bak"
  echo "        Done. Review the change before committing."
else
  echo "[SKIP] backend.tf already patched"
fi

echo ""
echo "==> Bootstrap complete."
echo ""
echo "Next steps:"
echo "  1. cd infrastructure/environments/dev"
echo "  2. (optional) cp terraform.tfvars.example terraform.tfvars and adjust; every value is a default"
echo "  3. terraform init"
echo "  4. terraform plan"
echo "  5. terraform apply"
