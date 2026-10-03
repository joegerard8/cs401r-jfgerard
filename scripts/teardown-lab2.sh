#!/usr/bin/env bash
# teardown-lab2.sh
# Full Lab 2 teardown: terraform destroy plus the resources Terraform does not own.
#
# WHY THIS SCRIPT EXISTS
# `terraform destroy` alone does NOT fully tear down Lab 2. Six things are
# created outside Terraform's state and are left behind:
#
#   1. Glue ENIs            - orphaned in the private subnet after job runs;
#                             block subnet and security group deletion
#                             (removed BEFORE destroy so it does not hang)
#   2. SageMaker EFS        - Studio creates a filesystem Terraform never sees;
#                             KEEPS BILLING and its mount target blocks the subnet
#   3. SageMaker NFS SGs    - two auto-created security groups block VPC deletion
#   4. S3 object versions   - versioned bucket cannot be deleted while non-empty
#   5. Feature Store catalog- the sagemaker_featurestore Glue database and table
#   6. SageMaker lineage    - contexts and artifacts created by the feature group;
#                             they survive DeleteFeatureGroup and must have their
#                             associations removed before they will delete
#
# Items 1-3 make destroy hang for 10+ minutes before failing. Item 2 is the one
# that actually costs money after you think you are done.
#
# Usage: bash scripts/teardown-lab2.sh
# Prerequisites: AWS CLI authenticated against the lab account.

set -uo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PROJECT="${PROJECT:-northstar}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
TF_DIR="infrastructure/environments/${ENVIRONMENT}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
# Without credentials every step below fails quietly and the final check used
# to report OK anyway (2026-10-01). Stop here instead.
if [ -z "${ACCOUNT_ID}" ]; then
  echo "==> ERROR: AWS CLI is not authenticated (aws sts get-caller-identity failed)."
  echo "    Check AWS_PROFILE / credentials and re-run. Nothing was deleted."
  exit 1
fi
BUCKET="${PROJECT}-${ENVIRONMENT}-data-${ACCOUNT_ID}"

echo "==> Lab 2 teardown"
echo "    Account : ${ACCOUNT_ID}"
echo "    Region  : ${REGION}"
echo "    Bucket  : ${BUCKET}"
echo ""

# ── 1. Empty the versioned data bucket ─────────────────────────────────────────
# Do this BEFORE destroy so the S3 delete does not fail with BucketNotEmpty.
if aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
  echo "[1/7] Emptying ${BUCKET} (all versions and delete markers)"
  for key in Versions DeleteMarkers; do
    while true; do
      payload=$(aws s3api list-object-versions --bucket "${BUCKET}" --max-keys 500 \
        --output json --query "{Objects: ${key}[].{Key:Key,VersionId:VersionId}}" 2>/dev/null)
      count=$(echo "${payload}" | python3 -c "import sys,json; o=json.load(sys.stdin).get('Objects'); print(len(o) if o else 0)" 2>/dev/null || echo 0)
      [ "${count}" = "0" ] && break
      aws s3api delete-objects --bucket "${BUCKET}" --delete "${payload}" >/dev/null 2>&1
      echo "      removed ${count} ${key}"
    done
  done
else
  echo "[1/7] Bucket ${BUCKET} not present - skipping"
fi

# ── 2. Delete the Feature Store Glue catalog database ─────────────────────────
# Created automatically by the offline store; not in Terraform state.
echo "[2/7] Removing Feature Store Glue catalog database"
if aws glue get-database --name sagemaker_featurestore >/dev/null 2>&1; then
  aws glue delete-database --name sagemaker_featurestore >/dev/null 2>&1 \
    && echo "      deleted sagemaker_featurestore"
else
  echo "      not present"
fi

# ── 2b. Orphaned Glue ENIs, BEFORE destroy ────────────────────────────────────
# Every Glue job run inside the VPC leaves its network interfaces behind in
# "available" state. They pin the private subnet and the security group, and
# terraform destroy waits about ten minutes on each before giving up. Detached
# interfaces are safe to delete; do it first so destroy finishes in one pass.
echo "[2b/7] Removing orphaned Glue network interfaces"
for eni in $(aws ec2 describe-network-interfaces \
    --filters "Name=status,Values=available" \
    --query "NetworkInterfaces[?contains(Description, 'Glue')].NetworkInterfaceId" \
    --output text 2>/dev/null); do
  aws ec2 delete-network-interface --network-interface-id "${eni}" >/dev/null 2>&1 \
    && echo "      deleted ENI ${eni}"
done

# ── 3. terraform destroy ───────────────────────────────────────────────────────
echo "[3/7] terraform destroy"
mkdir -p docs
terraform -chdir="${TF_DIR}" destroy -auto-approve -input=false ${TF_DESTROY_ARGS:-} 2>&1 \
  | tee docs/lab2-destroy-output.txt | tail -5

# ── 4. Orphaned SageMaker EFS (the one that keeps billing) ────────────────────
# With retention_policy { home_efs_file_system = "Delete" } on the Domain this
# finds nothing. It stays here for Domains created without it.
echo "[4/7] Removing orphaned SageMaker Studio EFS filesystems"
for fs in $(aws efs describe-file-systems --query 'FileSystems[*].FileSystemId' --output text 2>/dev/null); do
  for mt in $(aws efs describe-mount-targets --file-system-id "${fs}" \
                --query 'MountTargets[*].MountTargetId' --output text 2>/dev/null); do
    aws efs delete-mount-target --mount-target-id "${mt}" >/dev/null 2>&1 \
      && echo "      deleted mount target ${mt}"
  done
  # Mount targets must be fully gone before the filesystem will delete.
  for _ in $(seq 1 30); do
    remaining=$(aws efs describe-mount-targets --file-system-id "${fs}" \
                  --query 'length(MountTargets)' --output text 2>/dev/null || echo 0)
    [ "${remaining}" = "0" ] && break
    sleep 5
  done
  aws efs delete-file-system --file-system-id "${fs}" >/dev/null 2>&1 \
    && echo "      deleted filesystem ${fs}"
done

# ── 5. Orphaned Glue ENIs and SageMaker NFS security groups ───────────────────
echo "[5/7] Removing orphaned ENIs and SageMaker NFS security groups"
for eni in $(aws ec2 describe-network-interfaces \
    --filters "Name=status,Values=available" \
    --query "NetworkInterfaces[?contains(Description, 'Glue')].NetworkInterfaceId" \
    --output text 2>/dev/null); do
  aws ec2 delete-network-interface --network-interface-id "${eni}" >/dev/null 2>&1 \
    && echo "      deleted ENI ${eni}"
done

for sg in $(aws ec2 describe-security-groups \
    --query "SecurityGroups[?contains(GroupName, 'nfs-d-')].GroupId" \
    --output text 2>/dev/null); do
  # These two reference each other, so strip rules before deleting.
  perms=$(aws ec2 describe-security-groups --group-ids "${sg}" \
            --query 'SecurityGroups[0].IpPermissions' --output json 2>/dev/null)
  [ "${perms}" != "[]" ] && aws ec2 revoke-security-group-ingress \
    --group-id "${sg}" --ip-permissions "${perms}" >/dev/null 2>&1
  eperms=$(aws ec2 describe-security-groups --group-ids "${sg}" \
            --query 'SecurityGroups[0].IpPermissionsEgress' --output json 2>/dev/null)
  [ "${eperms}" != "[]" ] && aws ec2 revoke-security-group-egress \
    --group-id "${sg}" --ip-permissions "${eperms}" >/dev/null 2>&1
done
for sg in $(aws ec2 describe-security-groups \
    --query "SecurityGroups[?contains(GroupName, 'nfs-d-')].GroupId" \
    --output text 2>/dev/null); do
  aws ec2 delete-security-group --group-id "${sg}" >/dev/null 2>&1 \
    && echo "      deleted security group ${sg}"
done

# If destroy failed earlier because of the above, a second pass now succeeds.
if grep -q "Error:" docs/lab2-destroy-output.txt 2>/dev/null; then
  echo "      re-running destroy now that blockers are cleared"
  terraform -chdir="${TF_DIR}" destroy -auto-approve -input=false ${TF_DESTROY_ARGS:-} 2>&1 \
    | tee -a docs/lab2-destroy-output.txt | tail -3
fi

# ── 6. Orphaned SageMaker lineage entities ────────────────────────────────────
# Feature Store creates lineage contexts and a DataSet artifact. DeleteFeatureGroup
# leaves them behind, and they refuse to delete while associations exist, so the
# association graph has to be unwound first.
echo "[6/7] Removing orphaned SageMaker lineage contexts and artifacts"
for arn in $(aws sagemaker list-contexts --query 'ContextSummaries[*].ContextArn' --output text 2>/dev/null | tr '\t' '\n') \
           $(aws sagemaker list-artifacts --query 'ArtifactSummaries[*].ArtifactArn' --output text 2>/dev/null | tr '\t' '\n'); do
  [ -z "${arn}" ] && continue
  aws sagemaker list-associations --destination-arn "${arn}" \
    --query 'AssociationSummaries[*].[SourceArn,DestinationArn]' --output text 2>/dev/null | \
  while read -r src dst; do
    [ -z "${src}" ] && continue
    aws sagemaker delete-association --source-arn "${src}" --destination-arn "${dst}" >/dev/null 2>&1 \
      && echo "      unlinked ${src##*/} -> ${dst##*/}"
  done
done
for a in $(aws sagemaker list-artifacts --query 'ArtifactSummaries[*].ArtifactArn' --output text 2>/dev/null | tr '\t' '\n'); do
  [ -z "${a}" ] && continue
  aws sagemaker delete-artifact --artifact-arn "${a}" >/dev/null 2>&1 && echo "      deleted artifact ${a##*/}"
done
for c in $(aws sagemaker list-contexts --query 'ContextSummaries[*].ContextName' --output text 2>/dev/null | tr '\t' '\n'); do
  [ -z "${c}" ] && continue
  aws sagemaker delete-context --context-name "${c}" >/dev/null 2>&1 && echo "      deleted context ${c}"
done

# Empty CloudWatch log groups left by Glue job and crawler runs.
for lg in /aws-glue/crawlers /aws-glue/jobs/error /aws-glue/jobs/logs-v2 /aws-glue/jobs/output; do
  aws logs delete-log-group --log-group-name "${lg}" >/dev/null 2>&1 && echo "      deleted log group ${lg}"
done

# ── 7. Verify nothing billable survives ───────────────────────────────────────
echo "[7/7] Verifying teardown"
fail=0
# The CLI applies --query per page of a paginated response, so `length(...)`
# can print one number per page. Sum whatever comes back.
#
# A failed AWS call is a FAILED check, never "OK". Piping straight into awk
# turned an error (expired credentials, wrong profile) into "0 -> OK", so the
# teardown could report clean while resources were still billing (2026-10-01).
check () {
  if ! raw=$(eval "$2" 2>/dev/null); then
    printf "      %-22s CHECK FAILED (AWS call errored; credentials/region?)\n" "$1"; fail=1; return
  fi
  n=$(printf '%s\n' "${raw}" | awk '{ s += $1 } END { print s + 0 }')
  if [ "${n}" = "0" ]; then
    printf "      %-22s OK\n" "$1"
  else
    printf "      %-22s STILL PRESENT: %s\n" "$1" "${n}"; fail=1
  fi
}
check "NAT gateways"      "aws ec2 describe-nat-gateways --filter Name=state,Values=available,pending --query 'length(NatGateways)' --output text"
check "Elastic IPs"       "aws ec2 describe-addresses --query 'length(Addresses)' --output text"
check "SageMaker domains" "aws sagemaker list-domains --query 'length(Domains)' --output text"
check "Feature groups"    "aws sagemaker list-feature-groups --query 'length(FeatureGroupSummaries)' --output text"
check "EFS filesystems"   "aws efs describe-file-systems --query 'length(FileSystems)' --output text"
check "Glue jobs"         "aws glue list-jobs --query 'length(JobNames)' --output text"
check "NorthStar VPCs"    "aws ec2 describe-vpcs --filters Name=tag:Project,Values=${PROJECT} --query 'length(Vpcs)' --output text"
check "Lineage contexts"  "aws sagemaker list-contexts --query 'length(ContextSummaries)' --output text"
check "Lineage artifacts" "aws sagemaker list-artifacts --query 'length(ArtifactSummaries)' --output text"
check "Glue databases"    "aws glue get-databases --query 'length(DatabaseList)' --output text"

echo ""
if [ "${fail}" = "0" ]; then
  echo "==> Teardown complete. No billable Lab 2 resources remain."
  echo "    The Terraform state bucket is intentionally retained for Lab 3."
else
  echo "==> WARNING: teardown NOT verified. Fix anything marked STILL PRESENT or"
  echo "    CHECK FAILED and re-run. Resources may still be billing."
  exit 1
fi
