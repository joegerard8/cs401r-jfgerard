#!/usr/bin/env bash
# verify-lab2.sh
# Automated rubric verification for Lab 2. Run this BEFORE you submit.
#
# This checks the same assertions the TA runs. Every check prints PASS or FAIL
# with the observed value, so a failure tells you what to fix rather than just
# that something is wrong.
#
# Requires the stack to be deployed and both Glue jobs to have run successfully.
#
# Usage: bash scripts/verify-lab2.sh

set -uo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PROJECT="${PROJECT:-northstar}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
P="${PROJECT}-${ENVIRONMENT}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
BUCKET="${P}-data-${ACCOUNT_ID}"
DB=$(echo "${PROJECT}_${ENVIRONMENT}" | tr '-' '_')

# Colour only on a terminal: you commit this script's output, and a file full
# of raw escape codes is unreadable for whoever grades it.
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; B=$'\033[1m'; N=$'\033[0m'; else G=''; R=''; B=''; N=''; fi
PASS=0; FAIL=0
ok ()   { printf "  ${G}PASS${N}  %-52s %s\n" "$1" "${2:-}"; PASS=$((PASS+1)); }
bad ()  { printf "  ${R}FAIL${N}  %-52s %s\n" "$1" "${2:-}"; FAIL=$((FAIL+1)); }
head2 () { printf "\n${B}%s${N}\n" "$1"; }

echo "NorthStar Lab 2 verification - account ${ACCOUNT_ID}, region ${REGION}"

# ── Task 1: infrastructure ─────────────────────────────────────────────────────
head2 "Task 1 - Platform infrastructure (25 pts)"

SUBNET=$(aws ec2 describe-subnets --filters "Name=tag:Name,Values=${P}-private-1" \
  --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
[ "${SUBNET}" != "None" ] && [ -n "${SUBNET}" ] \
  && ok "private subnet exists" "${SUBNET}" \
  || bad "private subnet exists" "not found"

NAT=$(aws ec2 describe-nat-gateways --filter "Name=state,Values=available" \
  --query 'NatGateways[0].NatGatewayId' --output text 2>/dev/null)
[ "${NAT}" != "None" ] && [ -n "${NAT}" ] \
  && ok "NAT gateway available" "${NAT}" \
  || bad "NAT gateway available" "none in available state"

DOMAIN_ID=$(aws sagemaker list-domains --query 'Domains[0].DomainId' --output text 2>/dev/null)
if [ "${DOMAIN_ID}" != "None" ] && [ -n "${DOMAIN_ID}" ]; then
  DSTATUS=$(aws sagemaker describe-domain --domain-id "${DOMAIN_ID}" --query 'Status' --output text)
  DSUBNET=$(aws sagemaker describe-domain --domain-id "${DOMAIN_ID}" --query 'SubnetIds[0]' --output text)
  DNET=$(aws sagemaker describe-domain --domain-id "${DOMAIN_ID}" --query 'AppNetworkAccessType' --output text)
  [ "${DSTATUS}" = "InService" ] && ok "domain InService" "${DOMAIN_ID}" || bad "domain InService" "${DSTATUS}"
  [ "${DSUBNET}" = "${SUBNET}" ] && ok "domain in private subnet" "${DSUBNET}" \
    || bad "domain in private subnet" "in ${DSUBNET}, expected ${SUBNET}"
  [ "${DNET}" = "VpcOnly" ] && ok "domain AppNetworkAccessType VpcOnly" || bad "domain VpcOnly" "${DNET}"
else
  bad "SageMaker domain exists" "none found"
fi

for role in MLEngineer DataEngineer ModelMonitor; do
  aws iam get-role --role-name "${P}-${role}" >/dev/null 2>&1 \
    && ok "IAM role ${role} exists" || bad "IAM role ${role} exists" "not found"
done

# IAM boundaries. implicitDeny is the expected result for a denied action.
simulate () { # arn action resource
  aws iam simulate-principal-policy --policy-source-arn "$1" --action-names "$2" \
    --resource-arns "$3" --query 'EvaluationResults[0].EvalDecision' --output text 2>/dev/null
}
DE_ARN=$(aws iam get-role --role-name "${P}-DataEngineer" --query 'Role.Arn' --output text 2>/dev/null)
MM_ARN=$(aws iam get-role --role-name "${P}-ModelMonitor" --query 'Role.Arn' --output text 2>/dev/null)
if [ -n "${DE_ARN}" ] && [ "${DE_ARN}" != "None" ]; then
  r=$(simulate "${DE_ARN}" s3:PutObject "arn:aws:s3:::${BUCKET}/artifacts/models/m.tar.gz")
  [ "${r}" = "implicitDeny" ] || [ "${r}" = "explicitDeny" ] \
    && ok "DataEngineer cannot write artifacts/" "${r}" \
    || bad "DataEngineer cannot write artifacts/" "${r}"
  r=$(simulate "${DE_ARN}" s3:GetObject "arn:aws:s3:::${BUCKET}/artifacts/glue/transform.py")
  [ "${r}" = "allowed" ] && ok "DataEngineer can read artifacts/glue/" \
    || bad "DataEngineer can read artifacts/glue/" "${r}"
fi
if [ -n "${MM_ARN}" ] && [ "${MM_ARN}" != "None" ]; then
  r=$(simulate "${MM_ARN}" s3:PutObject "arn:aws:s3:::${BUCKET}/artifacts/x")
  [ "${r}" = "implicitDeny" ] || [ "${r}" = "explicitDeny" ] \
    && ok "ModelMonitor cannot write S3" "${r}" \
    || bad "ModelMonitor cannot write S3" "${r}"
fi

RULES=$(aws s3api get-bucket-lifecycle-configuration --bucket "${BUCKET}" \
  --query 'length(Rules)' --output text 2>/dev/null || echo 0)
# 5 rules per the Task 1 rubric: expire-raw-data, expire-raw-versions,
# expire-processed-versions, expire-feature-versions, expire-datacapture.
[ "${RULES}" = "5" ] && ok "S3 lifecycle rules present" "${RULES} rules" \
  || bad "S3 lifecycle rules present" "found ${RULES}, expected 5"

# ── Task 2: ingestion ──────────────────────────────────────────────────────────
head2 "Task 2 - Data ingestion pipeline (25 pts)"

aws glue get-database --name "${DB}" >/dev/null 2>&1 \
  && ok "Glue catalog database ${DB}" || bad "Glue catalog database ${DB}" "not found"

TABLE=$(aws glue get-tables --database-name "${DB}" --query 'TableList[0].Name' --output text 2>/dev/null)
[ -n "${TABLE}" ] && [ "${TABLE}" != "None" ] \
  && ok "crawler registered a table" "${TABLE}" || bad "crawler registered a table" "none"

for job in "${P}-transform" "${P}-feature-engineer"; do
  STATE=$(aws glue get-job-runs --job-name "${job}" --query 'JobRuns[0].JobRunState' --output text 2>/dev/null)
  [ "${STATE}" = "SUCCEEDED" ] && ok "job ${job} SUCCEEDED" \
    || bad "job ${job} SUCCEEDED" "last run: ${STATE}"
done

PROC=$(aws s3 ls "s3://${BUCKET}/processed/customers/" --recursive 2>/dev/null | grep -c "\.parquet" || echo 0)
[ "${PROC}" -gt 0 ] && ok "processed/customers/ has Parquet" "${PROC} file(s)" \
  || bad "processed/customers/ has Parquet" "none"

# ── Data quality assertions on the actual output ──────────────────────────────
# These checks run in an embedded Python block. It writes its own PASS/FAIL
# tally to ${TMP}/dq_counts, and the tally is added to the Summary below. A
# data-quality FAIL therefore fails the whole script, the same as any other
# check. (Before 2026-09-23 these lines were printed but not counted, so a
# grain failure could still end in "0 failed".)
head2 "Data quality - processed and features (Tasks 2 and 3)"

TMP=$(mktemp -d)
aws s3 cp "s3://${BUCKET}/processed/customers/" "${TMP}/processed/" --recursive \
  --exclude "*_glue_temp*" >/dev/null 2>&1
aws s3 cp "s3://${BUCKET}/features/customers/" "${TMP}/features/" --recursive >/dev/null 2>&1

python3 - "${TMP}" <<'PY'
import glob, sys
tmp = sys.argv[1]
tty = sys.stdout.isatty()
G = "\033[32mPASS\033[0m" if tty else "PASS"; R = "\033[31mFAIL\033[0m" if tty else "FAIL"
tally = {"pass": 0, "fail": 0}
def line(status, label, detail=""):
    tally["pass" if status == G else "fail"] += 1
    print(f"  {status}  {label:<52} {detail}")
def write_tally():
    with open(f"{tmp}/dq_counts", "w") as fh:
        fh.write(f"{tally['pass']} {tally['fail']}\n")
try:
    import pandas as pd
    import pyarrow  # noqa: F401  (read_parquet needs an engine)
except ImportError:
    line(R, "pandas/pyarrow available", "pip install pandas pyarrow")
    write_tally()
    sys.exit(0)

def checks():

    # processed
    files = glob.glob(f"{tmp}/processed/*.parquet")
    if not files:
        line(R, "processed Parquet readable", "no files downloaded")
    else:
        df = pd.concat([pd.read_parquet(f) for f in files])
        line(G if df.customer_id.isna().sum()==0 else R,
             "processed: 0 null customer_id", f"{df.customer_id.isna().sum()} nulls")
        dups = df.transaction_id.duplicated().sum()
        line(G if dups==0 else R, "processed: 0 duplicate transaction_id", f"{dups} dups")
        line(G if df.purchase_date.isna().sum()==0 else R,
             "processed: all purchase_date parsed", f"{df.purchase_date.isna().sum()} nulls")
        grain = len(df) > df.customer_id.nunique()
        line(G if grain else R, "processed: transaction-level grain preserved",
             f"{len(df)} rows / {df.customer_id.nunique()} customers")

    # features
    files = glob.glob(f"{tmp}/features/*.parquet")
    if not files:
        line(R, "features Parquet readable", "no files downloaded")
    else:
        fd = pd.concat([pd.read_parquet(f) for f in files])
        nulls = int(fd.isna().sum().sum())
        line(G if nulls==0 else R, "features: no null values", f"{nulls} nulls")
        one_row = len(fd) == fd.customer_id.nunique()
        line(G if one_row else R, "features: one row per customer",
             f"{len(fd)} rows / {fd.customer_id.nunique()} customers")
        lo, hi = fd.churn_risk_score.min(), fd.churn_risk_score.max()
        line(G if lo>=0 and hi<=1 else R, "features: churn_risk_score in [0,1]",
             f"{lo:.3f} - {hi:.3f}")
        expected = ["days_since_last_purchase","customer_tenure_days","purchase_frequency_30d",
                    "purchase_frequency_90d","purchase_frequency_180d","avg_order_value",
                    "total_spend_90d","total_lifetime_value","avg_basket_size_6m",
                    "category_diversity_score","online_to_store_ratio","loyalty_tier",
                    "churn_risk_score","churn_label"]
        missing = [c for c in expected if c not in fd.columns]
        line(G if not missing else R, "features: all 14 columns present",
             "missing: " + ", ".join(missing) if missing else "")
        if "churn_label" in fd.columns:
            rate = fd.churn_label.mean()
            line(G if 0.15 <= rate <= 0.30 else R,
                 "features: churn_label rate plausible (15-30%)", f"{rate:.1%}")
            # Leakage smoke test: a label perfectly separable by recency alone
            # means the temporal split was not applied.
            if "days_since_last_purchase" in fd.columns and fd.churn_label.nunique() > 1:
                hi_r = fd[fd.churn_label==1].days_since_last_purchase.min()
                lo_r = fd[fd.churn_label==0].days_since_last_purchase.max()
                line(G if hi_r < lo_r else R,
                     "features: label not trivially separable by recency",
                     f"churner min recency {hi_r:.0f} vs active max {lo_r:.0f}")
        tiers = set(fd.loyalty_tier.unique())
        want = {"Bronze","Silver","Gold","Platinum"}
        line(G if tiers==want else R, "features: all 4 loyalty tiers present",
             ", ".join(sorted(tiers)))
        line(G if fd.churn_risk_score.nunique()>3 else R,
             "features: churn score non-degenerate", f"{fd.churn_risk_score.nunique()} distinct")

try:
    checks()
except Exception as e:  # a crash (e.g. a missing column) is a failure, not a skip
    line(R, "data-quality checks completed", f"{type(e).__name__}: {e}")
write_tally()
PY
if [ -s "${TMP}/dq_counts" ]; then
  read -r DQ_PASS DQ_FAIL < "${TMP}/dq_counts"
  PASS=$((PASS + DQ_PASS)); FAIL=$((FAIL + DQ_FAIL))
else
  bad "data-quality checks ran" "python3 block did not complete"
fi
rm -rf "${TMP}"

# ── Task 3: Feature Store ──────────────────────────────────────────────────────
head2 "Task 3 - Feature Store (20 pts)"

FG="${P}-customer-features"
FGSTATUS=$(aws sagemaker describe-feature-group --feature-group-name "${FG}" \
  --query 'FeatureGroupStatus' --output text 2>/dev/null)
[ "${FGSTATUS}" = "Created" ] && ok "feature group Created" "${FG}" \
  || bad "feature group Created" "${FGSTATUS}"

NFEAT=$(aws sagemaker describe-feature-group --feature-group-name "${FG}" \
  --query 'length(FeatureDefinitions)' --output text 2>/dev/null || echo 0)
[ "${NFEAT}" = "16" ] && ok "feature group has 16 definitions" \
  || bad "feature group has 16 definitions" "found ${NFEAT}"

LBLTYPE=$(aws sagemaker describe-feature-group --feature-group-name "${FG}" \
  --query "FeatureDefinitions[?FeatureName=='churn_label'].FeatureType" --output text 2>/dev/null)
[ "${LBLTYPE}" = "Integral" ] && ok "churn_label is Integral" \
  || bad "churn_label is Integral" "${LBLTYPE}"

ETTYPE=$(aws sagemaker describe-feature-group --feature-group-name "${FG}" \
  --query "FeatureDefinitions[?FeatureName=='event_time'].FeatureType" --output text 2>/dev/null)
[ "${ETTYPE}" = "Fractional" ] && ok "event_time is Fractional" \
  || bad "event_time is Fractional" "${ETTYPE} - lab requires Fractional (epoch seconds)"

ONLINE=$(aws sagemaker describe-feature-group --feature-group-name "${FG}" \
  --query 'OnlineStoreConfig.EnableOnlineStore' --output text 2>/dev/null)
[ "${ONLINE}" = "True" ] && ok "online store enabled" || bad "online store enabled" "${ONLINE}"

OFFLINE=$(aws sagemaker describe-feature-group --feature-group-name "${FG}" \
  --query 'OfflineStoreConfig.S3StorageConfig.S3Uri' --output text 2>/dev/null)
echo "${OFFLINE}" | grep -q "${BUCKET}" \
  && ok "offline store points at data bucket" \
  || bad "offline store points at data bucket" "${OFFLINE}"

# Online store round trip - the real test that records actually landed.
RID=$(aws s3 ls "s3://${BUCKET}/features/customers/" >/dev/null 2>&1 && echo yes || echo no)
SAMPLE=$(aws sagemaker-featurestore-runtime get-record \
  --feature-group-name "${FG}" --record-identifier-value-as-string "CUST-10000776" \
  --query 'length(Record)' --output text 2>/dev/null || echo 0)
if [ "${SAMPLE}" != "0" ] && [ -n "${SAMPLE}" ] && [ "${SAMPLE}" != "None" ]; then
  ok "GetRecord returns a record" "${SAMPLE} features"
else
  bad "GetRecord returns a record" "empty - check event_time type and that the job ran"
fi

# ── Tasks 4 and 5: deliverables ────────────────────────────────────────────────
head2 "Tasks 4 and 5 - Deliverables and repo quality (30 pts)"

for f in docs/lab2-data-contract.md docs/lab2-data-lineage.png \
         docs/lab2-extend-output.txt infrastructure/modules/glue/main.tf \
         infrastructure/modules/feature_store/main.tf; do
  [ -f "${f}" ] && ok "${f} present" || bad "${f} present" "missing"
done

grep -q "Apply complete" docs/lab2-extend-output.txt 2>/dev/null \
  && ok "extend output shows Apply complete" \
  || bad "extend output shows Apply complete" "not found in file"

check_section () { # label pattern
  grep -qiE "$2" docs/lab2-data-contract.md 2>/dev/null \
    && ok "data contract has $1 section" \
    || bad "data contract has $1 section" "missing"
}
check_section "Schema"            "^#+ *Schema"
check_section "Quality Guarantee" "Quality Guarantee"
check_section "SLA"               "SLA|Service Level"
check_section "Versioning"        "Versioning|Breaking Change"

if terraform -chdir=infrastructure/environments/"${ENVIRONMENT}" fmt -check -recursive >/dev/null 2>&1; then
  ok "terraform fmt clean"
else
  bad "terraform fmt clean" "run: terraform fmt -recursive"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
printf "\n${B}Summary: %d passed, %d failed${N}\n" "${PASS}" "${FAIL}"
if [ "${FAIL}" -gt 0 ]; then
  echo "Fix the FAIL items above before submitting."
  exit 1
fi
echo "All checks passed. Remember to run scripts/teardown-lab2.sh after you submit."
