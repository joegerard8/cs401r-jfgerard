# AWS Account Setup Guide
## CS 401R — Lab Environment Configuration

---

## Required Courses (Complete Before Lab 1)

This course requires two AWS Academy courses. Complete them before the first lab:

1. **AWS Academy Cloud Foundations** — core services, IAM, S3, EC2, VPC fundamentals
2. **AWS Academy Generative AI** — Bedrock, prompt engineering, foundation models

Both are accessed through AWS Academy's Canvas (not associated with BYU's Canvas). They are free and self-paced. Budget approximately 6–8 hours total. Labs 3, 5, and 6 assume fluency with concepts covered in these courses.

---

## Step 1: Create Your Personal AWS Account

You will use a **personal AWS Free Tier account** for all lab work. This gives you up to **$200 in AWS credits** to spend across the semester.

1. Go to [aws.amazon.com/free](https://aws.amazon.com/free) and click **Create a Free Account**
2. Use your personal email (not your `@byu.edu` email — AWS Academy uses that one)
3. Add a valid credit/debit card when prompted — **you will not be charged** as long as you stay within your credit balance.
4. Choose the **Free Plan** during signup, then upgrade to the **Paid Plan** after Step 2 below.

> **Why Paid Plan?** SageMaker Studio is not available on the Free Plan. Upgrading to Paid is free — it only means AWS has your payment method on file. You will not be charged until your credits run out. Credits last for 6 months or until depleted.

### Your $200 Credit Budget

| Credit | Amount | How to Earn |
|--------|--------|-------------|
| Sign-up bonus | $100 | Automatic on account creation |
| Onboarding bonus | up to $100 | Complete AWS onboarding activities (EC2, Bedrock exploration) — these align with Lab 1 work |
| **Total** | **up to $200** | |

Credits expire **6 months from account creation** or when depleted — whichever comes first. Budget carefully.

### Upgrade to Paid Plan (Required for SageMaker)

After creating your account:
1. Go to **AWS Console → Billing → Free Tier**
2. Click **Upgrade to Paid Plan**
3. Confirm your payment method

Your credits are not affected by this. Upgrade is instant.

---

## Step 2: Install Required Tools

Install on your local machine:

```bash
# Terraform (version >= 1.6)
brew install terraform           # macOS
# or: https://developer.hashicorp.com/terraform/downloads

# AWS CLI (version >= 2.0)
brew install awscli
# or: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

# Python 3.11+
brew install python@3.11

# Git
brew install git

# Docker (for LocalStack)
# https://www.docker.com/products/docker-desktop/

# Verify installs
terraform --version
aws --version
python3 --version
git --version
docker --version
```

---

## Step 3: Create an IAM User and Configure AWS CLI

Do **not** use your root account credentials for lab work. Create a dedicated IAM user:

1. **AWS Console → IAM → Users → Create user**
2. Username: `cs401r-lab`
3. Attach policy: `AdministratorAccess` (for lab use — never in production)
4. **Security credentials tab → Create access key → CLI use case**
5. Download the `.csv` file — you will not be able to retrieve the secret key again

Configure the CLI with these credentials:

```bash
aws configure
# AWS Access Key ID: [your key ID, starts with AKIA]
# AWS Secret Access Key: [your secret key]
# Default region name: us-east-1
# Default output format: json
```

**These credentials do not expire.** You configure them once, and they persist until you revoke them.

**Use `us-east-1` for all labs.** SageMaker feature availability is most complete in us-east-1.

Verify your credentials work:

```bash
aws sts get-caller-identity
```

This should return your account ID and user ARN. If it returns an error, re-check your access key configuration.

---

## Step 4: Install Python Dependencies

```bash
pip3 install boto3 sagemaker pandas numpy scikit-learn pyarrow faker
```

For Lab 3 (LLM evaluation):
```bash
pip3 install ragas langchain openai
```

---

## CRITICAL: Cost Controls

Your $200 credit must cover all seven labs plus any experimentation. The primary risk is leaving compute resources running between sessions.

### What's Free (Within Limits)

| Service | Free Allowance | Notes |
|---------|----------------|-------|
| VPC | Free | No charge for VPC, subnets, IGW, route tables |
| IAM | Free | No charge for roles, policies, users |
| S3 | 5 GB storage, 20K GETs, 2K PUTs/month | Covered by Always Free tier |
| DynamoDB | 25 GB storage, 200M requests/month | Terraform state lock table = always free |
| Lambda | 1M requests/month | Always Free |
| CloudWatch | 10 custom metrics, 10 alarms | Always Free |
| ECR | 500 MB storage/month | Always Free |

### What Costs Credits

| Resource | Cost | Rule |
|----------|------|------|
| SageMaker Studio kernels | **~$0.05/hour** (`ml.t3.medium`) — free for first 250 hours across 2 months, then charged | **Shut down after every session. No exceptions.** |
| SageMaker Training Jobs | Varies by instance; `ml.m5.xlarge` ~$0.23/hour | Ephemeral — stop automatically when done |
| SageMaker Endpoints | `ml.m5.xlarge` ~$0.23/hour | **Delete when not in use** |
| NAT Gateway | ~$32–45/month | **Not used in Lab 1.** Lab 2 adds it — destroy between labs. |
| Bedrock | Per token; Claude Haiku is cheapest | Use Haiku (not Sonnet) unless required |

### Hard Rules

1. **Shut down Studio kernels after every session:** File → Shut Down → Shut Down All
2. **Run `terraform destroy` after submitting each lab** — do not leave resources running
3. **Delete SageMaker endpoints immediately** after testing — a single forgotten endpoint costs $167/month if left running
4. **Never run more than one SageMaker Domain** — check the console before applying a new environment

### Set a Budget Alert

```bash
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "CS401R-Lab-Budget",
    "BudgetLimit": {"Amount": "200", "Unit": "USD"},
    "TimeUnit": "ANNUALLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 75
    },
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "YOUR_EMAIL@example.com"}]
  }]'
```

Replace `YOUR_EMAIL@example.com` with your personal email. You will get an alert when you reach $150 (75% of $200).

### End-of-Lab Cleanup

Run this after submitting each lab:

```bash
#!/bin/bash
# cleanup.sh — run after each lab submission

echo "Destroying Terraform resources..."
cd infrastructure/environments/dev
terraform destroy -auto-approve

echo "Deleting any remaining SageMaker endpoints..."
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text | \
  tr '\t' '\n' | \
  xargs -I{} aws sagemaker delete-endpoint --endpoint-name {}

echo "Done. Check the AWS Console to confirm no resources are running."
```

---

## Estimated Cost Per Lab

These estimates assume you shut down resources after each session and destroy with `terraform destroy` after submitting.

| Lab | Key AWS Services                              | Estimated Cost |
|-----|-----------------|----------------|
| 1 | VPC, S3, SageMaker Domain                     | ~$3–6 |
| 2 | Glue ETL, S3, NAT Gateway (destroy after)     | ~$6–10 |
| 3 | SageMaker Training (XGBoost), Bedrock (Haiku) | ~$12–18 |
| 4 | CodePipeline, CodeBuild, SageMaker Training   | ~$8–14 |
| 5 | SageMaker Endpoint (brief), Secrets Manager   | ~$6–10 |
| 6 | SageMaker Endpoint, CloudWatch, Evidently drift job | ~$10–16 |
| 7 | Athena queries, CloudWatch custom metrics     | ~$2–5 |
| **Total** |                                               | **~$47–79** |

You have $200 in credits — there is buffer. But a single forgotten endpoint or NAT Gateway can consume that buffer in a week.

---

## Terraform State Setup

Before running any Terraform, create your state bucket:

```bash
# Run the bootstrap script from your repo root
bash scripts/bootstrap-state.sh
```

This script:
- Creates `s3://northstar-tfstate-{account-id}` with versioning and encryption
- Creates DynamoDB table `northstar-tfstate-lock`
- Patches `backend.tf` with your real account ID
- Is idempotent — safe to run multiple times

---

## First-Use: SageMaker Service-Linked Role

The first time you run `terraform apply` with a SageMaker Domain on a **new AWS account**, the apply may fail with a service-linked role error. This is an AWS account bootstrap issue, not a code error.

**Fix (one-time only):**

1. **AWS Console → IAM → Roles → Create role**
2. **Trusted entity:** AWS service → SageMaker → SageMaker Studio
3. Click through to create — no custom permissions needed
4. Re-run `terraform apply`

This creates `AWSServiceRoleForAmazonSageMakerNotebooks` in your account. AWS should do this automatically on first SageMaker use, but new accounts occasionally miss it. Once created, you will never need to do this again.

---

## Getting Help

- **Cost questions:** Post in the `#lab-help` Canvas discussion with your current spend shown in **AWS Console → Billing → Bills**
- **Terraform errors:** Include the full error output and the `.tf` file in your post
- **Service errors:** Include the AWS request ID from the error response
- **Office hours:** See Canvas for the current schedule
