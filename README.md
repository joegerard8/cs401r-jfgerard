# cs401r-jfgerard

NorthStar Retail AI platform for CS 401R. All infrastructure is Terraform in `infrastructure/`.

## Layout

```
infrastructure/
  environments/dev/     real AWS (S3 remote state)
  environments/local/   LocalStack validation (vpc, storage, iam only)
  modules/
    vpc/                VPC, public + private subnet, IGW, NAT Gateway, SageMaker security group
    storage/            data bucket (raw/, processed/, features/, artifacts/) + lifecycle rules
    iam/                MLEngineer, DataEngineer, ModelMonitor roles
    sagemaker/          SageMaker Domain (private subnet, VpcOnly) + MLEngineer user profile
    glue/               catalog database, raw crawler, VPC connection, transform + feature-engineer jobs
    feature_store/      customer Feature Group (online + offline store)
glue-scripts/           transform.py, feature_engineer.py (uploaded to artifacts/glue/ by Terraform)
scripts/                verify and teardown scripts
docs/                   lab evidence, data contract, diagrams
```

## Lab 2 additions

- **vpc**: private subnet `10.0.1.0/24` with a NAT Gateway for outbound traffic, and a self-referencing ingress rule on the security group so Glue can run inside the VPC. `enable_nat_gateway = false` in local.
- **storage**: 5 lifecycle rules (raw/ expires after 90 days, old versions on raw/ and processed/ after 30 days and features/ after 60, datacapture/ after 7 days). `enable_lifecycle_rules = false` in local.
- **iam**: `DataEngineer` (Glue, Feature Store writes, read/write on raw/, processed/, features/, read on artifacts/glue/) and `ModelMonitor` (CloudWatch metrics, read on artifacts/).
- **sagemaker**: Domain moved to the private subnet with `VpcOnly` networking.
- **glue** (new): `northstar_dev` catalog database, `northstar-dev-raw-crawler` on `raw/customers/`, `northstar-dev-vpc-connection`, and the two Glue 4.0 jobs.
- **feature_store** (new): `northstar-dev-customer-features` with 16 feature definitions, keyed on `customer_id` with a Fractional `event_time`.

## Data pipeline

```
raw/customers/ (CSV) -> crawler -> northstar_dev.customers
  -> northstar-dev-transform -> processed/customers/ (Parquet, one row per transaction)
  -> northstar-dev-feature-engineer -> features/customers/ (Parquet, one row per customer)
                                    -> Feature Store (PutRecord)
```

The transform job trims whitespace, parses both date formats, drops rows with no `customer_id`, fills nulls (median / `unknown`), and dedupes on `transaction_id`. The feature job computes 13 features from purchases on or before 2026-04-01 and sets `churn_label` from whether the customer bought anything between 2026-04-01 and 2026-06-30.

## Running it end to end

```bash
# 1. Deploy
cd infrastructure/environments/dev
terraform init
terraform apply
cd ../../..

# 2. Upload the raw data
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws s3 cp northstar-raw-sample.csv s3://northstar-dev-data-$ACCOUNT/raw/customers/northstar-raw-sample.csv

# 3. Crawl raw/customers/
aws glue start-crawler --name northstar-dev-raw-crawler
until [ "$(aws glue get-crawler --name northstar-dev-raw-crawler --query 'Crawler.State' --output text)" = "READY" ]; do sleep 15; done

# 4. Run the transform, then the feature job once it SUCCEEDS
aws glue start-job-run --job-name northstar-dev-transform
aws glue get-job-runs --job-name northstar-dev-transform --query 'JobRuns[0].JobRunState'
aws glue start-job-run --job-name northstar-dev-feature-engineer
aws glue get-job-runs --job-name northstar-dev-feature-engineer --query 'JobRuns[0].JobRunState'

# 5. Check a record landed in the online store
aws sagemaker-featurestore-runtime get-record \
  --feature-group-name northstar-dev-customer-features \
  --record-identifier-value-as-string CUST-10000776

# 6. Verify
bash scripts/verify-lab2.sh
```

LocalStack check: `make local-validate LOCAL_OUT=docs/lab2-localstack-output.txt`

## Teardown

The NAT Gateway bills hourly, so tear down when done:

```bash
bash scripts/teardown-lab2.sh
```

`terraform destroy` alone leaves Glue ENIs, the Studio EFS, SageMaker security groups, and Feature Store leftovers behind; the script cleans those up first.
