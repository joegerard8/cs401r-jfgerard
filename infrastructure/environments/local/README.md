# What `make local-validate` is doing

Task B5 hands you one command. Read this so you know what it verifies and what
its output means. Later labs assume you can read this output without help.

## The command sequence

```bash
docker compose up -d --wait
terraform -chdir=infrastructure/environments/local init
terraform -chdir=infrastructure/environments/local apply -auto-approve

awslocal sts get-caller-identity
awslocal s3 ls s3://northstar-local-data-000000000000/ --recursive
awslocal iam list-roles --query 'Roles[?starts_with(RoleName, `northstar`)].RoleName'
awslocal ec2 describe-vpcs --query 'Vpcs[*].{Id:VpcId,CIDR:CidrBlock}'
awslocal ec2 describe-subnets --query 'Subnets[*].{Id:SubnetId,AZ:AvailabilityZone,CIDR:CidrBlock}'
```

Everything is written to `docs/lab1b-localstack-output.txt`. That file is your
evidence for B5.

## 1. Start LocalStack

`docker compose up -d --wait` starts the containers in `docker-compose.yml`
(here, just LocalStack), detached (`-d`), and blocks until the healthcheck
passes (`--wait`). Without `--wait`, Terraform would race the container and
fail with a connection refused.

LocalStack is an AWS emulator listening on `localhost:4566`. It answers the
same API calls as AWS for the services you enable (`SERVICES=s3,iam,ec2,sts`
in the compose file). It does not emulate SageMaker in the free edition, which
is why this environment omits the `sagemaker` module.

## 2. Apply the Terraform

`environments/local` calls the same `vpc`, `storage`, and `iam` modules as
`environments/dev`. The only differences are in `versions.tf`:

- `endpoints { ... = "http://localhost:4566" }` routes every API call to
  LocalStack instead of AWS.
- `access_key = "test"` / `secret_key = "test"`: LocalStack accepts any
  credentials. Setting fake ones keeps the provider from reading your real
  `~/.aws` profile, so you cannot accidentally create resources in AWS.
- `skip_*` flags turn off the provider's start-up checks that only make sense
  against real AWS.
- `environment = "local"` so every resource name reads `northstar-local-*`
  and can never collide with `northstar-dev-*`.

If `apply` fails here, your module code is wrong, not the emulator. Fix it
before you spend money running it against AWS.

## 3. `awslocal`

`awslocal` is the AWS CLI with the endpoint pre-set to `localhost:4566`.
`awslocal s3 ls` is exactly `aws --endpoint-url=http://localhost:4566 s3 ls`.
Everything you know about the AWS CLI applies.

### `sts get-caller-identity`

"Who am I?" Against AWS this returns your account and IAM principal. Against
LocalStack it always returns account `000000000000`:

```json
{
    "UserId": "AKIAIOSFODNN7EXAMPLE",
    "Account": "000000000000",
    "Arn": "arn:aws:iam::000000000000:root"
}
```

That account ID is why the bucket is `northstar-local-data-000000000000`. The
storage module builds the name as
`${project}-${environment}-data-${account_id}`; in `dev` the suffix is your
real 12-digit account.

### `s3 ls ... --recursive`

Lists every object in the bucket. Expected:

```
2026-09-13 19:26:10          0 artifacts/
2026-09-13 19:26:10          0 features/
2026-09-13 19:26:10          0 processed/
2026-09-13 19:26:10          0 raw/
```

Four zero-byte objects whose keys end in `/`. S3 has no directories; an empty
object with a trailing slash is how a prefix is made to exist before anything
is written under it. Those are your four `aws_s3_object` resources.

### `iam list-roles --query '...'`

`--query` is a JMESPath expression, evaluated client-side on the JSON the API
returns. Reading it left to right:

| Piece | Meaning |
|---|---|
| `Roles` | the array in the response |
| `[?starts_with(RoleName, `northstar`)]` | keep elements whose `RoleName` starts with `northstar` |
| `.RoleName` | project just that field |

Expected: `["northstar-local-MLEngineer"]`. The backticks around `northstar`
are JMESPath string literals, which is why the whole expression sits in single
quotes.

### `ec2 describe-vpcs` and `describe-subnets`

`Vpcs[*].{Id:VpcId,CIDR:CidrBlock}` is a JMESPath multiselect hash: for every
VPC, build a small object with two renamed fields. Same idea for subnets.

Expected, for the VPC you built:

```json
{ "Id": "vpc-...", "CIDR": "10.0.0.0/16" }
{ "Id": "subnet-...", "AZ": "us-east-1a", "CIDR": "10.0.100.0/24" }
```

You will also see a `172.31.0.0/16` VPC with six `/20` subnets, one per AZ.
That is LocalStack emulating AWS's default VPC, which every real region also
has. It is not yours and the rubric ignores it.

## What passing proves, and what it does not

Passing proves your `vpc`, `storage`, and `iam` modules produce the right
resources with the right names, and that your Terraform is syntactically and
semantically sound. That is worth knowing before an `apply` against AWS that
costs money and takes ten minutes to destroy.

It does not prove your IAM policy grants the right permissions (LocalStack
Community does not enforce IAM), that the SageMaker module works (not
emulated), or that the resources behave under load. `scripts/verify-lab1.sh`
against real AWS covers the first two.

## Teardown

```bash
make local-destroy   # terraform destroy in environments/local
make local-clean     # also stop the container and remove local state
```
