# ── environments/local ───────────────────────────────────────────────────────
# Same modules as environments/dev, pointed at LocalStack instead of AWS.
# Nothing here costs money. Used only by Task B5 (make local-validate).
#
# Do not add a backend block: local state is fine for a throwaway emulator.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # LocalStack accepts any credentials; these keep the provider from reading
  # your real ~/.aws profile.
  access_key = "test"
  secret_key = "test"

  # Skip the calls that only make sense against real AWS.
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  skip_region_validation      = true
  s3_use_path_style           = true

  endpoints {
    ec2 = "http://localhost:4566"
    iam = "http://localhost:4566"
    s3  = "http://localhost:4566"
    sts = "http://localhost:4566"
  }

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
