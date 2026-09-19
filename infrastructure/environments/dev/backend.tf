# Remote state for the dev environment.

terraform {
  backend "s3" {
    bucket         = "northstar-tfstate-060245184393"
    key            = "dev/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "northstar-tfstate-lock"
  }
}
