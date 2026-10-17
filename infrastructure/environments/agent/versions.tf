# Lab 3 Track C: the AgentCore agent stack.
#
# This is a SEPARATE Terraform root from environments/dev (Labs 1-2), on purpose.
# Your Labs 1-2 root pins hashicorp/aws ~> 5.0, and no 5.x release has a single
# aws_bedrockagentcore_* resource. AgentCore needs the 6.x provider. Leave the
# Labs 1-2 root alone; this root has its own pin and its own state file.
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.67.0" # exact: the version this stack was verified on
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project = "northstar"
      Lab     = "lab3-trackc"
    }
  }
}
