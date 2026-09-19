# SageMaker is omitted because it is not available in LocalStack Community.
module "vpc" {
  source      = "../../modules/vpc"
  project     = var.project
  environment = var.environment
}

module "storage" {
  source      = "../../modules/storage"
  project     = var.project
  environment = var.environment
}

module "iam" {
  source      = "../../modules/iam"
  project     = var.project
  environment = var.environment
}
