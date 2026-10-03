# SageMaker is omitted because it is not available in LocalStack Community.
module "vpc" {
  source             = "../../modules/vpc"
  project            = var.project
  environment        = var.environment
  enable_nat_gateway = false
}

module "storage" {
  source                 = "../../modules/storage"
  project                = var.project
  environment            = var.environment
  enable_lifecycle_rules = false
}

module "iam" {
  source      = "../../modules/iam"
  project     = var.project
  environment = var.environment
}
