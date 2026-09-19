# ── environments/dev ─────────────────────────────────────────────────────────
# Wire the four modules together here. Each module call passes var.project and
# var.environment down; nothing in modules/ hardcodes a name.
#
# Uncomment each block as you implement the module it calls.

# module "vpc" {
#   source             = "../../modules/vpc"
#   project            = var.project
#   environment        = var.environment
#   vpc_cidr           = var.vpc_cidr
#   public_subnet_cidr = var.public_subnet_cidr
#   availability_zone  = var.availability_zone
# }

# module "storage" {
#   source      = "../../modules/storage"
#   project     = var.project
#   environment = var.environment
# }

# module "iam" {
#   source      = "../../modules/iam"
#   project     = var.project
#   environment = var.environment
# }

# module "sagemaker" {
#   source             = "../../modules/sagemaker"
#   project            = var.project
#   environment        = var.environment
#   vpc_id             = module.vpc.vpc_id
#   subnet_ids         = [module.vpc.public_subnet_id]
#   security_group_ids = [module.vpc.security_group_id]
#   execution_role_arn = module.iam.ml_engineer_role_arn
#   instance_type      = var.sagemaker_instance_type
# }
