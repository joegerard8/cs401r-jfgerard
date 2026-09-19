# Surface what later labs need. Lab 2 reads these from `terraform output`,
# and scripts/verify-lab1.sh reads ALL FIVE of them with `terraform output
# -raw`, so every one must exist (uncommented and wired) before you run it.

# output "vpc_id" {
#   description = "ID of the VPC"
#   value       = module.vpc.vpc_id
# }

# output "public_subnet_id" {
#   description = "ID of the public subnet"
#   value       = module.vpc.public_subnet_id
# }

# output "s3_bucket_name" {
#   description = "Name of the data bucket"
#   value       = module.storage.bucket_name
# }

# output "ml_engineer_role_arn" {
#   description = "ARN of the MLEngineer role"
#   value       = module.iam.ml_engineer_role_arn
# }

# output "sagemaker_domain_id" {
#   description = "ID of the SageMaker Domain"
#   value       = module.sagemaker.domain_id
# }
