variable "project" {
  description = "Project name, used as the first element of every resource name"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "bucket_name" {
  description = "Data bucket backing the offline store"
  type        = string
}

variable "role_arn" {
  description = "ARN of the DataEngineer role Feature Store uses for the offline store"
  type        = string
}
