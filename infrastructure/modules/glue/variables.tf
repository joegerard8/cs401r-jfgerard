variable "project" {
  description = "Project name, used as the first element of every resource name"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "bucket_name" {
  description = "Name of the data bucket holding raw/, processed/ and artifacts/glue/"
  type        = string
}

variable "role_arn" {
  description = "ARN of the DataEngineer role used by the crawler and jobs"
  type        = string
}

variable "subnet_id" {
  description = "Private subnet the Glue workers run in"
  type        = string
}

variable "availability_zone" {
  description = "Availability Zone of subnet_id"
  type        = string
}

variable "security_group_ids" {
  description = "Security groups for the Glue connection; one must have a self-referencing all-ports ingress rule"
  type        = list(string)
}

variable "transform_script_path" {
  description = "Local path to glue-scripts/transform.py"
  type        = string
}

variable "feature_script_path" {
  description = "Local path to glue-scripts/feature_engineer.py"
  type        = string
}

variable "feature_group_name" {
  description = "SageMaker Feature Group the feature engineering job ingests into"
  type        = string
}

variable "region" {
  description = "AWS region for the Feature Store runtime client"
  type        = string
}
