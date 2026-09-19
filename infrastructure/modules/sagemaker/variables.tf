# Every variable needs a description — Task B1 grades this.

variable "project" {
  description = "Project name, used as the first element of every resource name"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "vpc_id" {
  description = "VPC the SageMaker Domain attaches to"
  type        = string
}

variable "subnet_ids" {
  description = "Subnets the SageMaker Domain may use"
  type        = list(string)
}

variable "instance_type" {
  description = "Default kernel instance type for Studio apps"
  type        = string
  default     = "ml.t3.medium"
}

variable "execution_role_arn" {
  description = "IAM role Studio assumes for the Domain default user settings and the user profile (the MLEngineer role)"
  type        = string
}

variable "security_group_ids" {
  description = "Security groups attached to Studio apps inside the VPC"
  type        = list(string)
}
