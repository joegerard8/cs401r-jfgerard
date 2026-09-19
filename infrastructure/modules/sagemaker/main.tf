# SageMaker uses the shared execution role and security group from the VPC module.
resource "aws_sagemaker_domain" "this" {
  domain_name = "${var.project}-${var.environment}-domain"
  auth_mode   = "IAM"
  vpc_id      = var.vpc_id
  subnet_ids  = var.subnet_ids

  app_network_access_type = "PublicInternetOnly"

  default_user_settings {
    execution_role  = var.execution_role_arn
    security_groups = var.security_group_ids

    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = var.instance_type
      }
    }
  }

  retention_policy {
    home_efs_file_system = "Delete"
  }

  tags = {
    Name = "${var.project}-${var.environment}-domain"
  }
}

resource "aws_sagemaker_user_profile" "ml_engineer" {
  domain_id         = aws_sagemaker_domain.this.id
  user_profile_name = "MLEngineer"

  user_settings {
    execution_role  = var.execution_role_arn
    security_groups = var.security_group_ids
  }

  tags = {
    Name = "${var.project}-${var.environment}-MLEngineer"
  }
}
