data "aws_caller_identity" "current" {}

resource "aws_iam_role" "ml_engineer" {
  name = "${var.project}-${var.environment}-MLEngineer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "ml_engineer" {
  name = "${var.project}-${var.environment}-MLEngineerPolicy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SageMakerCore"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateMlflowApp",
          "sagemaker:DescribeMlflowApp",
          "sagemaker:ListMlflowApps",
          "sagemaker:CreatePresignedMlflowAppUrl",
          "sagemaker:RegisterModel",
          "sagemaker:DescribeModelPackage",
          "sagemaker:ListModelPackages",
        ]
        Resource = "*"
      },
      {
        Sid    = "StudioSelfService"
        Effect = "Allow"
        Action = [
          "sagemaker:DescribeDomain",
          "sagemaker:ListDomains",
          "sagemaker:DescribeUserProfile",
          "sagemaker:ListUserProfiles",
          "sagemaker:DescribeSpace",
          "sagemaker:ListSpaces",
          "sagemaker:CreateSpace",
          "sagemaker:UpdateSpace",
          "sagemaker:DeleteSpace",
          "sagemaker:AddTags",
          "sagemaker:DescribeApp",
          "sagemaker:ListApps",
          "sagemaker:CreateApp",
          "sagemaker:DeleteApp",
          "sagemaker:CreatePresignedDomainUrl",
        ]
        Resource = [
          "arn:aws:sagemaker:*:*:domain/*",
          "arn:aws:sagemaker:*:*:user-profile/*",
          "arn:aws:sagemaker:*:*:space/*",
          "arn:aws:sagemaker:*:*:app/*",
        ]
      },
      {
        Sid    = "S3ArtifactsAndFeatures"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = [
          "arn:aws:s3:::${var.project}-${var.environment}-data-${data.aws_caller_identity.current.account_id}/artifacts/*",
          "arn:aws:s3:::${var.project}-${var.environment}-data-${data.aws_caller_identity.current.account_id}/features/*",
        ]
      },
      {
        Sid      = "S3BucketList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = "arn:aws:s3:::${var.project}-${var.environment}-data-${data.aws_caller_identity.current.account_id}"
      },
      {
        Sid      = "CloudWatchLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      },
      {
        Sid      = "ECRRead"
        Effect   = "Allow"
        Action   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:GetAuthorizationToken"]
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ml_engineer" {
  role       = aws_iam_role.ml_engineer.name
  policy_arn = aws_iam_policy.ml_engineer.arn
}

locals {
  bucket_arn = "arn:aws:s3:::${var.project}-${var.environment}-data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_iam_role" "data_engineer" {
  name = "${var.project}-${var.environment}-DataEngineer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = ["glue.amazonaws.com", "lambda.amazonaws.com", "sagemaker.amazonaws.com"]
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "data_engineer" {
  name = "${var.project}-${var.environment}-DataEngineerPolicy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "Glue"
        Effect   = "Allow"
        Action   = ["glue:*"]
        Resource = "*"
      },
      {
        Sid    = "NetworkInterfaces"
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DeleteNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeVpcEndpoints",
          "ec2:DescribeRouteTables",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeVpcs",
          "ec2:DescribeVpcAttribute",
        ]
        Resource = "*"
      },
      {
        Sid      = "NetworkInterfaceTags"
        Effect   = "Allow"
        Action   = ["ec2:CreateTags", "ec2:DeleteTags"]
        Resource = "arn:aws:ec2:*:${data.aws_caller_identity.current.account_id}:network-interface/*"
      },
      {
        Sid    = "S3Data"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = [
          "${local.bucket_arn}/raw/*",
          "${local.bucket_arn}/processed/*",
          "${local.bucket_arn}/features/*",
          # Glue writes these folder markers on overwrite
          "${local.bucket_arn}/raw_$folder$",
          "${local.bucket_arn}/processed_$folder$",
          "${local.bucket_arn}/features_$folder$",
        ]
      },
      {
        Sid      = "GlueScripts"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${local.bucket_arn}/artifacts/glue/*"
      },
      {
        Sid      = "S3BucketLocationAndAcl"
        Effect   = "Allow"
        Action   = ["s3:GetBucketLocation", "s3:GetBucketAcl"]
        Resource = local.bucket_arn
      },
      {
        Sid      = "FeatureStoreOfflineAcl"
        Effect   = "Allow"
        Action   = ["s3:PutObjectAcl"]
        Resource = "${local.bucket_arn}/features/*"
      },
      {
        # No prefix condition, Spark lists parent paths like "processed" on overwrite
        Sid      = "S3BucketList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = local.bucket_arn
      },
      {
        Sid    = "FeatureStore"
        Effect = "Allow"
        Action = [
          "sagemaker:PutRecord",
          "sagemaker:CreateFeatureGroup",
          "sagemaker:DescribeFeatureGroup",
        ]
        Resource = "arn:aws:sagemaker:*:${data.aws_caller_identity.current.account_id}:feature-group/${var.project}-${var.environment}-*"
      },
      {
        Sid      = "CloudWatchLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/aws-glue/*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "data_engineer" {
  role       = aws_iam_role.data_engineer.name
  policy_arn = aws_iam_policy.data_engineer.arn
}

resource "aws_iam_role" "model_monitor" {
  name = "${var.project}-${var.environment}-ModelMonitor"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "model_monitor" {
  name = "${var.project}-${var.environment}-ModelMonitorPolicy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatch"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DescribeAlarms",
        ]
        Resource = "*"
      },
      {
        Sid      = "ProcessingJobs"
        Effect   = "Allow"
        Action   = ["sagemaker:ListProcessingJobs", "sagemaker:DescribeProcessingJob"]
        Resource = "*"
      },
      {
        Sid      = "S3Artifacts"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${local.bucket_arn}/artifacts/*"
      },
      {
        Sid      = "S3BucketLocation"
        Effect   = "Allow"
        Action   = ["s3:GetBucketLocation"]
        Resource = local.bucket_arn
      },
      {
        Sid      = "S3BucketList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = local.bucket_arn
        Condition = {
          StringLike = {
            "s3:prefix" = "artifacts/*"
          }
        }
      },
      {
        Sid      = "CloudWatchLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "model_monitor" {
  role       = aws_iam_role.model_monitor.name
  policy_arn = aws_iam_policy.model_monitor.arn
}
