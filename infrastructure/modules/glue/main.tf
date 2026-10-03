locals {
  prefix      = "${var.project}-${var.environment}"
  script_key  = "artifacts/glue/transform.py"
  fe_key      = "artifacts/glue/feature_engineer.py"
  bucket_path = "s3://${var.bucket_name}"
}

resource "aws_glue_catalog_database" "this" {
  name = replace(local.prefix, "-", "_")
}

resource "aws_glue_crawler" "raw" {
  name          = "${local.prefix}-raw-crawler"
  database_name = aws_glue_catalog_database.this.name
  role          = var.role_arn

  s3_target {
    path = "${local.bucket_path}/raw/customers/"
  }
}

resource "aws_glue_connection" "vpc" {
  name            = "${local.prefix}-vpc-connection"
  connection_type = "NETWORK"

  physical_connection_requirements {
    availability_zone      = var.availability_zone
    subnet_id              = var.subnet_id
    security_group_id_list = var.security_group_ids
  }
}

resource "aws_s3_object" "transform_script" {
  bucket = var.bucket_name
  key    = local.script_key
  source = var.transform_script_path
  etag   = filemd5(var.transform_script_path)
}

resource "aws_glue_job" "transform" {
  name              = "${local.prefix}-transform"
  role_arn          = var.role_arn
  glue_version      = "4.0"
  worker_type       = "G.1X"
  number_of_workers = 2
  timeout           = 30
  connections       = [aws_glue_connection.vpc.name]

  command {
    name            = "glueetl"
    python_version  = "3"
    script_location = "${local.bucket_path}/${aws_s3_object.transform_script.key}"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--enable-continuous-cloudwatch-log" = "true"
    "--database_name"                    = aws_glue_catalog_database.this.name
    "--table_name"                       = "customers"
    "--output_path"                      = "${local.bucket_path}/processed/customers/"
  }
}

resource "aws_s3_object" "feature_script" {
  bucket = var.bucket_name
  key    = local.fe_key
  source = var.feature_script_path
  etag   = filemd5(var.feature_script_path)
}

resource "aws_glue_job" "feature_engineer" {
  name              = "${local.prefix}-feature-engineer"
  role_arn          = var.role_arn
  glue_version      = "4.0"
  worker_type       = "G.1X"
  number_of_workers = 2
  timeout           = 30
  connections       = [aws_glue_connection.vpc.name]

  command {
    name            = "glueetl"
    python_version  = "3"
    script_location = "${local.bucket_path}/${aws_s3_object.feature_script.key}"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--enable-continuous-cloudwatch-log" = "true"
    "--input_path"                       = "${local.bucket_path}/processed/customers/"
    "--output_path"                      = "${local.bucket_path}/features/customers/"
    "--feature_group_name"               = var.feature_group_name
    "--region"                           = var.region
  }
}
