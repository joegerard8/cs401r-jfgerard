# One bucket is used for all four data prefixes.
data "aws_caller_identity" "current" {}

locals {
  bucket_name = "${var.project}-${var.environment}-data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "data" {
  bucket = local.bucket_name
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_object" "prefix" {
  for_each = toset(var.prefixes)

  bucket  = aws_s3_bucket.data.id
  key     = each.value
  content = ""
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  count  = var.enable_lifecycle_rules ? 1 : 0
  bucket = aws_s3_bucket.data.id

  depends_on = [aws_s3_bucket_versioning.data]

  rule {
    id     = "expire-raw-data"
    status = "Enabled"

    filter {
      prefix = "raw/"
    }

    expiration {
      days = 90
    }
  }

  rule {
    id     = "expire-raw-versions"
    status = "Enabled"

    filter {
      prefix = "raw/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "expire-processed-versions"
    status = "Enabled"

    filter {
      prefix = "processed/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "expire-feature-versions"
    status = "Enabled"

    filter {
      prefix = "features/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 60
    }
  }

  rule {
    id     = "expire-datacapture"
    status = "Enabled"

    filter {
      prefix = "datacapture/"
    }

    expiration {
      days = 7
    }
  }
}
