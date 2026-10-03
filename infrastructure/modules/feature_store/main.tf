locals {
  fractional_features = [
    "event_time",
    "days_since_last_purchase",
    "customer_tenure_days",
    "purchase_frequency_30d",
    "purchase_frequency_90d",
    "purchase_frequency_180d",
    "avg_order_value",
    "total_spend_90d",
    "total_lifetime_value",
    "avg_basket_size_6m",
    "category_diversity_score",
    "online_to_store_ratio",
    "churn_risk_score",
  ]
}

resource "aws_sagemaker_feature_group" "customers" {
  feature_group_name             = "${var.project}-${var.environment}-customer-features"
  record_identifier_feature_name = "customer_id"
  event_time_feature_name        = "event_time"
  role_arn                       = var.role_arn

  feature_definition {
    feature_name = "customer_id"
    feature_type = "String"
  }

  dynamic "feature_definition" {
    for_each = local.fractional_features
    content {
      feature_name = feature_definition.value
      feature_type = "Fractional"
    }
  }

  feature_definition {
    feature_name = "loyalty_tier"
    feature_type = "String"
  }

  feature_definition {
    feature_name = "churn_label"
    feature_type = "Integral"
  }

  online_store_config {
    enable_online_store = true
  }

  offline_store_config {
    s3_storage_config {
      s3_uri = "s3://${var.bucket_name}/features/offline-store/"
    }
  }
}
