output "feature_group_name" {
  description = "Name of the customer Feature Group"
  value       = aws_sagemaker_feature_group.customers.feature_group_name
}
