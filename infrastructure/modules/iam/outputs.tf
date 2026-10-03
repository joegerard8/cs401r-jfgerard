output "ml_engineer_role_arn" {
  description = "ARN of the MLEngineer role — later labs pass this to SageMaker"
  value       = aws_iam_role.ml_engineer.arn
}

output "data_engineer_role_arn" {
  description = "ARN of the DataEngineer role"
  value       = aws_iam_role.data_engineer.arn
}

output "model_monitor_role_arn" {
  description = "ARN of the ModelMonitor role"
  value       = aws_iam_role.model_monitor.arn
}
