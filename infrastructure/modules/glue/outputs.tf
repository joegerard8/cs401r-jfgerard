output "database_name" {
  description = "Name of the Glue catalog database"
  value       = aws_glue_catalog_database.this.name
}

output "crawler_name" {
  description = "Name of the raw/customers/ crawler"
  value       = aws_glue_crawler.raw.name
}

output "transform_job_name" {
  description = "Name of the transform ETL job"
  value       = aws_glue_job.transform.name
}

output "feature_engineer_job_name" {
  description = "Name of the feature engineering ETL job"
  value       = aws_glue_job.feature_engineer.name
}
