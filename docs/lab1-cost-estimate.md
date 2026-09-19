## Monthly Cost Estimate

### Context

This is a small Lab 1 estimate for the NorthStar development environment. It assumes one data scientist using SageMaker Studio part time. It does not include larger production workloads, training jobs, endpoints, Bedrock usage, or a NAT Gateway.

### Estimate

| Component | Monthly Estimate | Key Assumptions | One Optimization |
|---|---:|---|---|
| SageMaker Studio | **$2.40** | 1 data scientist, 1 `ml.t3.medium` instance, 4 hours/day for 12 days/month. 48 hours at about $0.05/hour. | Shut down Studio after each session. |
| S3 storage | **$0.50** | About 20 GB in the data bucket at $0.023/GB, plus a small amount for requests. | Move older data to a cheaper storage class if it is not used often. |
| Internet Gateway | **$0.10** | About 10 GB of internet data transfer at the lab estimate of $0.01/GB. | Keep workloads small and avoid unnecessary downloads. |
| DynamoDB state lock | **$0.01** | One on-demand lock table with very few reads and writes. | Keep one shared lock table instead of creating extra tables. |
| S3 state bucket | **$0.03** | About 1 GB of Terraform state and a small number of requests. | Delete old state versions if they are no longer needed. |
| **Total** | **$3.04/month** |  |  |

### Assumptions and Limits

The estimate is before AWS credits or Free Tier discounts. SageMaker Studio itself does not have a separate hourly charge. The charge comes from the notebook compute that runs inside Studio. The `ml.t3.medium` estimate is about $0.05 per hour.

The S3 estimate assumes small lab data volumes. The actual data bucket uses versioning, so storing many old versions would increase the cost. The state bucket and DynamoDB table should stay very small because they only support Terraform.

This estimate is for the resources in Lab 1. It does not represent the full production NorthStar platform. A production system would also need more compute, monitoring, training, model hosting, data transfer, and possibly private subnets with a NAT Gateway.

### Quantified Optimization

The biggest easy saving is shutting down SageMaker Studio when it is not being used. At 48 hours per month, the estimate is $2.40. If the same `ml.t3.medium` instance ran all month, it would cost about $36.00 for 720 hours. Shutting it down when finished saves about **$33.60 per month** before credits.
