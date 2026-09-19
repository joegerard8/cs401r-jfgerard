## ADR-001: NorthStar Platform Foundation

### Status
Accepted

### Context

NorthStar wants to use AI to reduce customer churn. About 18% of its 2.1 million active customers become inactive each year. The project estimates about $140 million in yearly opportunity, with around $340 in value connected to each customer. The first use case is a weekly churn model that predicts 30-day risk and focuses on the top 10% of risky customers.

The platform will also support a RAG offer generator and a customer service agent. The offer generator needs to respond within two seconds. The service agent needs 99.5% availability during business hours. The platform will use data from stores, e-commerce, CRM, loyalty systems, clickstream data, products, and store operations.

Some of this data can include personal information. The project also has to consider GDPR, CCPA, and NorthStar's 24-month retention policy. Because of this, the platform needs basic networking, storage, and permissions before the actual ML systems are built.

### Decision

For Lab 1, I used one AWS VPC named `northstar-dev-vpc` with CIDR `10.0.0.0/16`. It has one public subnet named `northstar-dev-public-1`, using `10.0.100.0/24` in `us-east-1a`. The subnet uses an Internet Gateway and has a default route through `0.0.0.0/0`. The SageMaker security group only allows inbound traffic from the VPC CIDR.

I used one S3 bucket for the project. The AWS account ID is part of the bucket name so different students do not run into the global S3 naming problem. The bucket has versioning, SSE-S3 encryption, and public access blocked. It has four prefixes:

- `raw/`
- `processed/`
- `features/`
- `artifacts/`

The MLEngineer role can work with the features and artifacts areas, but it cannot write to raw or processed data. This helps prevent someone working on a model from changing the original data.

I created the `northstar-dev-MLEngineer` IAM role with SageMaker as the trusted service. It includes the SageMaker, S3, logging, and ECR permissions needed for the lab. I also added the Studio permissions needed to create spaces, apps, and tags. SageMaker Studio is connected to the VPC, subnet, security group, and role. The Domain is set to delete its EFS file system when it is destroyed so cleanup is easier.

### Consequences

#### What this makes easy

The churn model has a consistent place to read features and save model files. The other two AI systems can use the same storage setup later. The prefixes make it easier to tell raw data from processed data and model outputs. The account ID in the bucket name avoids bucket-name conflicts.

Terraform also makes the VPC and subnet repeatable. The same resources can be rebuilt without manually clicking through the AWS console.

#### What this makes harder

This is a simple lab network, not a production network. Studio is in a public subnet and there are no private subnets or NAT Gateway yet. The role permissions are basic and do not cover every governance need. Later work would need better monitoring, lifecycle rules, PII handling, and data-quality checks, including the transactions with unknown customer IDs.

This lab also does not build the real-time two-second offer system or the service agent's 99.5% availability setup. Those requirements would need more services and testing.

#### What would cause me to revisit this decision

I would change this design before production if the workloads became larger, the data became more sensitive, or the service needed its full availability target. A security review, a need for private network paths, or costs moving too far above the $85,000 monthly platform budget would also be reasons to change it. The retention policy, PII in search queries, and FCRA/ECOA requirements for credit-related offers would also need stronger controls.

### Alternative Considered

The main alternative was to start with private subnets and a NAT Gateway. That would be safer for a production setup, but it would add more routing and troubleshooting work to the first lab. A NAT Gateway also costs money even when the lab is mostly idle. I chose the public-subnet design for Lab 1 so the basic relationships were easier to build and test. Private subnets and NAT would make more sense in a later lab.

### AWS Service Selection

- **Networking:** VPC, subnet, route table, Internet Gateway, and security group provide the basic network boundary.
- **Storage:** S3 provides encrypted, versioned storage for raw data, processed data, features, and artifacts.
- **Identity:** IAM provides the SageMaker execution role and controls which project resources it can access.
- **ML environment:** SageMaker Studio provides the workspace for building and testing the churn model and future AI systems.
