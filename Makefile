# NorthStar AI Platform — Lab 1
# Run from the repo root.
#
#   make local-validate   Task B5: apply vpc/storage/iam against LocalStack and
#                         save the evidence to docs/lab1b-localstack-output.txt
#   make local-destroy    tear the LocalStack stack down
#   make local-clean      also drop local state and LocalStack data

INFRA_DIR   ?= infrastructure
LOCAL_ENV    = $(INFRA_DIR)/environments/local
LOCAL_OUT    = docs/lab1b-localstack-output.txt

.PHONY: local-validate local-destroy local-clean

local-validate:
	@docker compose up -d --wait
	@mkdir -p docs
	@set -e; { \
	  echo "== NorthStar Lab 1 — LocalStack validation =="; \
	  echo "date: $$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	  echo; \
	  echo "== terraform init =="; \
	  terraform -chdir=$(LOCAL_ENV) init -input=false -no-color; \
	  echo; \
	  echo "== terraform apply =="; \
	  terraform -chdir=$(LOCAL_ENV) apply -auto-approve -input=false -no-color; \
	  echo; \
	  echo "== awslocal sts get-caller-identity =="; \
	  awslocal sts get-caller-identity; \
	  echo; \
	  BUCKET=$$(terraform -chdir=$(LOCAL_ENV) output -raw s3_bucket_name 2>/dev/null) \
	    || { echo "ERROR: output s3_bucket_name is not defined in $(LOCAL_ENV)/outputs.tf — uncomment it"; exit 1; }; \
	  echo "== awslocal s3 ls s3://$$BUCKET/ --recursive  (name from: terraform output s3_bucket_name) =="; \
	  awslocal s3 ls s3://$$BUCKET/ --recursive; \
	  echo; \
	  echo "== awslocal iam list-roles (northstar*) =="; \
	  awslocal iam list-roles --query 'Roles[?starts_with(RoleName, `northstar`)].RoleName'; \
	  echo; \
	  echo "== awslocal ec2 describe-vpcs =="; \
	  awslocal ec2 describe-vpcs --query 'Vpcs[*].{Id:VpcId,CIDR:CidrBlock}'; \
	  echo; \
	  echo "== awslocal ec2 describe-subnets =="; \
	  awslocal ec2 describe-subnets --query 'Subnets[*].{Id:SubnetId,AZ:AvailabilityZone,CIDR:CidrBlock}'; \
	} 2>&1 | tee $(LOCAL_OUT)
	@echo
	@echo "Saved to $(LOCAL_OUT) — commit it."

local-destroy:
	terraform -chdir=$(LOCAL_ENV) destroy -auto-approve -input=false

local-clean: local-destroy
	docker compose down
	rm -rf $(LOCAL_ENV)/.terraform $(LOCAL_ENV)/terraform.tfstate* .localstack
