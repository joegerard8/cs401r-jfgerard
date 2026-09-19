# ── modules/sagemaker ────────────────────────────────────────────────────────
# Required resources (Task B1). Only these belong in this module:
#
#   aws_sagemaker_domain
#   aws_sagemaker_user_profile
#
# A brand-new AWS account has no service-linked role for Studio, and the
# Domain fails to create with a service-linked role error. Fix it once in the
# console (IAM -> Roles -> Create Role -> AWS Service -> SageMaker -> SageMaker
# Studio) and re-apply. See the New Account Bootstrap note in the lab.
#
# The Domain is also the slowest resource here by a wide margin -- several
# minutes to create and to delete. Factor that into your apply/destroy timings
# for Task B2.

# The Domain needs an execution role (default_user_settings.execution_role)
# and the security group from modules/vpc; both arrive as variables. See the
# module call in environments/dev/main.tf.

# TODO: implement the two resources above.
