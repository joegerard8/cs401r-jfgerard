# ── modules/storage ──────────────────────────────────────────────────────────
# Required resources (Task B1). Only these belong in this module:
#
#   aws_s3_bucket
#   aws_s3_bucket_public_access_block
#   aws_s3_bucket_versioning
#   aws_s3_bucket_server_side_encryption_configuration
#   aws_s3_object  x4                 the raw/ processed/ features/ artifacts/ prefixes
#
# ONE bucket with four prefixes, not four buckets. Later labs derive the name
# as ${project}-${environment}-data-${account_id}, so keep that shape.
#
# The four aws_s3_object resources create the prefixes. S3 has no real
# directories; an empty object with a trailing slash is how a prefix is made
# to exist before anything is written to it.

# TODO: implement the resources above.
