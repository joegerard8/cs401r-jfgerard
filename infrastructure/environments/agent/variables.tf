variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for IAM roles and the tools Lambda"
  type        = string
  default     = "northstar-agent"
}

variable "gateway_name" {
  type    = string
  default = "northstar-tools-gw"
}

variable "harness_name" {
  description = "Letters, digits, underscores; max 40 chars"
  type        = string
  default     = "northstar_agent"
}

variable "credit_cap_dollars" {
  description = "Largest loyalty credit the gateway will let the agent apply in one call"
  type        = number
  default     = 50
}

variable "idle_session_timeout_seconds" {
  description = <<-EOT
    Seconds an idle harness session keeps running. Memory is billed for that
    whole time. The AWS default is 900 s; measured on the same six calls,
    900 s cost $0.106 of runtime and 60 s cost $0.020. Do not raise it.
  EOT
  type        = number
  default     = 60
}

variable "max_session_lifetime_seconds" {
  type    = number
  default = 3600
}
