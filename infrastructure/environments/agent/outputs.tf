output "harness_arn" { value = aws_bedrockagentcore_harness.this.arn }
output "gateway_id" { value = aws_bedrockagentcore_gateway.this.gateway_id }
output "gateway_url" { value = aws_bedrockagentcore_gateway.this.gateway_url }
output "policy_engine_id" { value = aws_bedrockagentcore_policy_engine.this.policy_engine_id }
output "tools_lambda_name" { value = aws_lambda_function.tools.function_name }
