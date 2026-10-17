"""AgentCore Gateway Lambda target: dispatches NorthStar tool calls to tools.py.

Gateway passes the tool's arguments as the event and the tool name in
context.client_context.custom["bedrockAgentCoreToolName"] as "<target>___<tool>".
The raw event and context are logged on every call so the contract is on record.
"""
import json
import logging

import tools

log = logging.getLogger()
log.setLevel(logging.INFO)

DISPATCH = {
    "lookup_order": lambda a: tools.lookup_order(a["order_id"]),
    "query_policy": lambda a: tools.query_policy(a["question"]),
    "initiate_return": lambda a: tools.initiate_return(a["order_id"], a.get("reason", "")),
    "apply_loyalty_credit": lambda a: tools.apply_loyalty_credit(
        a["customer_id"], float(a["amount"]), a.get("reason", "")),
    "escalate_to_human": lambda a: tools.escalate_to_human(
        a.get("customer_id", "UNKNOWN"), a.get("issue_summary", ""), a.get("priority", "normal")),
}


def _tool_name(event, context):
    custom = {}
    cc = getattr(context, "client_context", None)
    if cc is not None and getattr(cc, "custom", None):
        custom = cc.custom
    name = custom.get("bedrockAgentCoreToolName") or event.get("toolName") or event.get("name") or ""
    return name.split("___")[-1], custom


def handler(event, context):
    name, custom = _tool_name(event, context)
    log.info("RAW_EVENT %s", json.dumps({"event": event, "client_context_custom": custom}, default=str))
    args = event.get("arguments", event) if isinstance(event, dict) else {}
    fn = DISPATCH.get(name)
    if fn is None:
        return {"error": "UNKNOWN_TOOL", "message": f"Tool '{name}' is not available."}
    try:
        result = fn(args)
    except KeyError as e:
        result = {"error": "MISSING_ARGUMENT", "message": f"Missing required argument: {e}"}
    log.info("TOOL_RESULT %s %s", name, json.dumps(result, default=str)[:2000])
    return result
