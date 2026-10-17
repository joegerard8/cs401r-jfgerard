"""
Call the gateway's tools directly (no model in the loop) and record what the
policy engine allows and denies. This is how you prove the $50 cap is enforced
by the platform, not by the prompt.

    python agent/policy_tests.py            # writes docs/lab3-evidence/policy-tests.json

Run it after TODO 3 is in place. Every test calls the real tools Lambda, so an
ALLOW really applies a (mock) credit.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import ssl
import urllib.error
import urllib.request

try:  # python.org macOS builds ship without root certificates
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()

TF_DIR = Path("infrastructure/environments/agent")
TARGET = "northstar-tools"
OUT = Path("docs/lab3-evidence/policy-tests.json")


def gateway_url() -> str:
    if os.environ.get("GATEWAY_URL"):
        return os.environ["GATEWAY_URL"]
    return subprocess.check_output(["terraform", f"-chdir={TF_DIR}", "output", "-raw", "gateway_url"], text=True).strip()


def call_tool(url, tool, args, rid):
    """One MCP tools/call, SigV4-signed with your AWS credentials."""
    body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                       "params": {"name": f"{TARGET}___{tool}", "arguments": args}})
    req = AWSRequest(method="POST", url=url, data=body,
                     headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    SigV4Auth(boto3.Session().get_credentials(), "bedrock-agentcore", "us-east-1").add_auth(req)
    r = urllib.request.Request(url, data=body.encode(), headers=dict(req.headers), method="POST")
    try:
        with urllib.request.urlopen(r, timeout=60, context=_SSL) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"decision": "HTTP_ERROR", "detail": f"{e.code} {e.read().decode()[:300]}"}
    res = payload.get("result") or payload.get("error") or {}
    if isinstance(res, dict) and res.get("isError") is False:
        return {"decision": "ALLOW", "detail": (res.get("content") or [{}])[0].get("text", "")[:300]}
    msg = res.get("message") or (res.get("content") or [{}])[0].get("text", "")
    return {"decision": "DENY" if "Denied" in msg else "ERROR", "detail": msg[:300]}


def main():
    url = gateway_url()
    cases = [
        ("read-only tool", "lookup_order", {"order_id": "ORD-1234567"}, "ALLOW"),
        ("credit under cap", "apply_loyalty_credit", {"customer_id": "CUST-001", "amount": 30, "reason": "policy test"}, "ALLOW"),
        ("credit at cap", "apply_loyalty_credit", {"customer_id": "CUST-001", "amount": 50, "reason": "policy test"}, "ALLOW"),
        ("credit just over cap", "apply_loyalty_credit", {"customer_id": "CUST-001", "amount": 51, "reason": "policy test"}, "DENY"),
        ("credit well over cap", "apply_loyalty_credit", {"customer_id": "CUST-001", "amount": 75, "reason": "policy test"}, "DENY"),
        # The hole a per-call rule cannot close. Explain it in track-c-failures.md.
        ("split credit, part 1 of 2", "apply_loyalty_credit", {"customer_id": "CUST-001", "amount": 40, "reason": "split"}, "ALLOW"),
        ("split credit, part 2 of 2", "apply_loyalty_credit", {"customer_id": "CUST-001", "amount": 40, "reason": "split"}, "ALLOW"),
    ]
    results, failed = [], 0
    for i, (label, tool, args, expected) in enumerate(cases, 1):
        r = call_tool(url, tool, args, i)
        ok = r["decision"] == expected
        failed += not ok
        results.append({"case": label, "tool": tool, "arguments": args, "expected": expected, **r, "as_expected": ok})
        print(f"{'OK ' if ok else 'BAD'} {label:28} expected {expected:5} got {r['decision']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"gateway_url": url, "results": results}, indent=2))
    print(f"\nwrote {OUT}. {len(cases) - failed}/{len(cases)} as expected.")
    if any(r["decision"] == "ALLOW" for r in results[5:7]):
        print("Note: both split credits were allowed ($80 total). Why can't a per-call policy stop this?")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
