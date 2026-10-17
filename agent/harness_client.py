"""
Invoke the Lab 3 Track C agent (an AgentCore Harness) and record evidence.

    from agent.harness_client import HarnessAgent
    agent = HarnessAgent()                       # reads HARNESS_ARN from env or terraform output
    reply, tools = agent.invoke("Where is ORD-2847301?", session_id="demo-1")

`agent.invoke` matches the `invoke_fn(message, session_id) -> (reply, [tool names])`
contract that `evaluation_harness.py`'s LocalAgentEvaluator expects:

    python - <<'EOF'
    import evaluation_harness as eh
    from agent.harness_client import HarnessAgent
    agent = HarnessAgent()
    ev = eh.LocalAgentEvaluator(invoke_fn=agent.invoke)
    ...
    EOF

Every call appends evidence (the rubric's file names):
    docs/lab3-evidence/agent-traces.jsonl   one JSON object per tool call
    docs/lab3-evidence/agent-cost.csv       one row per run

Requires boto3 >= 1.43.57 (the first version this was verified on). Older SDKs
have no `invoke_harness` and fail with an AttributeError.
"""
import csv
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import boto3

EVIDENCE = Path("docs/lab3-evidence")
TF_DIR = Path("infrastructure/environments/agent")

# Haiku 4.5 through the us. inference profile (AWS Price List, us-east-1, 2026-10-05).
COST_PER_INPUT_TOKEN = 1.10 / 1_000_000
COST_PER_OUTPUT_TOKEN = 5.50 / 1_000_000
# Runtime (memory + vCPU) per session with the 60 s idle timeout, measured
# 2026-10-12: $0.020 for six one-message sessions. It is billed per session,
# not per token, and it arrives in CloudWatch 15-20 minutes late, so this is an
# estimate. At the AWS default (900 s) it was $0.018 per session.
RUNTIME_COST_PER_SESSION_EST = 0.0033


def _harness_arn() -> str:
    if os.environ.get("HARNESS_ARN"):
        return os.environ["HARNESS_ARN"]
    r = subprocess.run(["terraform", f"-chdir={TF_DIR}", "output", "-raw", "harness_arn"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.startswith("arn:"):
        raise RuntimeError("Could not read harness_arn. Set HARNESS_ARN, or run from the repo root "
                           f"after terraform apply.\nterraform said: {r.stderr.strip()[-400:]}")
    return r.stdout.strip()


class HarnessAgent:
    def __init__(self, harness_arn: str | None = None, region: str = "us-east-1",
                 record: bool = True, run_label: str = "eval"):
        self.arn = harness_arn or _harness_arn()
        self.client = boto3.client("bedrock-agentcore", region_name=region)
        self.record = record
        self.run_label = run_label
        if record:
            EVIDENCE.mkdir(parents=True, exist_ok=True)

    def invoke(self, message: str, session_id: str | None = None,
               system_prompt: str | None = None):
        """Return (reply_text, [tool names in call order]).

        A NEW runtime session per call unless you pass the same session_id.
        Evaluation scenarios must each get their own session: reusing one
        carries earlier scenarios' history into later ones.
        """
        # runtimeSessionId must be >= 33 characters.
        sid = f"{session_id or 'session'}-{uuid.uuid4()}"
        kwargs = dict(harnessArn=self.arn, runtimeSessionId=sid,
                      messages=[{"role": "user", "content": [{"text": message}]}])
        if system_prompt:
            kwargs["systemPrompt"] = [{"text": system_prompt}]
        t_start = time.time()
        resp = self.client.invoke_harness(**kwargs)

        text, calls, usage, stop = [], [], {"inputTokens": 0, "outputTokens": 0}, None
        open_call = None
        for ev in resp["stream"]:
            now = time.time()
            if "contentBlockStart" in ev:
                start = ev["contentBlockStart"].get("start", {})
                if "toolUse" in start:
                    open_call = {"tool": start["toolUse"].get("name", "").split("___")[-1],
                                 "input": "", "output": "", "t0": now}
                    calls.append(open_call)
                elif "toolResult" in start and calls:
                    calls[-1]["status"] = start["toolResult"].get("status", "success")
                    calls[-1]["latency_ms"] = round((now - calls[-1]["t0"]) * 1000, 1)
            elif "contentBlockDelta" in ev:
                d = ev["contentBlockDelta"].get("delta", {})
                if "text" in d:
                    text.append(d["text"])
                elif "toolUse" in d and open_call is not None:
                    open_call["input"] += d["toolUse"].get("input", "")
                elif "toolResult" in d and calls:
                    for block in d["toolResult"]:
                        calls[-1]["output"] += block.get("text", "") if isinstance(block, dict) else str(block)
            elif "messageStop" in ev:
                stop = ev["messageStop"].get("stopReason")
            elif "metadata" in ev:
                u = ev["metadata"].get("usage", {})
                usage["inputTokens"] += u.get("inputTokens", 0)
                usage["outputTokens"] += u.get("outputTokens", 0)
            else:
                for k in ("validationException", "internalServerException", "runtimeClientError"):
                    if k in ev:
                        raise RuntimeError(f"{k}: {ev[k]}")

        reply = "".join(text)
        tools = [c["tool"] for c in calls]
        if self.record:
            self._record(sid, calls, usage, stop, time.time() - t_start)
        return reply, tools

    def _record(self, sid, calls, usage, stop, seconds):
        with open(EVIDENCE / "agent-traces.jsonl", "a") as f:
            for c in calls:
                try:
                    tool_input = json.loads(c["input"]) if c["input"] else {}
                except json.JSONDecodeError:
                    tool_input = c["input"]
                f.write(json.dumps({
                    "session_id": sid, "tool": c["tool"], "input": tool_input,
                    "output": c["output"][:4000], "status": c.get("status"),
                    # Measured at the client: tool_use block start -> tool_result block start.
                    "latency_ms": c.get("latency_ms"),
                }) + "\n")
        cost_path = EVIDENCE / "agent-cost.csv"
        new = not cost_path.exists()
        model_cost = usage["inputTokens"] * COST_PER_INPUT_TOKEN + usage["outputTokens"] * COST_PER_OUTPUT_TOKEN
        with open(cost_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["run", "session_id", "input_tokens", "output_tokens", "model_cost_usd",
                            "runtime_cost_est_usd", "total_est_usd", "tool_calls", "stop_reason", "seconds"])
            w.writerow([self.run_label, sid, usage["inputTokens"], usage["outputTokens"],
                        round(model_cost, 6), RUNTIME_COST_PER_SESSION_EST,
                        round(model_cost + RUNTIME_COST_PER_SESSION_EST, 6), len(calls), stop, round(seconds, 1)])
