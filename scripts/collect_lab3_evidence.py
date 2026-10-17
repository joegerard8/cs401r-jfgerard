#!/usr/bin/env python3
"""collect_lab3_evidence.py -- export and check the evidence Lab 3 is graded on.

The TA grades your `lab3-submit` tag and has NO access to your AWS account.
Anything that lives only in AWS -- your MLflow runs, your Model Registry entry,
the Athena query that pulled your training data -- does not exist as far as
grading is concerned until it is exported into the repository. This script does
that export, then checks every required evidence file and cross-checks the
numbers between them.

Run it AFTER your final training run and evaluation, and BEFORE teardown:

    bash scripts/collect-lab3-evidence.sh --track B      # or --track C

Writes to docs/lab3-evidence/:
    account.json            who ran this (ties the evidence to your account)
    athena-query.json       the Athena query named in train-final.log
    mlflow-app.json         your MLflow App (name, role, artifact store)
    mlflow-runs.csv         every run in the experiment, params + metrics
    model-registry.json     every version in northstar-churn-models
    collect-output.txt      this script's report (written by the .sh wrapper)

Exit status is non-zero if any check FAILs. WARN does not fail the run, but
read each one -- most are things a grader will notice.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

try:
    import boto3
except ImportError:
    sys.exit("boto3 is not installed in this Python. Activate the environment you train in.")

EVIDENCE_FILES_ALL = [
    ("docs/lab3-evidence/train-final.log",         "Task 1: console log of your FINAL training run (tee)"),
    ("docs/lab3-evidence/evaluation_metrics.json", "Task 1: written by the training script (--output-data-dir docs/lab3-evidence)"),
    ("docs/lab3-evidence/slice-evaluation.csv",    "Task 1: per-tier n, positives, AUC, recall"),
    ("docs/lab3-evidence/confusion-matrix.png",    "Task 1: at your chosen threshold"),
    ("docs/lab3-evidence/feature-importance.png",  "Task 1: from the registered model"),
    ("docs/lab3-evaluation-report.md",             "Task 1: the evaluation report"),
    ("docs/lab3-model-design.md",                  "Task 3: design justification (track declared at top)"),
    ("docs/bedrock-access-verification.txt",       "Task 4: from Pre-Lab 3"),
    ("notebooks/lab3-demo.ipynb",                  "Task 2: committed WITH outputs from one clean Restart & Run All"),
]
EVIDENCE_FILES_TRACK = {
    "B": [("docs/lab3-evidence/rag_eval.csv",       "Task 2B: evaluation_harness.py --track B output, renamed"),
          ("docs/lab3-evidence/track-b-examples.md", "Task 2B: 5 input/output pairs + 2 failure cases with fixes")],
    "C": [("docs/lab3-evidence/agent_eval.csv",     "Task 2C: evaluation_harness.py --track C output, renamed"),
          ("docs/lab3-evidence/agent-traces.jsonl", "Task 2C: one JSON line per tool call (tool, input, output, latency_ms)"),
          ("docs/lab3-evidence/agent-cost.csv",     "Task 2C: tokens and cost per run"),
          ("docs/lab3-evidence/track-c-failures.md", "Task 2C: >=2 failure modes, each with a concrete fix"),
          ("docs/lab3-evidence/policy-tests.json",  "Task 2C: python agent/policy_tests.py (the $50 cap, enforced at the gateway)")],
}
SHARED_METRICS = ("auc_roc", "precision_top10", "recall_top10")
TOL = 6e-5            # every figure is rounded to 4 dp; allow for that only

results = []          # (level, item, detail)


def report(level, item, detail=""):
    results.append((level, item, detail))
    print(f"  {level:4s}  {item}" + (f"  --  {detail}" if detail else ""))


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── AWS exports ──────────────────────────────────────────────────────────────

def export_agentcore(sess, out):
    """Track C: what the TA cannot see in your account. Harness settings, the
    gateway and its tools, and every Cedar policy, as AgentCore recorded them."""
    c = sess.client("bedrock-agentcore-control")
    harnesses = c.list_harnesses().get("harnesses", [])
    if not harnesses:
        report("FAIL", "AgentCore harness", "none found. Run this BEFORE teardown, while the stack is up")
        return
    doc = {"harnesses": [], "gateways": [], "policy_engines": []}
    for h in harnesses:
        hid = h.get("harnessId") or h["arn"].split("/")[-1]
        d = c.get_harness(harnessId=hid)
        d = d.get("harness", d)
        d.pop("ResponseMetadata", None)
        env = (d.get("environment") or {}).get("agentCoreRuntimeEnvironment", {})
        idle = (env.get("lifecycleConfiguration") or {}).get("idleRuntimeSessionTimeout")
        doc["harnesses"].append(d)
        if idle is None or idle > 300:
            report("WARN", f"harness {d.get('harnessName')}", f"idle timeout {idle}s. Runtime memory bills for every idle second; use 60")
        else:
            report("OK", f"harness {d.get('harnessName')}", f"idle timeout {idle}s, {len(d.get('tools') or [])} tool source(s)")
    for g in c.list_gateways().get("items", []):
        gd = c.get_gateway(gatewayIdentifier=g["gatewayId"])
        gd.pop("ResponseMetadata", None)
        gd["targets"] = c.list_gateway_targets(gatewayIdentifier=g["gatewayId"]).get("items", [])
        doc["gateways"].append(gd)
        mode = (gd.get("policyEngineConfiguration") or {}).get("mode")
        report("OK" if mode == "ENFORCE" else "FAIL", f"gateway {gd.get('name')}", f"policy engine mode {mode}")
    for e in c.list_policy_engines().get("policyEngines", []):
        pols = []
        for pol in c.list_policies(policyEngineId=e["policyEngineId"]).get("policies", []):
            pd = c.get_policy(policyEngineId=e["policyEngineId"], policyId=pol["policyId"])
            pd.pop("ResponseMetadata", None)
            pols.append(pd)
        doc["policy_engines"].append({"engine": e, "policies": pols})
        stmts = " ".join(json.dumps(p.get("definition", {})) for p in pols)
        has_forbid = "forbid" in stmts and "apply_loyalty_credit" in stmts
        report("OK" if has_forbid else "FAIL", f"policy engine {e.get('name')}",
               f"{len(pols)} policies" + ("" if has_forbid else "; no forbid on apply_loyalty_credit (TODO 3)"))
    dump(os.path.join(out, "agentcore-config.json"), doc)


def export_identity(sess, out):
    ident = sess.client("sts").get_caller_identity()
    ident.pop("ResponseMetadata", None)
    ident["CollectedAtUTC"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    dump(os.path.join(out, "account.json"), ident)
    report("OK", f"account {ident['Account']}", ident["Arn"])
    return ident["Account"]


def export_athena(sess, out, log_text):
    """The query ID printed in train-final.log, looked up in Athena itself."""
    if log_text is None:
        report("FAIL", "Athena query", "train-final.log missing, so there is no query to look up")
        return
    if "Loading features from local file" in log_text:
        report("FAIL", "Athena query", "train-final.log shows the --local-data-path CSV loader; "
               "the graded run must read the Feature Store through Athena")
    m = re.search(r"Athena QueryExecutionId:\s*([0-9a-f-]{36})", log_text)
    if not m:
        report("FAIL", "Athena query", "no 'Athena QueryExecutionId: <id>' line in train-final.log "
               "(print it after query.run -- see the lab)")
        return
    qid = m.group(1)
    try:
        q = sess.client("athena").get_query_execution(QueryExecutionId=qid)["QueryExecution"]
    except Exception as e:
        report("FAIL", "Athena query", f"{qid} not found in this account/region: {e}")
        return
    rec = {
        "QueryExecutionId": qid,
        "Query": q.get("Query"),
        "State": q["Status"]["State"],
        "Submitted": q["Status"].get("SubmissionDateTime"),
        "Database": q.get("QueryExecutionContext", {}).get("Database"),
        "DataScannedBytes": q.get("Statistics", {}).get("DataScannedInBytes"),
        "OutputLocation": q.get("ResultConfiguration", {}).get("OutputLocation"),
    }
    dump(os.path.join(out, "athena-query.json"), rec)
    if rec["State"] != "SUCCEEDED":
        report("FAIL", "Athena query", f"{qid} state {rec['State']}")
    elif not re.search(r"sagemaker_featurestore|feature", (rec["Database"] or "") + (rec["Query"] or ""), re.I):
        report("WARN", "Athena query", "succeeded, but neither database nor SQL names the Feature Store "
               "offline store -- check it read the Lab 2 feature group")
    elif not rec["DataScannedBytes"]:
        report("WARN", "Athena query", f"{qid} scanned 0 bytes -- a real pull of ~10,000 customers "
               "scans data; this query did not read the offline store")
    else:
        report("OK", "Athena query", f"{qid} SUCCEEDED, {rec['DataScannedBytes']} bytes scanned")


def find_mlflow_app(sm, name):
    apps = sm.list_mlflow_apps().get("Summaries", [])
    for a in apps:
        if a.get("Name") == name:
            return a.get("Arn")
    if len(apps) == 1:
        return apps[0].get("Arn")
    return None


def export_mlflow(sess, out, app_name, experiment, eval_metrics):
    sm = sess.client("sagemaker")
    servers = sm.list_mlflow_tracking_servers().get("TrackingServerSummaries", [])
    live = [s["TrackingServerName"] for s in servers if s.get("TrackingServerStatus") not in ("Deleted", "Deleting")]
    if live:
        report("WARN", "MLflow Tracking Server exists", f"{', '.join(live)} -- $0.60/hr each; "
               "delete it (the App is the free one)")
    arn = find_mlflow_app(sm, app_name)
    if not arn:
        report("FAIL", "MLflow App", f"no App named {app_name!r} (use --mlflow-app-name)")
        return
    app = sm.describe_mlflow_app(Arn=arn)
    app.pop("ResponseMetadata", None)
    dump(os.path.join(out, "mlflow-app.json"), app)
    role, store = app.get("RoleArn", ""), app.get("ArtifactStoreUri", "")
    if "MLEngineer" in role and "/artifacts/" in store:
        report("OK", "MLflow App reuses Lab 1 infrastructure", f"{role.split('/')[-1]}, {store}")
    else:
        report("WARN", "MLflow App infrastructure", f"role {role or '?'}, store {store or '?'} -- "
               "Task 4 expects the Lab 1 MLEngineer role and an artifacts/ prefix")

    try:
        import mlflow
    except ImportError:
        report("FAIL", "MLflow runs", "mlflow is not installed (pip install mlflow sagemaker-mlflow)")
        return
    mlflow.set_tracking_uri(arn)
    try:
        runs = mlflow.search_runs(experiment_names=[experiment])
    except Exception as e:
        report("FAIL", "MLflow runs", f"search_runs failed: {e}")
        return
    runs.to_csv(os.path.join(out, "mlflow-runs.csv"), index=False)
    pcols = [c for c in runs.columns if c.startswith("params.")]
    mcols = [c for c in runs.columns if c.startswith("metrics.")]
    complete = runs[runs[pcols].notna().any(axis=1) & runs[mcols].notna().any(axis=1)] if pcols and mcols else runs.iloc[0:0]
    n = len(complete)
    if n >= 3:
        report("OK", "MLflow runs", f"{n} runs with params and metrics in {experiment!r}")
    else:
        report("FAIL", "MLflow runs", f"{n} runs with params AND metrics in {experiment!r}; need >= 3")
    if n:
        varied = [c for c in pcols if c not in ("params.seed", "params.random_state")
                  and complete[c].astype(str).nunique() > 1]
        if not varied:
            report("WARN", "MLflow runs", "no hyperparameter differs between runs -- "
                   "that is one experiment repeated, and earns 2 of 5")
        else:
            report("OK", "MLflow runs vary", ", ".join(c[7:] for c in varied))
    if eval_metrics and "metrics.auc_roc" in runs.columns:
        auc = as_float(eval_metrics.get("auc_roc"))
        if auc is not None and not (abs(runs["metrics.auc_roc"] - auc) < TOL).any():
            report("FAIL", "consistency: MLflow", f"no run logged auc_roc={auc}; the graded run "
                   "must be one of your tracked runs")
        elif auc is not None:
            report("OK", "consistency: MLflow", f"a tracked run matches auc_roc={auc}")


def export_registry(sess, out, group, eval_metrics):
    sm = sess.client("sagemaker")
    try:
        pkgs = sm.list_model_packages(ModelPackageGroupName=group, MaxResults=100)["ModelPackageSummaryList"]
    except Exception as e:
        report("FAIL", "Model Registry", f"group {group!r}: {e}")
        return
    full = []
    for p in pkgs:
        d = sm.describe_model_package(ModelPackageName=p["ModelPackageArn"])
        full.append({k: d.get(k) for k in ("ModelPackageArn", "ModelPackageVersion", "ModelApprovalStatus",
                                            "CreationTime", "ModelPackageDescription",
                                            "CustomerMetadataProperties")})
    dump(os.path.join(out, "model-registry.json"), {"ModelPackageGroupName": group, "Versions": full})
    pending = [v for v in full if v["ModelApprovalStatus"] == "PendingManualApproval"]
    approved = [v for v in full if v["ModelApprovalStatus"] == "Approved"]
    if approved:
        report("WARN", "Model Registry", f"version(s) {[v['ModelPackageVersion'] for v in approved]} are "
               "Approved -- Lab 3 must never auto-approve")
    if not pending:
        report("FAIL", "Model Registry", f"no PendingManualApproval version in {group!r} "
               f"({len(full)} version(s) found)")
        return
    report("OK", "Model Registry", f"{len(pending)} PendingManualApproval version(s); latest "
           f"v{max(v['ModelPackageVersion'] for v in pending)}")
    if not eval_metrics:
        return
    want = {k: as_float(eval_metrics.get(k)) for k in SHARED_METRICS}
    match = [v for v in pending if all(
        want[k] is not None and as_float((v["CustomerMetadataProperties"] or {}).get(k)) is not None
        and abs(as_float(v["CustomerMetadataProperties"][k]) - want[k]) < TOL for k in SHARED_METRICS)]
    if match:
        report("OK", "consistency: registry", f"v{match[-1]['ModelPackageVersion']} metadata matches "
               "evaluation_metrics.json")
    else:
        report("FAIL", "consistency: registry", "no PendingManualApproval version carries the "
               f"{'/'.join(SHARED_METRICS)} in evaluation_metrics.json (CustomerMetadataProperties)")


# ── Local evidence checks ────────────────────────────────────────────────────

def check_files(track):
    missing = 0
    for path, why in EVIDENCE_FILES_ALL + EVIDENCE_FILES_TRACK[track]:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            report("OK", path)
        else:
            report("FAIL", path, f"missing or empty -- {why}")
            missing += 1
    return missing


def check_metrics():
    p = "docs/lab3-evidence/evaluation_metrics.json"
    if not os.path.isfile(p):
        return None
    try:
        m = json.load(open(p))
    except ValueError as e:
        report("FAIL", "evaluation_metrics.json", f"not valid JSON: {e}")
        return None
    need = ("auc_roc", "precision_top10", "recall_top10", "baseline_auc", "lift_ci_low", "lift_ci_high", "n_val_samples")
    gone = [k for k in need if k not in m]
    if gone:
        report("FAIL", "evaluation_metrics.json", f"missing keys {gone}")
        return m
    gates = [("precision_top10 >= 0.50", m["precision_top10"] >= 0.50),
             ("recall_top10 >= 0.25", m["recall_top10"] >= 0.25),
             ("lift 95% CI excludes zero", m["lift_ci_low"] > 0)]
    for name, ok in gates:
        report("OK" if ok else "WARN", f"gate {name}", "" if ok else "below the rubric threshold")
    return m


def check_slices():
    p = "docs/lab3-evidence/slice-evaluation.csv"
    if not os.path.isfile(p):
        return
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = " ".join(rows[0].keys()).lower() if rows else ""
    lacking = [c for c in ("tier", "n", "auc", "recall") if not re.search(rf"\b{c}", cols)]
    if not rows or lacking:
        report("FAIL", "slice-evaluation.csv", f"needs columns for tier, n, auc, recall (missing: {lacking or 'all rows'})")
    else:
        report("OK", "slice-evaluation.csv", f"{len(rows)} tiers")


def check_notebook():
    p = "notebooks/lab3-demo.ipynb"
    if not os.path.isfile(p):
        return
    try:
        nb = json.load(open(p))
    except ValueError as e:
        report("FAIL", "demo notebook", f"not valid JSON: {e}")
        return
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code" and "".join(c.get("source", "")).strip()]
    with_out = [c for c in code if c.get("outputs")]
    counts = [c.get("execution_count") for c in code]
    if not code or not with_out:
        report("FAIL", "demo notebook", "committed without outputs -- Restart & Run All, then save and commit")
    elif counts != list(range(1, len(code) + 1)):
        report("WARN", "demo notebook", "execution counts are not 1..N -- outputs are not from one clean "
               "top-to-bottom run")
    else:
        report("OK", "demo notebook", f"{len(code)} code cells, one clean run, {len(with_out)} with output")


def check_track(track):
    if track == "C":
        p = "docs/lab3-evidence/agent-traces.jsonl"
        if os.path.isfile(p):
            recs, bad = [], 0
            for line in open(p):
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        bad += 1
            keys = set().union(*(r.keys() for r in recs)) if recs else set()
            lacking = [k for k in ("tool", "input", "output", "latency") if not any(k in x for x in keys)]
            if bad or not recs or lacking:
                report("FAIL", "agent-traces.jsonl", f"{len(recs)} records, {bad} unparseable, missing fields {lacking}")
            else:
                report("OK", "agent-traces.jsonl", f"{len(recs)} tool calls")
        p = "docs/lab3-evidence/policy-tests.json"
        if os.path.isfile(p):
            rs = json.load(open(p)).get("results", [])
            over = [r for r in rs if r.get("tool") == "apply_loyalty_credit" and (r.get("arguments") or {}).get("amount", 0) > 50]
            if over and all(r.get("decision") == "DENY" for r in over):
                report("OK", "policy-tests.json", f"{len(over)} over-cap credits denied at the gateway")
            else:
                report("FAIL", "policy-tests.json", "an over-cap credit was not denied. Check TODO 3 and re-run agent/policy_tests.py")
    p = f"docs/lab3-evidence/{'rag' if track == 'B' else 'agent'}_eval.csv"
    if os.path.isfile(p):
        with open(p, newline="") as f:
            n = sum(1 for _ in csv.DictReader(f))
        want = 4 if track == "B" else 5
        report("OK" if n >= want else "FAIL", os.path.basename(p), f"{n} test cases (harness has {want}+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--track", choices=["B", "C"], required=True)
    ap.add_argument("--mlflow-app-name", default="northstar-mlflow")
    ap.add_argument("--experiment", default="northstar-churn")
    ap.add_argument("--model-package-group", default="northstar-churn-models")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    a = ap.parse_args()

    out = "docs/lab3-evidence"
    if not os.path.isdir("docs"):
        sys.exit("Run this from the root of your cs401r-<netid> repository (no docs/ here).")
    os.makedirs(out, exist_ok=True)
    sess = boto3.Session(region_name=a.region)

    print(f"==> Lab 3 evidence  (track {a.track}, {a.region})")
    print("[1/5] Identity")
    export_identity(sess, out)
    print("[2/5] Evidence files")
    check_files(a.track)
    print("[3/5] Content checks")
    m = check_metrics()
    check_slices()
    check_notebook()
    check_track(a.track)
    print("[4/5] AWS exports: Athena, MLflow, Model Registry" + (", AgentCore" if a.track == "C" else ""))
    log = open(f"{out}/train-final.log", errors="replace").read() if os.path.isfile(f"{out}/train-final.log") else None
    for step in (lambda: export_athena(sess, out, log),
                 lambda: export_mlflow(sess, out, a.mlflow_app_name, a.experiment, m),
                 lambda: export_registry(sess, out, a.model_package_group, m),
                 *([lambda: export_agentcore(sess, out)] if a.track == "C" else [])):
        try:
            step()
        except Exception as e:          # one failed export must not hide the others
            report("FAIL", "AWS export", f"{type(e).__name__}: {e}")
    print("[5/5] Summary")
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print(f"  {len(results) - len(fails) - len(warns)} OK, {len(warns)} WARN, {len(fails)} FAIL")
    if fails:
        print("  Fix every FAIL, re-run, then teardown -> commit -> tag.")
        sys.exit(1)
    print("  Evidence complete. Next: bash scripts/teardown-lab3.sh, then commit and tag lab3-submit.")


if __name__ == "__main__":
    main()
