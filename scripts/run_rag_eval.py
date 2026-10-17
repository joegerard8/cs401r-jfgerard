"""
Run the Track B RAGAS harness repeatedly, per pipeline configuration.

    python scripts/run_rag_eval.py --configs starter starter+profile v2-prompt --runs 2
    python scripts/run_rag_eval.py --configs v2-multi final --runs 3

Each run calls evaluation_harness.RAGEvaluator unchanged (same four test
cases, same Bedrock judge) and writes its raw output to
evaluation_results/rag_eval_<config>_run<k>_<ts>.csv, as the harness does.

Then, across every run in evaluation_results/ (so separate invocations
accumulate), it writes:
    docs/lab3-evidence/rag_eval.csv          per-case scores, every run, with the
                                             judge and generator model named
    docs/lab3-evidence/rag_eval_summary.csv  per config x metric: mean of run
                                             means, min and max run mean
    docs/lab3-evidence/rag_generations.jsonl answers, retrieved chunk IDs and
                                             scores per case, for the examples doc

A run in which any judge call returned NaN (usually Bedrock throttling) is
re-run rather than averaged: NaN means the judge failed, not that the answer
scored zero.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import evaluation_harness as eh                      # noqa: E402
from offer_rag import CONFIGS, GENERATION_MODEL_ID, OfferRAG   # noqa: E402

METRICS = ["faithfulness", "answer_relevancy", "context_recall"]
RESULTS_DIR = REPO_ROOT / "evaluation_results"
EVIDENCE_DIR = REPO_ROOT / "docs" / "lab3-evidence"


def run_once(rag: OfferRAG, cfg_name: str, judge: str, max_attempts: int) -> pd.DataFrame:
    cfg = CONFIGS[cfg_name]
    for attempt in range(1, max_attempts + 1):
        n_before = len(rag.log)
        evaluator = eh.RAGEvaluator(rag_invoke_fn=rag.invoke_fn(cfg),
                                    bedrock_model_id=judge, region=rag_region())
        df = evaluator.run(eh.OFFER_TEST_CASES)
        gens = rag.log[n_before:]
        df["customer_id"] = [g["customer_id"] for g in gens]
        df["chunk_ids"] = [";".join(g["chunk_ids"]) for g in gens]
        df["attempt"] = attempt
        if not df[METRICS].isna().any().any():
            return df, gens
        print(f"  NaN judge score in {cfg_name} attempt {attempt}; re-running")
    raise RuntimeError(f"{cfg_name}: judge returned NaN on {max_attempts} attempts")


def rag_region() -> str:
    return os.environ.get("AWS_REGION", "us-east-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["final"], choices=list(CONFIGS))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--judge-model-id", default=eh.DEFAULT_BEDROCK_MODEL_ID)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    # The harness's contexts carry no customer_id; add it so every output row
    # and log record can be traced back to its test case.
    for tc in eh.OFFER_TEST_CASES:
        tc.customer_context["customer_id"] = tc.customer_id

    RESULTS_DIR.mkdir(exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    rag = OfferRAG(region=rag_region())
    print(f"Index: {len(rag.chunks)} chunks embedded with Titan V2 "
          f"({rag.embedder.input_tokens} tokens)")

    for cfg_name in args.configs:
        existing = len(glob.glob(str(RESULTS_DIR / f"rag_eval_{cfg_name}_run*.csv")))
        for r in range(existing + 1, existing + args.runs + 1):
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            print(f"\n=== {cfg_name} run {r} ===")
            df, gens = run_once(rag, cfg_name, args.judge_model_id, args.max_attempts)
            df.insert(0, "config", cfg_name)
            df.insert(1, "run", r)
            df["judge_model"] = args.judge_model_id
            df["generator_model"] = GENERATION_MODEL_ID
            df["timestamp_utc"] = ts
            df.to_csv(RESULTS_DIR / f"rag_eval_{cfg_name}_run{r}_{ts}.csv", index=False)
            with open(RESULTS_DIR / f"rag_generations_{cfg_name}_run{r}_{ts}.jsonl",
                      "w", encoding="utf-8") as f:
                for g, (_, row) in zip(gens, df.iterrows()):
                    g = {**g, "run": r, **{m: row[m] for m in METRICS}}
                    f.write(json.dumps(g) + "\n")

    print(f"\nGenerator usage this invocation: {rag.usage}")
    aggregate()


def aggregate():
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(RESULTS_DIR / "rag_eval_*_run*.csv")))]
    allruns = pd.concat(frames, ignore_index=True)
    order = {name: i for i, name in enumerate(CONFIGS)}
    allruns = allruns.sort_values(["config", "run"], key=lambda s: s.map(order) if s.name == "config" else s)
    allruns.to_csv(EVIDENCE_DIR / "rag_eval.csv", index=False)

    run_means = allruns.groupby(["config", "run"])[METRICS].mean().reset_index()
    rows = []
    for cfg, g in run_means.groupby("config"):
        for m in METRICS:
            rows.append({"config": cfg, "metric": m, "runs": len(g),
                         "mean": round(g[m].mean(), 4), "min_run": round(g[m].min(), 4),
                         "max_run": round(g[m].max(), 4)})
    summary = pd.DataFrame(rows).sort_values(["config", "metric"],
                                             key=lambda s: s.map(order) if s.name == "config" else s)
    summary["judge_model"] = allruns["judge_model"].iloc[0]
    summary.to_csv(EVIDENCE_DIR / "rag_eval_summary.csv", index=False)

    with open(EVIDENCE_DIR / "rag_generations.jsonl", "w", encoding="utf-8") as out:
        for p in sorted(glob.glob(str(RESULTS_DIR / "rag_generations_*_run*.jsonl"))):
            out.write(Path(p).read_text(encoding="utf-8"))

    print("\n── Summary (mean of per-run means; range across runs) ──")
    print(summary.to_string(index=False))
    print("\nPer-case means by config:")
    print(allruns.groupby(["config", "customer_id"])[METRICS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
