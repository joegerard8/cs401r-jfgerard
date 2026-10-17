"""
Show WHY the RAGAS judge scored faithfulness as it did, statement by statement.

    python scripts/explain_faithfulness.py --config final

ragas.evaluate() returns only the score. This re-runs the two steps RAGAS's
Faithfulness metric is made of (statement extraction, then an NLI verdict per
statement against the contexts), with the same Bedrock judge the harness uses,
on the answers and contexts already saved in docs/lab3-evidence/rag_eval.csv.
It writes docs/lab3-evidence/faithfulness_verdicts_<config>.jsonl.

The judge is an LLM, so a re-judged score can differ from the score in
rag_eval.csv for the same answer; both are recorded so the difference is visible.
"""

import argparse
import ast
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import evaluation_harness as eh                 # noqa: E402
from ragas.metrics import faithfulness          # noqa: E402


async def explain(row: dict) -> dict:
    statements = (await faithfulness._create_statements(row, callbacks=None)).statements
    verdicts = await faithfulness._create_verdicts(row, statements, callbacks=None)
    items = [{"statement": v.statement, "supported": bool(v.verdict), "reason": v.reason}
             for v in verdicts.statements]
    score = sum(i["supported"] for i in items) / len(items) if items else float("nan")
    return {"rejudged_faithfulness": round(score, 4), "statements": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="final")
    ap.add_argument("--judge-model-id", default=eh.DEFAULT_BEDROCK_MODEL_ID)
    args = ap.parse_args()

    llm, _ = eh.build_bedrock_judge(args.judge_model_id)
    faithfulness.llm = llm

    df = pd.read_csv(REPO_ROOT / "docs" / "lab3-evidence" / "rag_eval.csv")
    df = df[df["config"] == args.config]
    out_path = REPO_ROOT / "docs" / "lab3-evidence" / f"faithfulness_verdicts_{args.config}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for _, r in df.iterrows():
            row = {"user_input": r["user_input"], "response": r["response"],
                   "retrieved_contexts": ast.literal_eval(r["retrieved_contexts"])}
            res = asyncio.run(explain(row))
            rec = {"config": r["config"], "run": int(r["run"]), "customer_id": r["customer_id"],
                   "harness_faithfulness": r["faithfulness"], **res,
                   "judge_model": args.judge_model_id}
            out.write(json.dumps(rec) + "\n")
            print(f"\n{r['customer_id']} run {rec['run']}: harness {r['faithfulness']:.3f}, "
                  f"re-judged {res['rejudged_faithfulness']:.3f}")
            for s in res["statements"]:
                if not s["supported"]:
                    print(f"  UNSUPPORTED: {s['statement']}\n      reason: {s['reason']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
