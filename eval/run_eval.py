"""Retrieval eval runner -- measures recall@k / hit@k over eval/golden.jsonl.

Calls research_agent.retrieve_trials() -- the SAME function the agent's
search_clinical_trials tool uses -- so any retrieval change (hybrid fusion,
reranking, embedding swap) is measured on exactly the production path.

Usage:
    uv run python eval/run_eval.py                     # print metrics
    uv run python eval/run_eval.py --write-baseline    # also save baseline.json
    uv run python eval/run_eval.py -k 20               # change cutoff

Metrics per query:
    recall@k = |top-k NCT IDs ∩ expected| / |expected|
    hit@k    = 1 if any expected NCT ID appears in top-k else 0
Aggregates are means over all golden queries. baseline.json stores the
aggregates plus per-query details so regressions are attributable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Repo root on sys.path so `import research_agent` works when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOLDEN = Path(__file__).parent / "golden.jsonl"
BASELINE = Path(__file__).parent / "baseline.json"


def run(k: int = 20) -> dict:
    import os

    from research_agent import retrieve_trials  # deferred: heavy import chain

    # QDRANT_COLLECTION overrides the collection under test (research_agent's
    # COLLECTION_NAME is a hardcoded constant, not env-driven) -- used to
    # measure a rebuilt collection (e.g. clinical_trials_hybrid) BEFORE
    # swapping it in behind the production alias.
    collection = os.getenv("QDRANT_COLLECTION")

    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    per_query = []
    started = time.perf_counter()
    for row in rows:
        expected = set(row["expected_nct_ids"])
        got = retrieve_trials(row["query"], limit=k, collection=collection)
        got_ids = [r["NCTId"] for r in got if r.get("NCTId")]
        found = expected.intersection(got_ids)
        per_query.append({
            "query": row["query"],
            "expected": len(expected),
            "found": len(found),
            "recall": len(found) / len(expected),
            "hit": bool(found),
        })
        print(f"  recall@{k} {len(found):2d}/{len(expected):2d}  {row['query'][:70]!r}")

    recalls = [q["recall"] for q in per_query]
    summary = {
        "k": k,
        "queries": len(per_query),
        "mean_recall": round(statistics.mean(recalls), 4),
        "median_recall": round(statistics.median(recalls), 4),
        "hit_rate": round(sum(q["hit"] for q in per_query) / len(per_query), 4),
        "zero_recall_queries": sum(1 for q in per_query if q["recall"] == 0),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "per_query": per_query,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    summary = run(k=args.k)
    print(json.dumps({key: v for key, v in summary.items() if key != "per_query"},
                     indent=2))
    if args.write_baseline:
        BASELINE.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"baseline written -> {BASELINE}")


if __name__ == "__main__":
    main()
