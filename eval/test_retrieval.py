"""Pytest gate: retrieval quality must not regress vs the committed baseline.

Runs the same eval as run_eval.py and asserts aggregate metrics stay within
REGRESSION_MARGIN of eval/baseline.json (regenerate the baseline DELIBERATELY
with `uv run python eval/run_eval.py --write-baseline` whenever a retrieval
change is verified as an improvement -- the diff makes the improvement
reviewable).

Skips (not fails) when Qdrant is unreachable or the collection is too small
to be the real corpus -- so `pytest` stays runnable on machines without the
seeded database, while CI/dev machines with the corpus get the real gate.
Point the eval at a specific instance/collection with QDRANT_HOST /
QDRANT_PORT / QDRANT_COLLECTION, and at a matching baseline file with
EVAL_BASELINE (defaults to eval/baseline.json, which was recorded against
the FULL 597K-point corpus -- a subset corpus needs its own baseline file).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

BASELINE_PATH = Path(os.getenv("EVAL_BASELINE", Path(__file__).parent / "baseline.json"))
# A collection with fewer points than this can't be the corpus the baseline
# was measured on -- refuse to produce misleading green/red numbers.
MIN_POINTS = int(os.getenv("EVAL_MIN_POINTS", "500000"))
REGRESSION_MARGIN = 0.05


def _qdrant_ready() -> tuple[bool, str]:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            timeout=10,
        )
        collection = os.getenv("QDRANT_COLLECTION", "clinical_trials")
        count = client.count(collection).count
        if count < MIN_POINTS:
            return False, f"collection {collection} has {count} points (< {MIN_POINTS})"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def test_retrieval_no_regression():
    ready, reason = _qdrant_ready()
    if not ready:
        pytest.skip(f"Qdrant corpus unavailable: {reason}")
    if not BASELINE_PATH.exists():
        pytest.skip(f"no baseline at {BASELINE_PATH}")

    baseline = json.loads(BASELINE_PATH.read_text())

    from eval.run_eval import run
    summary = run(k=baseline["k"])

    print(f"\nmean_recall {summary['mean_recall']} (baseline {baseline['mean_recall']}), "
          f"hit_rate {summary['hit_rate']} (baseline {baseline['hit_rate']})")

    assert summary["mean_recall"] >= baseline["mean_recall"] - REGRESSION_MARGIN, (
        f"mean recall@{baseline['k']} regressed: {summary['mean_recall']} vs "
        f"baseline {baseline['mean_recall']}"
    )
    assert summary["hit_rate"] >= baseline["hit_rate"] - REGRESSION_MARGIN, (
        f"hit rate regressed: {summary['hit_rate']} vs baseline {baseline['hit_rate']}"
    )
