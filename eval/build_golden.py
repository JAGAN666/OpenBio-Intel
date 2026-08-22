"""Generate eval/golden.jsonl -- the retrieval-eval golden set.

Ground-truth design (and why it is NOT circular): each golden query's
expected NCT IDs come from EXACT payload matching in Qdrant (server-side
filters on the literal intervention name / condition strings), never from
vector similarity. The dense/hybrid retriever being evaluated plays no part
in defining what "relevant" means -- it is then measured on whether its
top-k finds those lexically-certain matches from natural-language phrasings
of the same intent.

Query selection targets SMALL ground-truth sets (2..15 trials globally) so
recall@20 is a meaningful number: a (drug, condition) pair with 900 matching
trials can't be "recalled" into 20 slots, but a rare drug code with 6 trials
can -- and rare exact tokens (development codes like "BMS-986278") are
precisely where dense embeddings are weakest and hybrid BM25 fusion should
show up in the numbers.

One-time (re)generation script, run against the FULL local corpus:
    uv run python eval/build_golden.py
The output golden.jsonl is committed; regenerating is only needed if the
corpus changes so much that expected trials disappear.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "clinical_trials")

SAMPLE_POINTS = 40_000        # payload sample used to discover candidates
MIN_EXPECTED, MAX_EXPECTED = 2, 15   # global ground-truth size band
TARGET_QUERIES = 50
SEED = 20260822               # deterministic regeneration

# Natural-language templates -- the point is that the QUERY never contains
# the exact ground-truth filter syntax, only a human phrasing of it.
PAIR_TEMPLATES = [
    "Which clinical trials are studying {drug} for {cond}?",
    "trials evaluating {drug} in patients with {cond}",
    "What studies test {drug} as a treatment for {cond}?",
]
DRUG_TEMPLATES = [
    "Which trials use {drug}?",
    "clinical studies involving the drug {drug}",
    "What is {drug} being tested for?",
]


def _names(payload: dict) -> list[str]:
    out = []
    for iv in payload.get("interventions") or []:
        if isinstance(iv, dict) and iv.get("name"):
            out.append(iv["name"])
    return out


def _expected_ncts(client: QdrantClient, drug: str, cond: str | None) -> list[str]:
    """Global exact-match ground truth via server-side payload filter."""
    must = [qmodels.FieldCondition(key="interventions[].name",
                                   match=qmodels.MatchValue(value=drug))]
    if cond is not None:
        must.append(qmodels.FieldCondition(key="conditions",
                                           match=qmodels.MatchValue(value=cond)))
    ncts: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qmodels.Filter(must=must),
            limit=64, offset=offset, with_payload=["NCTId"], with_vectors=False,
        )
        for p in points:
            if p.payload and p.payload.get("NCTId"):
                ncts.add(p.payload["NCTId"])
        if offset is None or len(ncts) > MAX_EXPECTED + 10:
            break
    return sorted(ncts)


def main() -> None:
    rng = random.Random(SEED)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    print(f"sampling {SAMPLE_POINTS} payloads from {COLLECTION} ...", file=sys.stderr)
    drug_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    seen = 0
    offset = None
    while seen < SAMPLE_POINTS:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=1000, offset=offset,
            with_payload=["interventions", "conditions"], with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            names = _names(pl)
            conds = [c for c in (pl.get("conditions") or []) if isinstance(c, str)]
            for n in names:
                # Skip generic non-informative intervention labels.
                if len(n) < 4 or n.lower() in {"placebo", "saline", "questionnaire",
                                               "observation", "standard of care"}:
                    continue
                drug_counts[n] += 1
                for c in conds[:3]:
                    pair_counts[(n, c)] += 1
        seen += len(points)
        if offset is None:
            break
    print(f"sampled {seen} points, {len(drug_counts)} distinct interventions",
          file=sys.stderr)

    # Candidates: appear a handful of times in the sample (rare-ish globally).
    drug_candidates = [d for d, c in drug_counts.items() if 2 <= c <= 8]
    pair_candidates = [pc for pc, c in pair_counts.items() if 2 <= c <= 6]
    rng.shuffle(drug_candidates)
    rng.shuffle(pair_candidates)

    golden: list[dict] = []
    used_drugs: set[str] = set()

    def _try_add(drug: str, cond: str | None) -> None:
        if len(golden) >= TARGET_QUERIES or drug in used_drugs:
            return
        expected = _expected_ncts(client, drug, cond)
        if not (MIN_EXPECTED <= len(expected) <= MAX_EXPECTED):
            return
        if cond is None:
            q = rng.choice(DRUG_TEMPLATES).format(drug=drug)
        else:
            q = rng.choice(PAIR_TEMPLATES).format(drug=drug, cond=cond)
        golden.append({
            "query": q,
            "expected_nct_ids": expected,
            "ground_truth": {"drug": drug, "condition": cond},
        })
        used_drugs.add(drug)
        print(f"  [{len(golden):2d}] {q!r} -> {len(expected)} expected", file=sys.stderr)

    # Aim for roughly half pair queries, half drug-only queries.
    for drug, cond in pair_candidates:
        if len(golden) >= TARGET_QUERIES // 2:
            break
        _try_add(drug, cond)
    for drug in drug_candidates:
        if len(golden) >= TARGET_QUERIES:
            break
        _try_add(drug, None)

    out = Path(__file__).parent / "golden.jsonl"
    with out.open("w") as f:
        for row in golden:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(golden)} golden queries -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
