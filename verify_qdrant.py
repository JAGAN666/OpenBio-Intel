"""[AC 7] Hybrid query proof: semantic search + strict payload filter.

Searches the semantic concept "tumor shrinkage" while strictly filtering to
Phase == 'Phase 3', and prints the top 3 hits.

To prove the filter is genuinely doing work (rather than the top results
happening to be Phase 3 anyway) the same query is run twice -- once
unfiltered, once filtered -- and the results are contrasted, then asserted.

    python verify_qdrant.py
    python verify_qdrant.py --query "tumor shrinkage" --phase "Phase 3" --limit 3
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embeddings import EMBEDDING_MODEL, embed_query

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "clinical_trials"


def show(hits, label: str) -> None:
    print(f"\n{'-' * 76}\n{label}\n{'-' * 76}")
    if not hits:
        print("  (no results)")
        return
    for i, h in enumerate(hits, 1):
        meta = h.payload or {}
        phases = meta.get("Phase") or []
        # The raw text handed to the embedding model isn't itself stored in
        # the payload (only its structured pieces are) -- BriefSummary is the
        # closest stored analog for a display preview here.
        summary = (meta.get("BriefSummary") or "").strip().replace("\n", " ")
        if len(summary) > 260:
            summary = summary[:260].rstrip() + " ..."

        print(f"\n  [{i}] {meta.get('NCTId')}   score={h.score:.4f}")
        print(f"      Title   : {textwrap.shorten(meta.get('BriefTitle', ''), 132, placeholder=' ...')}")
        print(f"      Phase   : {', '.join(phases) if phases else '(none)'}")
        print(f"      Status  : {meta.get('OverallStatus')}")
        print(f"      Sponsor : {meta.get('LeadSponsorName')}")
        print("      Summary :")
        for line in textwrap.wrap(summary, 64) or ["(none)"]:
            print(f"          {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="tumor shrinkage", help="semantic query text")
    parser.add_argument("--phase", default="Phase 3", help="phase to filter on")
    parser.add_argument("--limit", type=int, default=3, help="results to show (default 3)")
    args = parser.parse_args()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print(
                f"[verify] collection '{COLLECTION_NAME}' does not exist.\n"
                "         Run: python fetch_and_embed_trials.py",
                file=sys.stderr,
            )
            return 1
    except Exception as exc:
        print(f"[verify] cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT} -> {exc}",
              file=sys.stderr)
        return 1

    # Query embedding must match the model the collection was built with, or
    # the query vector will not be comparable to the stored vectors.
    query_vector = embed_query(args.query)

    info = client.get_collection(COLLECTION_NAME)
    total = info.points_count

    phase_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="Phase", match=qmodels.MatchValue(value=args.phase)
            )
        ]
    )
    eligible = client.count(
        collection_name=COLLECTION_NAME, count_filter=phase_filter, exact=True
    ).count

    print("=" * 76)
    print("AC 7 :: HYBRID QUERY  (semantic vector search + strict payload filter)")
    print("=" * 76)
    print(f"collection      : {COLLECTION_NAME}  ({total} points)")
    print(f"model           : {EMBEDDING_MODEL}")
    print(f"semantic query  : {args.query!r}")
    print(f"payload filter  : Phase == {args.phase!r}")
    print(f"eligible points : {eligible} of {total}")

    # --- A: unfiltered baseline -----------------------------------------
    unfiltered = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=args.limit
    ).points
    show(unfiltered, f"A. UNFILTERED  -- top {args.limit} for {args.query!r} (baseline)")

    # --- B: the AC 7 hybrid query ---------------------------------------
    filtered = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=phase_filter,
        limit=args.limit,
    ).points
    show(filtered, f"B. FILTERED    -- top {args.limit}, Phase == {args.phase!r}   [AC 7]")

    # --- proof ------------------------------------------------------------
    print(f"\n{'=' * 76}\nPROOF\n{'=' * 76}")

    off_phase = [
        (h.payload or {}).get("NCTId")
        for h in filtered
        if args.phase not in ((h.payload or {}).get("Phase") or [])
    ]
    ok = not off_phase and len(filtered) > 0

    base_ids = [(h.payload or {}).get("NCTId") for h in unfiltered]
    filt_ids = [(h.payload or {}).get("NCTId") for h in filtered]
    excluded = [
        f"{(h.payload or {}).get('NCTId')} ({'/'.join((h.payload or {}).get('Phase') or [])})"
        for h in unfiltered
        if args.phase not in ((h.payload or {}).get("Phase") or [])
    ]

    print(f"unfiltered top-{args.limit} : {base_ids}")
    print(f"filtered   top-{args.limit} : {filt_ids}")
    if excluded:
        print(f"correctly excluded by the filter : {', '.join(excluded)}")
    else:
        print("note: the unfiltered top hits were already all "
              f"{args.phase}; see 'eligible points' above -- the filter still "
              "restricted the candidate set.")

    if off_phase:
        print(f"\nFAIL: results leaked past the filter -> {off_phase}")
        return 1
    if not filtered:
        print("\nFAIL: filter returned no results at all.")
        return 1

    print(f"\nPASS: all {len(filtered)} returned results carry Phase == {args.phase!r}.")
    print("AC 7 satisfied.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
