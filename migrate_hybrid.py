"""One-time migration: rebuild a dense-only Qdrant collection as a hybrid
(dense + BM25 sparse) collection, then swap it in behind an alias.

Why a rebuild instead of an in-place update: verified live against Qdrant
1.19.0 -- both qdrant-client's update_collection and the raw REST PATCH
refuse to ADD a new sparse vector name to an existing collection ("Wrong
input: Not existing vector name error: bm25"). Sparse vectors must be
declared at creation, so the only path is: create `<name>_hybrid` with the
same unnamed dense vector config PLUS the bm25 sparse config, stream every
point across (dense vectors COPIED via scroll -- no re-embedding, no OpenAI
cost), computing BM25 locally from each point's own payload.

The swap step then deletes the original collection and creates an ALIAS
`<name>` -> `<name>_hybrid`, so every existing consumer (agent tools,
health check, eval) keeps using the old name untouched -- verified in the
same spike that aliases work for query_points/scroll/count.

Usage:
    uv run python migrate_hybrid.py --collection clinical_trials          # build
    uv run python migrate_hybrid.py --collection clinical_trials --swap   # cut over
    uv run python migrate_hybrid.py --collection pubmed_literature
    uv run python migrate_hybrid.py --collection pubmed_literature --swap
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qmodels

from sparse_embeddings import (
    SPARSE_VECTOR_NAME,
    embed_docs,
    sparse_text_builder_for,
    sparse_vector_params,
)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
BATCH = 256

# Per-collection BM25 text builders live in sparse_embeddings._TEXT_BUILDERS
# (the registry ingestion also consults); this module only needs the lookup.
MIGRATABLE = ("clinical_trials", "pubmed_literature")


def build(client: QdrantClient, source: str) -> str:
    target = f"{source}_hybrid"
    info = client.get_collection(source)
    dense_cfg = info.config.params.vectors  # unnamed VectorParams
    total = client.count(source).count
    print(f"[build] {source}: {total} points, dense={dense_cfg.size}d "
          f"-> {target}")

    if client.collection_exists(target):
        done = client.count(target).count
        if done >= total:
            print(f"[build] {target} already complete ({done} points) -- skipping")
            return target
        print(f"[build] {target} exists with {done}/{total} points -- rebuilding "
              f"from scratch (scroll order gives no safe resume cursor)")
        client.delete_collection(target)

    client.create_collection(
        target,
        vectors_config=dense_cfg,
        sparse_vectors_config=sparse_vector_params(),
        on_disk_payload=True,
    )

    text_builder = sparse_text_builder_for(source)
    assert text_builder, f"no sparse text builder registered for {source}"
    moved = 0
    started = time.perf_counter()
    offset = None
    while True:
        points, offset = client.scroll(
            source, limit=BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not points:
            break
        texts = [text_builder(p.payload or {}) for p in points]
        sparse = embed_docs(texts)
        new_points = []
        for p, sv in zip(points, sparse):
            dense = p.vector if not isinstance(p.vector, dict) else p.vector.get("")
            new_points.append(qmodels.PointStruct(
                id=p.id,
                vector={"": dense, SPARSE_VECTOR_NAME: sv},
                payload=p.payload,
            ))
        client.upsert(target, points=new_points, wait=False)
        moved += len(points)
        if moved % (BATCH * 40) < BATCH:
            rate = moved / (time.perf_counter() - started)
            eta = (total - moved) / rate if rate else 0
            print(f"[build]   {moved}/{total} ({rate:.0f} pts/s, eta {eta/60:.1f} min)")
        if offset is None:
            break

    # wait=False above for throughput -- give the last batches a moment,
    # then verify the count matches before declaring success.
    time.sleep(3)
    final = client.count(target).count
    print(f"[build] done: {final}/{total} points in {(time.perf_counter()-started)/60:.1f} min")
    if final != total:
        sys.exit(f"[build] FAILED: target has {final} points, expected {total}")
    return target


def swap(client: QdrantClient, source: str) -> None:
    target = f"{source}_hybrid"
    src_count = client.count(source).count
    tgt_count = client.count(target).count
    if tgt_count < src_count:
        sys.exit(f"[swap] REFUSING: {target} has {tgt_count} < {source}'s {src_count}")
    print(f"[swap] deleting {source} ({src_count} pts) and aliasing "
          f"{source} -> {target} ({tgt_count} pts)")
    client.delete_collection(source)
    client.update_collection_aliases(change_aliases_operations=[
        qmodels.CreateAliasOperation(create_alias=qmodels.CreateAlias(
            collection_name=target, alias_name=source))
    ])
    # verify the alias answers
    print(f"[swap] alias check: count({source}) = {client.count(source).count}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True, choices=sorted(MIGRATABLE))
    ap.add_argument("--swap", action="store_true",
                    help="cut over: delete original, alias its name to the hybrid build")
    args = ap.parse_args()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    if args.swap:
        swap(client, args.collection)
    else:
        build(client, args.collection)


if __name__ == "__main__":
    main()
