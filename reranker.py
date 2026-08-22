"""Second-stage cross-encoder reranking for retrieval candidates.

Pipeline position: hybrid (dense+BM25 RRF) retrieval over-fetches
candidates, this module re-scores each (query, document) pair with a real
cross-encoder, and only the top-k survivors move on to the expensive
gpt-4o Map-Reduce extraction -- better precision at the cut AND a smaller
extraction fan-out.

Provider selection via env:
    RERANKER_PROVIDER=local   (default) fastembed ONNX cross-encoder, no
                              external calls -- fits the self-hosted ethos.
    RERANKER_PROVIDER=none    disable reranking entirely (retrieval returns
                              fusion order, exactly the pre-rerank behavior).
    RERANKER_MODEL            local model override. Default is
                              Xenova/ms-marco-MiniLM-L-6-v2 -- benchmarked
                              at 1.2s/100 pairs on an M-series CPU vs 10.1s
                              for BAAI/bge-reranker-base; on a 1-vCPU
                              Fargate task the bigger model would add
                              ~30-40s per tool call, which an interactive
                              query can't absorb.

Failure policy: reranking is an ORDERING refinement, never a correctness
gate -- any exception degrades to the original fusion order with a logged
warning instead of failing the query.
"""
from __future__ import annotations

import os
from typing import Callable, Sequence

RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "local").lower()
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# How many fused candidates the retrieval layer should over-fetch for the
# reranker to choose from. 100 is the standard retrieve-wide/rerank-narrow
# ratio from the 2024-26 hybrid-retrieval literature.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "100"))

_encoder = None


def enabled() -> bool:
    return RERANKER_PROVIDER not in ("none", "off", "")


def _local_encoder():
    global _encoder
    if _encoder is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _encoder = TextCrossEncoder(RERANKER_MODEL)
    return _encoder


def rerank(
    query: str,
    items: Sequence[dict],
    text_of: Callable[[dict], str],
    top_k: int,
) -> list[dict]:
    """Return `items` re-ordered by cross-encoder relevance to `query`,
    truncated to top_k. Degrades to the input order (truncated) on any
    failure or when disabled."""
    if not enabled() or len(items) <= 1:
        return list(items)[:top_k]
    try:
        docs = [text_of(it) for it in items]
        scores = list(_local_encoder().rerank(query, docs))
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        ranked = [dict(items[i], rerank_score=round(float(scores[i]), 4))
                  for i in order[:top_k]]
        return ranked
    except Exception as exc:  # noqa: BLE001 -- ordering refinement, not a gate
        print(f"[rerank] degraded to fusion order: {exc}")
        return list(items)[:top_k]
