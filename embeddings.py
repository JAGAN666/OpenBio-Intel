"""Shared OpenAI embedding client -- text-embedding-3-small, 1536-dim.

Replaces the local FastEmbed/Nomic path (nomic-embed-text-v1.5, 768-dim,
CPU-only ONNX inference) previously used by fetch_and_embed_trials.py,
seed_bulk_data.py, research_agent.py, ingest_pipeline.py, and
verify_qdrant.py. Offloading embedding compute to a managed API was
necessary to hydrate the 500,000+ record Data Lake without the local CPU
memory pressure/thrashing already measured seeding a few hundred
ClinicalTrials.gov records on this machine (~2s/doc; see seed_bulk_data.py's
verified smoke-test runs) -- at 500,000+ records that is many hours of
single-machine CPU time, not a batch job that finishes overnight.

RETRY LAYERING -- verified directly against the installed
langchain-openai==1.5.0 package (not assumed from the spec's wording):
OpenAIEmbeddings.embed_documents() already (a) batches internally via
`chunk_size` and (b) passes `max_retries` straight through to the
underlying `openai` Python SDK client, which implements its own
exponential-backoff-with-jitter retry for 429/5xx per HTTP call. That
inner layer is real and does not need reinventing. What it does NOT cover
is an entire batch call exhausting ITS OWN retries and raising outright --
a real risk over hundreds of thousands of records, not a hypothetical --
so embed_documents()/embed_query() below add a second, OUTER tenacity
retry at the call-site: one exhausted batch retries here (with its own
backoff) instead of aborting the whole bulk seed.
"""

from __future__ import annotations

import os
from pathlib import Path

import openai
import tenacity
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http import models as qmodels

# Defensive, like every other module in this codebase that needs .env values
# -- other scripts already load .env before importing this module, but this
# makes embeddings.py safe to import standalone too. override=False so an
# already-exported shell var still wins.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# Fixed for this model (not the `dimensions=` truncation param OpenAI's v3
# embedding models also support) -- kept at native size for max fidelity.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Same metric FastEmbed's Nomic config used -- verified directly before this
# migration: QdrantClient.get_fastembed_vector_params() for
# nomic-embed-text-v1.5 returned Distance.COSINE.
DISTANCE = qmodels.Distance.COSINE

# Sub-batch size for OpenAIEmbeddings' internal chunking. Conservative
# relative to the library's own default (1000): this corpus's documents run
# up to ~726 tokens each (see fetch_and_embed_trials.build_embedding_text's
# docstring), so a smaller chunk keeps each HTTP request's total token count
# well clear of the embeddings endpoint's per-request budget.
_CHUNK_SIZE = 300

# Outer retry: only genuinely transient conditions -- rate limiting,
# connection drops, timeouts, upstream 5xx. Deliberately NOT a bare
# `Exception` catch, which would also retry (and mask) real programming
# errors like a bad argument type.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

_embedder: OpenAIEmbeddings | None = None


def _client() -> OpenAIEmbeddings:
    """Lazy singleton. Fails fast with an actionable message if the key is
    missing, rather than a stack trace mid-batch."""
    global _embedder
    if _embedder is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(
                "[embeddings] OPENAI_API_KEY is not set.\n"
                "             Add it to .env:  OPENAI_API_KEY=sk-..."
            )
        _embedder = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            chunk_size=_CHUNK_SIZE,
            max_retries=5,  # inner, per-HTTP-call retry layer -- see module docstring
        )
    return _embedder


def vector_params() -> qmodels.VectorParams:
    """Qdrant vectors_config for a fresh collection at this model's
    dimension -- the size=1536 the spec calls out explicitly, since
    text-embedding-3-small's 1536 dims are NOT interchangeable with
    nomic-embed-text-v1.5's 768: any collection created under the old model
    MUST be recreated, never upserted into as-is."""
    return qmodels.VectorParams(size=EMBEDDING_DIM, distance=DISTANCE)


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(_RETRYABLE),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    stop=tenacity.stop_after_attempt(6),
    reraise=True,
)
def embed_documents(texts: list[str]) -> list[list[float]]:
    """Batch-embed many texts in ONE call -- callers pass the whole batch's
    documents as a list (never loop one text at a time into this), so the
    langchain-openai chunking above is what actually issues the bulk HTTP
    request(s)."""
    return _client().embed_documents(texts)


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(_RETRYABLE),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    stop=tenacity.stop_after_attempt(6),
    reraise=True,
)
def embed_query(text: str) -> list[float]:
    """Single query-side embedding. MUST be the same model as
    embed_documents(), or the query vector is not comparable to what is
    stored (and a 768-vs-1536 mismatch hard-fails at the Qdrant call)."""
    return _client().embed_query(text)
