"""BM25 sparse embeddings for Qdrant hybrid search -- shared by query-time
retrieval (research_agent.retrieve_trials), the one-time hybrid migration
(migrate_hybrid.py), and every ingestion path that writes new points.

Model: fastembed's "Qdrant/bm25" -- tokenizes/stems locally (no network
call per embed; the tokenizer config downloads from HF Hub once and is
cached) and emits token-hash indices with BM25 term-frequency values. The
IDF half of BM25 is computed SERVER-SIDE by Qdrant: collections must
declare the sparse vector with modifier=IDF (see migrate_hybrid.py), which
is why document values here only carry the TF component.

Why BM25 at all, next to OpenAI dense embeddings: biopharma queries are
dominated by exact tokens dense vectors blur -- NCT IDs, development codes
("BMS-986278"), gene/target names. Measured on eval/golden.jsonl, dense-only
retrieval left 13/50 queries with ZERO relevant results in top-20.
"""
from __future__ import annotations

from typing import Iterable

from qdrant_client import models as qmodels

SPARSE_VECTOR_NAME = "bm25"
_MODEL_NAME = "Qdrant/bm25"

_model = None


def _bm25():
    global _model
    if _model is None:
        from fastembed import SparseTextEmbedding  # deferred: pulls onnxruntime
        _model = SparseTextEmbedding(_MODEL_NAME)
    return _model


def sparse_vector_params() -> dict:
    """Sparse vector schema for create_collection -- IDF computed by Qdrant."""
    return {SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)}


def embed_docs(texts: list[str]) -> list[qmodels.SparseVector]:
    return [
        qmodels.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in _bm25().embed(texts)
    ]


def embed_query(text: str) -> qmodels.SparseVector:
    e = next(iter(_bm25().query_embed(text)))
    return qmodels.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def trial_sparse_text(payload: dict) -> str:
    """The text BM25 indexes for a clinical-trials point, built purely from
    the point's own payload (so the migration never re-fetches source data).

    Superset of the dense-embedded document (fetch_and_embed_trials.py's
    build_embedding_text): adds NCTId, BriefTitle and sponsor, because exact
    identifier/title/sponsor tokens are precisely what lexical search is FOR
    -- the dense text omitted them for semantic-noise reasons that don't
    apply to BM25.
    """
    conditions = payload.get("conditions") or []
    interventions = payload.get("interventions") or []
    iv_names = ", ".join(
        f"{iv.get('type', '')}: {iv.get('name', '')}"
        for iv in interventions if isinstance(iv, dict)
    )
    parts = [
        payload.get("NCTId") or "",
        payload.get("BriefTitle") or "",
        f"Sponsor: {payload.get('LeadSponsorName')}" if payload.get("LeadSponsorName") else "",
        f"Conditions: {', '.join(c for c in conditions if isinstance(c, str))}",
        f"Interventions: {iv_names}",
        f"Study Type: {payload.get('studyType') or ''}",
        f"Summary: {payload.get('BriefSummary') or ''}",
    ]
    return "\n".join(p for p in parts if p)


def generic_sparse_text(payload: dict, text_keys: Iterable[str]) -> str:
    """Sparse text for non-trial collections: concatenate the named payload
    fields that exist. Used by migrate_hybrid.py for pubmed_literature etc."""
    parts = []
    for key in text_keys:
        v = payload.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
        elif isinstance(v, list):
            parts.append(", ".join(str(x) for x in v))
    return "\n".join(parts)


def pubmed_sparse_text(payload: dict) -> str:
    return generic_sparse_text(payload, ["PMCID", "Title", "Journal", "Text"])


# Which collections carry a bm25 sparse vector, and how to build its text
# from a point's payload. A collection absent here is dense-only ON PURPOSE
# (e.g. openfda_drugsfda until it's deliberately migrated) -- ingestion and
# retrieval both consult this registry so the two can never disagree.
def crl_sparse_text(payload: dict) -> str:
    return generic_sparse_text(
        payload, ["ApplicationNumbers", "CompanyName", "LetterType",
                  "LetterDate", "Text"])


_TEXT_BUILDERS = {
    "clinical_trials": trial_sparse_text,
    "pubmed_literature": pubmed_sparse_text,
    "fda_crls": crl_sparse_text,
}


def sparse_text_builder_for(collection: str):
    """Payload->text builder for a collection, or None if that collection
    is not (yet) part of the hybrid rollout. Tolerates the migration's
    `_hybrid` suffix so `clinical_trials_hybrid` resolves like its alias."""
    base = collection[:-len("_hybrid")] if collection.endswith("_hybrid") else collection
    return _TEXT_BUILDERS.get(base)


def collection_has_bm25(client, collection: str) -> bool:
    """Does this (possibly pre-migration) collection declare the bm25
    sparse vector? Ingestion uses this to decide point shape at upsert time
    so it stays compatible with both schemas during the rollout."""
    try:
        info = client.get_collection(collection)
        return SPARSE_VECTOR_NAME in (info.config.params.sparse_vectors or {})
    except Exception:  # noqa: BLE001 -- unreachable/missing collection: treat as no
        return False
