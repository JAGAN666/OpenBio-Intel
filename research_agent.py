"""Multi-agent ReACT RAG over the local Qdrant clinical-trials collection.

"Deterministic Orchestrator" architecture -- defensive guardrails wrap a
Map-Reduce extraction core so the graph fails closed instead of hallucinating,
and scales to hundreds of retrieved documents without ever putting more than
one trial's raw text in front of an LLM at a time:

    START ─▶ [IntentClassifier] ──in domain?──▶ [OutOfDomain] ─▶ END
                    │ yes                        deterministic refusal,
                    ▼                             no LLM call, no Qdrant hit
               [Agent] ──tool_calls?──▶ [Tool Node]
                  │  no                      │
                  │                          ├─ has_results? no ─▶ [NoResultsFallback] ─▶ END
                  │                          │                     deterministic, no LLM call
                  │                          │ yes
                  │                          ▼
                  │             continue_to_extraction (Mapper)
                  │              Send("extract_trial", {"single_trial": t})
                  │                  once per distinct retrieved trial
                  │                          │
                  │                          ▼
                  │              [extract_trial] × N   (Map -- parallel workers,
                  │               one LLM call per trial, .with_structured_output(TrialRow);
                  │               writes merge via the operator.add reducer on extracted_rows)
                  │                          │
                  ▼                          ▼  (all N workers join)
          [synthesize_table] ◀───────────────┘  (Reduce -- narrative written ONLY from the
              │      ▲  loop back                extracted_rows array, never the raw
       valid  │      │  retry (max 2)             Qdrant text)
      output  │      └── on Pydantic ValidationError
              ▼
             END

  IntentClassifier   cheap/fast model gate: is this query in-domain at all?
                      Runs BEFORE Qdrant is ever touched.
  Agent               reasons about the question and chooses search arguments
  Tool Node           runs the Qdrant hybrid search; flags whether anything
                      it found is actually grounded in the query
  NoResultsFallback   deterministic "nothing found" message -- bypasses
                      extraction/synthesis entirely so the LLM never gets a
                      chance to narrate around an empty result set
  continue_to_extraction (Mapper)
                      routes each distinct retrieved trial to its own
                      extract_trial worker via the `Send` API -- the graph
                      dynamically fans out to N parallel branches at runtime,
                      N being however many trials Qdrant returned
  extract_trial (Worker, Map stage)
                      one independent LLM call per trial, constrained to the
                      TrialRow schema; never sees any other trial's text, so
                      context usage per worker is O(1), not O(all retrieved
                      trials) the way one monolithic synthesis prompt was
  synthesize_table (Reducer)
                      runs once every worker has written into extracted_rows
                      (merged via the operator.add state reducer -- parallel
                      writes append instead of racing/overwriting each
                      other), then writes narrative_summary from that
                      structured array only; wrapped in the same Pydantic
                      validate-or-retry loop (max 2 retries) as before

The previous design ran an iterative ReACT loop (Tool Node could hand back to
Agent for a refined second search) and asked ONE LLM call to both extract
every trial's fields AND write the narrative in a single monolithic prompt --
that is exactly what breaks down once retrieval returns hundreds of trials
instead of a handful. This refactor keeps the guardrails (IntentClassifier,
NoResultsFallback, the Pydantic retry loop) but replaces the "one giant
synthesis prompt" step with the Map-Reduce fan-out above; Agent now runs
exactly once per request instead of looping, since parallel extraction is
what absorbs the work a second refinement search used to do.

    python research_agent.py
    python research_agent.py --question "..." --model claude-opus-5
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embeddings import EMBEDDING_MODEL, embed_query

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# --- Qdrant / embedding (must match fetch_and_embed_trials.py) --------------
# Overridable via env: "localhost" is correct for bare-metal/host dev, but
# inside a Docker Compose network "localhost" from the backend container
# refers to the backend container itself, not the qdrant service -- Compose
# sets QDRANT_HOST=qdrant so the container reaches Qdrant by service name.
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "clinical_trials"
# Federated second source (see ingest_pipeline.py): unstructured PDF
# literature -- conference posters, FDA filings -- parsed via vision-based
# extraction. Kept in its own collection rather than mixed into
# COLLECTION_NAME because it has none of that collection's structured
# fields (NCTId, Phase, LeadSponsorName); search_pdf_literature below is the
# dedicated tool over it.
COLLECTION_NAME_PDF = "clinical_trials_pdf_extracts"
# Third federated source (see seed_bulk_data.py): openFDA's drug/drugsfda
# bulk dataset -- FDA drug APPROVAL/APPLICATION records (application number,
# sponsor, brand/generic names, dosage forms, RxNorm CUIs). Deliberately NOT
# drug/event (adverse-event reports) -- that partition was never seeded into
# this collection, so this tool cannot answer adverse-event/safety-signal
# questions; see search_fda_records' own docstring for the exact scope.
COLLECTION_NAME_FDA = "openfda_drugsfda"
# EMBEDDING_MODEL is imported from embeddings.py (text-embedding-3-small,
# 1536-dim, via OpenAI) -- must match the model the collections were indexed
# with, or the query vector is not comparable to the stored vectors (and a
# dimension mismatch hard-fails at the Qdrant call). All three collections
# above are indexed with this SAME model, so one Qdrant client (see
# _client()) can query any of them just by switching collection_name and
# re-embedding the query text via embed_query().

# --- Neo4j biomedical knowledge graph (see build_kg.py) ----------------------
# Overridable via env: same QDRANT_HOST-style container override pattern --
# "localhost" is correct for bare-metal/host dev; docker-compose.yml sets
# NEO4J_URI=bolt://neo4j:7687 for the containerized backend.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# --- LLM --------------------------------------------------------------------
# claude-opus-5 is the current default Claude model. Note two API facts that
# shape the request built below:
#   * temperature / top_p / top_k are REJECTED with a 400 on this model.
#   * thinking is ON by default, and max_tokens caps thinking + text together,
#     so max_tokens needs real headroom or answers truncate mid-sentence.
DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 8000
MAX_TOOL_ROUNDS = 3  # ReACT loop cap -- guards against a runaway agent
MAX_SYNTHESIS_RETRIES = 2  # Pydantic self-correction budget

# The intent gate deliberately uses a separate, cheap/fast model rather than
# the generator -- it is a single-purpose yes/no classifier that runs on
# every request BEFORE any Qdrant call, so its cost and latency matter more
# than its depth. Same provider as the generator (no second API key to manage).
INTENT_MODEL = "claude-haiku-4-5"

PHASE_LABELS = {
    "PHASE1": "Phase 1", "PHASE2": "Phase 2", "PHASE3": "Phase 3",
    "PHASE4": "Phase 4", "EARLY_PHASE1": "Early Phase 1", "NA": "Not Applicable",
}


# =============================================================================
# STRUCTURED OUTPUT SCHEMA  ("Smart Table" contract for the frontend)
# =============================================================================
# These models ARE the API contract between the agent and the Next.js grid.
# LangChain converts them to a tool schema, Anthropic constrains generation to
# it, and the response is validated before it reaches our code -- so a
# malformed row raises a ValidationError here rather than rendering an empty
# cell in production. Field descriptions are sent to the model, so they are
# prompt surface, not just documentation.
class TrialRow(BaseModel):
    """One row of the comparative trials grid."""

    nct_id: str = Field(
        description="The trial's NCT identifier, e.g. NCT06712888. Must be "
                    "copied exactly from a retrieved record — never invented."
    )
    sponsor: str = Field(
        description="Lead sponsor organisation running the trial."
    )
    # Deliberately `str`, NOT an Enum: trials legitimately register as
    # "Phase 2/Phase 3", which no single-value enum can represent.
    phase: str = Field(
        description="Trial phase as a display string, e.g. 'Phase 3' or "
                    "'Phase 2/Phase 3' when the trial spans both."
    )
    interventions: list[str] = Field(
        description="The specific intervention names being tested, taken from "
                    "the record's structured `interventions` field. Prefer "
                    "therapeutic agents (drugs/biologics) over assessment "
                    "procedures such as CT or biospecimen collection."
    )
    mechanism_or_findings: str = Field(
        description="One to two sentences on the mechanism or key finding, "
                    "taken ONLY from the retrieved trial text. Never supply a "
                    "mechanism from prior knowledge — an unsupported claim is "
                    "worse than a gap. If the mechanism is absent, describe "
                    "whatever trial design or clinical findings are available "
                    "in the text, and ensure the boolean flag is set to False."
    )
    # Declared AFTER the prose deliberately: Pydantic field order is the JSON
    # schema property order, which is generation order in the tool call. The
    # model therefore writes what it found first, then flags what it just
    # wrote -- a judgment over committed text rather than a prediction it has
    # to live up to.
    mechanism_described: bool = Field(
        description="True ONLY if the retrieved text explicitly names the "
                    "mechanism of action or biological target of the primary "
                    "intervention. False if the mechanism/target is not "
                    "stated, even if other trial design details (like "
                    "biomarkers) are present."
    )


class SmartTableResponse(BaseModel):
    """Final agent output: prose answer + machine-renderable grid."""

    narrative_summary: str = Field(
        description="The high-level analytical answer to the user's question. "
                    "Every factual claim must cite its NCT id in brackets, "
                    "e.g. 'targets PD-1 [NCT01234567]'."
    )
    table_data: list[TrialRow] = Field(
        description="One TrialRow for every distinct trial retrieved from the "
                    "database. This array backs the frontend data grid."
    )


class NarrativeSummary(BaseModel):
    """Reducer-stage output. Map-Reduce version of the old Synthesis contract:
    table_data is already fixed by the time this runs -- deterministically
    carried through from the Map stage's extracted_rows -- so the Reducer's
    LLM call has one job left: write the prose, from the structured array
    alone."""

    narrative_summary: str = Field(
        description="The high-level analytical answer to the user's question, "
                    "written using ONLY the provided array of already-extracted "
                    "trial rows. Every factual claim must cite its NCT id in "
                    "brackets, e.g. 'targets PD-1 [NCT01234567]'."
    )


class IntentClassification(BaseModel):
    """Input-guardrail verdict from the IntentClassifier node."""

    is_in_domain: bool = Field(
        description="True if the query is about clinical trials, oncology, "
                    "pharmaceutical drugs or biologics, trial sponsors, trial "
                    "phases, or drug mechanisms. False for anything else "
                    "(recipes, general chit-chat, coding help, unrelated "
                    "topics) -- even if it superficially uses a medical word."
    )
    reason: str = Field(
        description="One short sentence explaining the verdict. Logged for "
                    "debugging; not shown to the end user."
    )


# --- deterministic guardrail messages ---------------------------------------
# These are returned verbatim, with NO LLM call -- fixed strings, not prompts.
# A refusal or a "nothing found" message is exactly the kind of output that
# should never vary between two identical requests.
OUT_OF_DOMAIN_MESSAGE = (
    "I am a clinical intelligence agent and can only answer questions "
    "related to biopharma trials."
)
NO_RESULTS_MESSAGE = (
    "I could not find any clinical trials in the database matching your "
    "specific criteria."
)


def _normalise_phase(raw: str | None) -> str | None:
    """Accept 'Phase 3', 'phase3', 'PHASE3', '3' -- emit the stored form.

    The collection stores the human form ("Phase 3"); an un-normalised filter
    silently matches nothing and returns an empty result set with no error.
    """
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    key = s.upper().replace(" ", "").replace("-", "").replace("_", "")
    if key in PHASE_LABELS:
        return PHASE_LABELS[key]
    if key.isdigit():
        return PHASE_LABELS.get(f"PHASE{key}", s)
    if key.startswith("PHASE"):
        return PHASE_LABELS.get(key, s)
    return s


# =============================================================================
# GROUNDING CHECK -- lexical overlap, deliberately NOT a cosine-score cutoff
# =============================================================================
# Empirically verified before writing this (see project notes): cosine
# similarity in this embedding space CANNOT reliably separate a real drug
# query from a fabricated one. A fictional drug embedded in authentic
# trial-language phrasing can score HIGHER than a real, short drug-name-only
# query -- e.g. "trials of XYZ-Fake-Drug-999 in oncology" scored 0.636, while
# "brentuximab vedotin" (a real drug actually in the corpus) scored only
# 0.555. Score magnitude tracks query phrasing style, not factual grounding.
# A threshold strict enough to reject the fictional case would also reject
# the real one -- i.e. it would break the AC-4 happy path to "fix" AC-3.
#
# The reliable signal instead: does the retrieved trial's OWN structured,
# curated data (title / conditions / intervention names -- exact strings from
# ClinicalTrials.gov, not a lossy vector) share any distinctive vocabulary
# with the query? kNN search always returns the k *nearest* points, even when
# nothing is actually relevant -- it has no notion of "nothing matches."
# Lexical overlap against ground-truth fields does.
# This list is bounded and domain-specific -- standard ClinicalTrials.gov
# protocol/regulatory phrasing -- NOT an attempt at general-purpose stopwords.
# It exists because a live run caught a real leak: the query "XYZ-Fake-Drug-999
# investigational compound efficacy and safety" scored has_results=True purely
# because a real trial's title happened to be "A Study to Evaluate the Safety
# and Tolerability of...". "safety"/"efficacy"/"tolerability" are exactly the
# kind of words that appear in nearly every trial's title regardless of what
# drug is being studied, so they must never count as grounding evidence.
_STOPWORDS = {
    "trial", "trials", "clinical", "phase", "mechanism", "mechanisms",
    "oncology", "cancer", "drug", "drugs", "patient", "patients", "study",
    "studies", "treatment", "therapy", "compare", "compared", "comparing",
    "involving", "action", "target", "targets", "best", "recipe", "table",
    "sponsor", "sponsors", "showing", "names", "database", "questions",
    "related", "answer", "about", "which",
    # regulatory / protocol boilerplate
    "safety", "efficacy", "tolerability", "investigational", "compound",
    "agent", "agents", "evaluate", "evaluating", "evaluation", "assess",
    "assessing", "assessment", "determine", "determining", "randomized",
    "randomised", "multicenter", "multicentre", "international",
    "open-label", "blinded", "double-blind", "participants", "subjects",
    "adults", "dosing", "dose", "doses", "group", "groups", "versus",
    "standard", "care", "advanced", "metastatic", "locally", "newly",
    "diagnosed", "recurrent", "refractory", "relapsed", "malignant",
    "malignancy", "pharmacokinetics", "pharmacodynamics", "administration",
    "administered", "receiving", "population", "primary", "secondary",
    "endpoint", "endpoints", "outcome", "outcomes", "superiority",
    "single-arm",
}

# Fraction of the corpus a token can appear in before it stops counting as
# "distinctive" -- defense-in-depth ON TOP OF the stopword list above, not a
# replacement for it. Measured directly against this corpus: at only 50
# trials, common regulatory words like "safety" (6%) and "study" (26%) don't
# YET have enough repetition to statistically separate from real, notable
# drug names like pembrolizumab (16%) -- so a threshold has to sit safely
# above that to avoid excluding legitimately common real drugs today. This
# becomes the primary defense once the corpus is large enough for boilerplate
# title templates ("A Study to Evaluate the Safety and Efficacy of...") to
# dominate their true frequency; the static list is what does the real work
# at today's scale.
_MAX_CORPUS_FREQUENCY = 0.30

_corpus_doc_freq: dict[str, int] | None = None
_corpus_size: int = 0


def _corpus_frequencies() -> tuple[dict[str, int], int]:
    """Lazily computed, cached for the process lifetime -- the collection
    doesn't change mid-run, and re-scrolling it on every tool call would be
    wasteful."""
    global _corpus_doc_freq, _corpus_size
    if _corpus_doc_freq is None:
        df: dict[str, int] = {}
        pts, _ = _client().scroll(COLLECTION_NAME, limit=10_000, with_payload=True)
        for p in pts:
            m = p.payload or {}
            haystack = " ".join([
                m.get("BriefTitle") or "",
                " ".join(m.get("conditions") or []),
                " ".join(m.get("interventionNames") or []),
            ])
            for tok in set(re.findall(r"[a-z0-9-]+", haystack.lower())):
                if len(tok) >= 5:
                    df[tok] = df.get(tok, 0) + 1
        _corpus_doc_freq, _corpus_size = df, len(pts)
    return _corpus_doc_freq, _corpus_size


def _content_tokens(text: str, min_len: int = 5) -> set[str]:
    df, n = _corpus_frequencies()
    too_common = {t for t, c in df.items() if n and c / n >= _MAX_CORPUS_FREQUENCY}
    return {w for w in re.findall(r"[a-z0-9-]+", text.lower())
            if len(w) >= min_len and w not in _STOPWORDS and w not in too_common}


def _is_grounded(query: str, trials: list[dict]) -> bool:
    """True if any retrieved trial's own fields share vocabulary with the query."""
    qtok = _content_tokens(query)
    if not qtok:
        # No distinctive tokens to check (a very generic query) -- don't
        # block on a signal we can't compute here; the citation contract in
        # Synthesis is the backstop for that case.
        return True
    for t in trials:
        haystack = " ".join([
            t.get("BriefTitle") or "",
            " ".join(t.get("conditions") or []),
            " ".join(iv.get("name", "") for iv in (t.get("interventions") or [])),
        ]).lower()
        if any(tok in haystack for tok in qtok):
            return True
    return False


def _is_grounded_literature(query: str, chunks: list[dict]) -> bool:
    """Same anti-hallucination principle as _is_grounded (kNN always returns
    the k nearest points regardless of true relevance), applied to the PDF
    literature collection instead of the trials registry.

    Reuses _content_tokens for the QUERY side only -- its stopword list and
    document-frequency filtering are computed against COLLECTION_NAME's
    vocabulary, which is a reasonable proxy for stripping trivial life-
    sciences boilerplate out of a query regardless of which collection that
    query is then checked against. The HAYSTACK side (each chunk's raw Text)
    is checked with plain token containment rather than curated structured
    fields, because COLLECTION_NAME_PDF has no equivalent to
    BriefTitle/conditions/interventions -- and is far too small today for
    its own document-frequency stats to be meaningful. Worth adding a
    PDF-corpus-specific _MAX_CORPUS_FREQUENCY-style filter once that
    collection holds more than a handful of documents.
    """
    qtok = _content_tokens(query)
    if not qtok:
        return True
    for c in chunks:
        haystack = (c.get("Text") or "").lower()
        if any(tok in haystack for tok in qtok):
            return True
    return False


def _is_grounded_fda(query: str, records: list[dict]) -> bool:
    """Same anti-hallucination principle as _is_grounded, applied to the
    openFDA drugsfda collection: does anything actually retrieved share real
    vocabulary with the query, since kNN always returns its k nearest points
    regardless of true relevance.

    Haystack is BrandNames + ActiveIngredients + SponsorName -- the FDA
    record's own curated identity fields, same role BriefTitle/conditions/
    interventions play for _is_grounded.
    """
    qtok = _content_tokens(query)
    if not qtok:
        return True
    for r in records:
        haystack = " ".join([
            " ".join(r.get("BrandNames") or []),
            " ".join(r.get("ActiveIngredients") or []),
            r.get("SponsorName") or "",
        ]).lower()
        if any(tok in haystack for tok in qtok):
            return True
    return False


# =============================================================================
# TOOLS -- federated Qdrant retrieval over three collections
# =============================================================================
_qdrant: QdrantClient | None = None


def _client() -> QdrantClient:
    """Lazy singleton -- a plain client now; embedding happens explicitly via
    embeddings.embed_query() before each search, not via a bound FastEmbed
    model."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant


@tool
def search_clinical_trials(query: str, phase_filter: Optional[str] = None) -> str:
    """Search the internal clinical-trials vector database by meaning.

    Use this whenever a question depends on what trials are in our database --
    do not answer from prior knowledge. Returns trial records as JSON,
    including structured pharmacology for each trial: `interventions` (a list
    of {type, name} -- the actual drug/procedure names), `conditions` (the
    indications studied) and `studyType`. Prefer these structured fields over
    inferring drugs or indications from the narrative BriefSummary.

    Args:
        query: A semantic description of what to find, e.g. "mechanism of
            action for immune checkpoint inhibition". Natural language, not
            keywords -- results are ranked by embedding similarity.
        phase_filter: Optional strict filter on trial phase. Accepts
            "Phase 2" or "Phase 3". When set, ONLY trials of that phase are
            returned (a hard payload filter, not a ranking hint).
    """
    phase = _normalise_phase(phase_filter)

    query_filter = None
    if phase:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="Phase", match=qmodels.MatchValue(value=phase)
            )]
        )

    try:
        query_vector = embed_query(query)
        hits = _client().query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,   # payload filter + vector search = hybrid
            limit=6,
        ).points
    except Exception as exc:  # surfaced to the agent as an observation
        return json.dumps({"error": f"Qdrant search failed: {exc}", "has_results": False})

    results = []
    for h in hits:
        meta = h.payload or {}
        results.append({
            "NCTId": meta.get("NCTId"),
            "BriefTitle": meta.get("BriefTitle"),
            "Phase": meta.get("Phase"),
            "OverallStatus": meta.get("OverallStatus"),
            "LeadSponsorName": meta.get("LeadSponsorName"),
            # --- structured pharmacology (payload enrichment) -------------
            "studyType": meta.get("studyType"),
            "conditions": meta.get("conditions"),
            "interventions": meta.get("interventions"),
            "score": round(float(h.score), 4),
            "BriefSummary": (meta.get("BriefSummary") or "")[:1200],
        })

    # has_results is NOT `len(results) > 0` -- Qdrant's kNN search always
    # returns up to `limit` nearest points, even for a query about something
    # that doesn't exist in the corpus. It is the lexical-grounding check
    # above: does anything actually retrieved share real vocabulary with what
    # was asked. See the _is_grounded docstring for the empirical case against
    # using a cosine-score cutoff instead.
    grounded = _is_grounded(query, results)

    return json.dumps(
        {"query": query, "phase_filter": phase, "returned": len(results),
         "has_results": grounded, "trials": results},
        indent=2,
    )


@tool
def search_pdf_literature(query: str) -> str:
    """Search unstructured PDF literature -- conference posters, FDA filings --
    parsed via vision-based table extraction into Markdown.

    Use this for reported RESULTS the structured trial registry does not
    carry: efficacy metrics (ORR, PFS, OS), safety / adverse-event tables,
    Kaplan-Meier survival data, and other findings as presented in a poster
    or paper. This is a SEPARATE corpus from search_clinical_trials -- it has
    no NCT ids, sponsors, or phase fields of its own. To relate an excerpt to
    a specific trial, look for the same drug/intervention name(s) or an
    explicit NCT id inside the excerpt's own text.

    Args:
        query: A semantic description of what to find, e.g. "progression-free
            survival for a PD-1 checkpoint inhibitor combination". Natural
            language, not keywords -- results are ranked by embedding
            similarity.
    """
    try:
        query_vector = embed_query(query)
        hits = _client().query_points(
            collection_name=COLLECTION_NAME_PDF,
            query=query_vector,
            limit=6,
        ).points
    except Exception as exc:  # surfaced to the agent as an observation
        return json.dumps({"error": f"Qdrant search failed: {exc}", "has_results": False})

    results = []
    for h in hits:
        meta = h.payload or {}
        results.append({
            "SourceFile": meta.get("SourceFile"),
            "ChunkIndex": meta.get("ChunkIndex"),
            "DocType": meta.get("DocType"),
            "score": round(float(h.score), 4),
            "Text": meta.get("Text") or "",
        })

    # Same kNN-always-returns-something caveat as search_clinical_trials --
    # see _is_grounded_literature.
    grounded = _is_grounded_literature(query, results)

    return json.dumps(
        {"query": query, "returned": len(results), "has_results": grounded,
         "chunks": results},
        indent=2,
    )


@tool
def search_fda_records(query: str) -> str:
    """Search FDA drug approval/application records (openFDA drug/drugsfda),
    embedded with the same OpenAI model as the other two collections.

    Use this for FDA REGULATORY/APPROVAL facts: whether a drug has an FDA
    application on file, its application number, the sponsor of record,
    approved brand names, active ingredients, and dosage forms. This is a
    SEPARATE corpus from search_clinical_trials -- it has no NCT ids, trial
    phases, or trial-level design fields; a returned record describes a drug
    PRODUCT's regulatory filing, not any specific trial.

    SCOPE NOTE: this collection is openFDA's drug/drugsfda partition only --
    approvals/applications. It does NOT include drug/event (adverse-event
    reports) or drug label text, so it cannot answer adverse-event/safety-
    signal or label-change questions; if asked those, say so rather than
    guessing from approval data alone.

    Args:
        query: A semantic description of what to find, e.g. "FDA approval
            status for a PD-1 checkpoint inhibitor" or a specific drug name.
            Natural language, not keywords -- results are ranked by embedding
            similarity.
    """
    try:
        query_vector = embed_query(query)
        hits = _client().query_points(
            collection_name=COLLECTION_NAME_FDA,
            query=query_vector,
            limit=6,
        ).points
    except Exception as exc:  # surfaced to the agent as an observation
        return json.dumps({"error": f"Qdrant search failed: {exc}", "has_results": False})

    results = []
    for h in hits:
        meta = h.payload or {}
        results.append({
            "ApplicationNumber": meta.get("ApplicationNumber"),
            "SponsorName": meta.get("SponsorName"),
            "BrandNames": meta.get("BrandNames"),
            "ActiveIngredients": meta.get("ActiveIngredients"),
            "DosageForms": meta.get("DosageForms"),
            "Rxcui": meta.get("Rxcui"),
            "ProductType": meta.get("ProductType"),
            "score": round(float(h.score), 4),
            "SourceURL": meta.get("SourceURL"),
        })

    # Same kNN-always-returns-something caveat as the other two tools -- see
    # _is_grounded_fda.
    grounded = _is_grounded_fda(query, results)

    return json.dumps(
        {"query": query, "returned": len(results), "has_results": grounded,
         "fda_records": results},
        indent=2,
    )


# =============================================================================
# TOOL -- knowledge graph (Neo4j): exact entity resolution, not vector search
# =============================================================================
_neo4j_driver: GraphDatabase.driver | None = None


def _graph_client():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _neo4j_driver


# Matches on three independent signals, any one of which is sufficient:
#   1. d.name CONTAINS entity      -- the raw drug name AS WRITTEN in a trial
#   2. c.standard_name = entity    -- the RxNorm generic/canonical name
#   3. entity IN c.brand_names     -- an official RxNorm trade name (built at
#                                      ingestion time via the free NIH RxNav
#                                      API -- see build_kg.py) -- this is
#                                      what resolves "Keytruda" -> the SAME
#                                      Concept node "pembrolizumab" is
#                                      attached to, in ONE Cypher query, no
#                                      live external API call needed here.
KG_MATCH_QUERY = """
MATCH (t:Trial)-[:INVESTIGATES]->(d:Drug)-[:MAPPED_TO_RXNORM]->(c:Concept)
WHERE toLower(d.name) CONTAINS toLower($entity)
   OR toLower(c.standard_name) = toLower($entity)
   OR ANY(b IN coalesce(c.brand_names, []) WHERE toLower(b) = toLower($entity))
RETURN DISTINCT
  t.id AS NCTId, t.title AS BriefTitle, t.phase AS Phase,
  t.status AS OverallStatus, t.sponsor AS LeadSponsorName,
  t.study_type AS studyType, t.conditions AS conditions,
  t.interventions_json AS interventions_json, t.summary AS BriefSummary,
  d.name AS MatchedDrugName, c.cui AS cui, c.standard_name AS standard_name
"""


@tool
def query_knowledge_graph(entity: str) -> str:
    """Query the biomedical knowledge graph for trials investigating a
    specific drug entity -- via exact Cypher graph traversal, not vector
    similarity.

    Prefer this tool FIRST whenever the question names a SPECIFIC drug --
    trade name or generic/RxNorm name -- rather than describing a topic in
    prose. It performs real entity resolution: a query for the brand name
    "Keytruda" is matched against the RxNorm concept it belongs to
    (pembrolizumab) and returns every trial investigating that concept,
    regardless of which name that trial's own record happens to use. This
    is deterministic, not a similarity guess -- an empty result means the
    entity genuinely is not in the graph, which is the signal to fall back
    to search_clinical_trials's semantic search instead (e.g. for a newer
    drug not yet resolved into the graph, or a topic that isn't one
    specific named entity).

    Args:
        entity: A drug name -- brand (e.g. "Keytruda") or generic/RxNorm
            standard name (e.g. "pembrolizumab"). Pass the name as given,
            not a rephrased description -- this is exact entity matching
            against the graph, not semantic search.
    """
    try:
        with _graph_client().session() as session:
            rows = list(session.run(KG_MATCH_QUERY, entity=entity))
    except Exception as exc:  # surfaced to the agent as an observation
        return json.dumps({"error": f"Neo4j query failed: {exc}", "has_results": False})

    trials = []
    resolved = {}
    for r in rows:
        try:
            interventions = json.loads(r["interventions_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            interventions = []
        trials.append({
            "NCTId": r["NCTId"],
            "BriefTitle": r["BriefTitle"],
            "Phase": r["Phase"],
            "OverallStatus": r["OverallStatus"],
            "LeadSponsorName": r["LeadSponsorName"],
            "studyType": r["studyType"],
            "conditions": r["conditions"],
            "interventions": interventions,
            "BriefSummary": r["BriefSummary"],
            "MappedConcept": {"cui": r["cui"], "standard_name": r["standard_name"],
                              "matched_drug_name": r["MatchedDrugName"]},
            "RetrievalSource": "knowledge_graph",
        })
        resolved[r["cui"]] = r["standard_name"]

    # Exact graph traversal, not kNN -- zero rows IS a genuine "not in the
    # graph" signal here, unlike vector search's kNN (which always returns
    # its k nearest points regardless of true relevance, hence the lexical
    # grounding checks the other two tools need but this one doesn't).
    return json.dumps({
        "entity": entity,
        "resolved_concepts": [{"cui": cui, "standard_name": name}
                              for cui, name in resolved.items()],
        "returned": len(trials),
        "has_results": len(trials) > 0,
        "trials": trials,
    }, indent=2)


TOOLS = [search_clinical_trials, search_pdf_literature, query_knowledge_graph,
         search_fda_records]


# =============================================================================
# GRAPH
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Set once, from the authenticated caller's JWT (see auth.py), by
    # whichever entry point built this state -- api.py's _initial_state()
    # or main()'s CLI invocation (None there, no caller to attribute to).
    # Read-only from every node's perspective: this is for LOGGING /
    # attribution ("which tenant asked this"), not access control -- no
    # node branches on it, and no Qdrant/Neo4j query is filtered by it. See
    # auth.py's own module docstring SCOPE NOTE for why that's a
    # deliberately separate, larger piece of work this does not attempt.
    tenant_id: Optional[str]
    tool_rounds: int
    # The validated Smart Table object produced by the terminal node (any of
    # synthesize_table / OutOfDomain / NoResultsFallback). Every exit path
    # from the graph sets this to the SAME type, so the FastAPI layer and
    # frontend never need to special-case a guardrail outcome vs. a normal
    # answer.
    result: Optional[SmartTableResponse]
    # --- guardrail state -----------------------------------------------------
    is_in_domain: Optional[bool]
    has_results: Optional[bool]
    synthesis_retries: int
    synthesis_error: Optional[str]
    # --- Map-Reduce state (Send-based parallel extraction) -------------------
    # operator.add on a list means "concatenate", not "overwrite" -- required
    # here because MULTIPLE extract_trial workers write to extracted_rows in
    # the same superstep. Without this reducer, LangGraph's default behaviour
    # (last write wins) would mean N-1 of every N parallel workers' output is
    # silently discarded instead of merged.
    retrieved_trials: Annotated[list[dict], operator.add]
    extracted_rows: Annotated[list[TrialRow], operator.add]
    # Federated retrieval: chunks from search_pdf_literature, accumulated the
    # same way as retrieved_trials. Unlike trials, literature chunks do NOT
    # each spawn their own extract_trial worker -- there is no "one row per
    # chunk" concept here, since a chunk isn't a trial. Instead the whole
    # deduped pool is broadcast into every worker's Send payload (see
    # continue_to_extraction) so each worker can judge which excerpts, if
    # any, are actually about ITS trial.
    retrieved_literature: Annotated[list[dict], operator.add]
    # Federated retrieval, third source: records from search_fda_records,
    # accumulated the same way as retrieved_literature -- an FDA approval
    # record doesn't spawn its own extract_trial worker either (it has no
    # NCTId), it is broadcast into every worker's Send payload so each can
    # judge whether it regulatorily matches ITS trial's drug(s).
    retrieved_fda: Annotated[list[dict], operator.add]
    # Per-worker input only. Set exclusively via the Send("extract_trial",
    # {"single_trial": ..., "literature": ..., "fda_records": ...}) payload in
    # continue_to_extraction -- no other node reads or writes these keys.
    single_trial: Optional[dict]
    literature: Optional[list[dict]]
    fda_records: Optional[list[dict]]


AGENT_SYSTEM = """You are a life sciences market intelligence analyst.

You have FOUR tools over FOUR independent sources -- pick whichever actually
match what the question is asking, not out of habit:

- query_knowledge_graph -- a Neo4j GRAPH of exact drug-entity relationships
  (Trial -[:INVESTIGATES]-> Drug -[:MAPPED_TO_RXNORM]-> RxNorm Concept),
  built by resolving raw trial drug names to standard RxNorm concepts.
  QUERY THIS FIRST whenever the question names a SPECIFIC drug -- trade name
  (e.g. "Keytruda") or generic/RxNorm name (e.g. "pembrolizumab") -- or asks
  about entity relationships, drug synonyms, or exact trial linkages for a
  named drug. It does exact graph traversal, not similarity search: it can
  resolve a brand name to its RxNorm concept and return every trial on that
  concept regardless of which name that trial's own record uses -- something
  vector search cannot guarantee, since a brand name and its generic don't
  reliably embed close together. An empty result means the entity genuinely
  is not in the graph yet -- THEN fall back to search_clinical_trials.
- search_clinical_trials -- the STRUCTURED trial registry (ClinicalTrials.gov
  records), searched by semantic similarity. Use it for trial design and
  administrative facts (NCT ids, phase, sponsor, study type, conditions,
  interventions) when the question is a topic/description rather than one
  named entity, or as the fallback when the knowledge graph has nothing.
- search_pdf_literature -- UNSTRUCTURED literature (conference posters, FDA
  filings) parsed via vision-based table extraction. Use it for reported
  RESULTS neither the registry nor the graph carries: efficacy metrics (ORR,
  PFS, OS), safety / adverse-event tables, Kaplan-Meier survival data, and
  other findings as presented in a poster or paper.
- search_fda_records -- FDA drug approval/application records (openFDA
  drug/drugsfda). Use it when the question is about FDA APPROVAL STATUS or
  APPROVED INDICATIONS/PRODUCTS: whether a drug has an FDA application on
  file, its application number, sponsor of record, approved brand names,
  active ingredients, or dosage forms. This corpus does NOT contain adverse-
  event reports or label text, so it cannot answer regulatory-warning or
  label-change questions -- say so rather than guessing if asked those.

These are not alternatives to pick one of -- for a holistic pipeline question
that touches trial design/entity relationships, reported results, AND
regulatory status (e.g. "what is the FDA approval status and Phase 3 efficacy
for X"), call the relevant tools CONCURRENTLY, in the same turn, and let the
evidence from each cross-reference the other. Do not assume registry or graph
data alone can answer an efficacy question, do not assume literature alone
can answer a design question, and do not assume the registry/graph/literature
can answer a regulatory-approval question: each tool only knows its own
corpus, and none substitutes for another.

Rules:
- Never answer from prior knowledge about specific trials or approvals. Every
  factual claim must come from a tool result.
- When the question restricts trials by phase (e.g. "Phase 3 trials"), you MUST
  pass phase_filter to search_clinical_trials. Do not filter mentally after
  the fact.
- You may search more than once to refine or broaden -- e.g. a second query
  with different wording if the first returns little of use.
- When you have enough evidence, stop calling tools and reply with a short
  plain-text note that you are ready. A separate synthesis step writes the
  final answer, so do not write it yourself."""

# --- Map stage prompt: EXACTLY one trial, no narrative, no other trials -----
# The literature pool IS shared across all workers (see continue_to_extraction
# and extract_trial_node) -- that's the one deliberate exception to "you never
# see any other trial's data," because unlike a trial record, a literature
# excerpt has no owning trial until a worker judges whether it matches theirs.
EXTRACTION_SYSTEM = """You are extracting exactly ONE row for a clinical
trials comparison grid, from exactly ONE retrieved trial record -- plus,
where genuinely relevant, findings from a shared pool of literature excerpts
(conference posters, FDA filings) and a shared pool of FDA drug approval
records, each retrieved separately from its own source.

This is the Map stage of a Map-Reduce pipeline: you never see any other
trial's registry record, only this one -- do not reference, compare against,
or assume anything about other trials in the corpus. The literature excerpts
and FDA records are the exception: they are shared across every worker
running this turn, most will have nothing to do with your trial, and it is
your job to judge which (if any) genuinely do.

STRICT CORPUS GROUNDING -- the hard boundary, now covering ALL THREE sources:
- The trial record, the literature excerpts, and the FDA records below are
  your ONLY sources. Your own pharmacological or regulatory knowledge is out
  of scope, even when you are confident it is correct.
- Do not state a drug's molecular target, modality, or mechanism unless one
  of these sources says so. If the trial record names an agent without
  describing how it works, the mechanism is unknown FOR OUR PURPOSES: set
  mechanism_described to false, and use mechanism_or_findings to report
  whatever the sources DO give you (population, biomarker, comparator,
  endpoint, study design, matching literature finding, or matching FDA
  approval status). Do not discard that detail -- a false flag with useful
  context is the goal, not a blank row.
- nct_id MUST be copied exactly from this record's own NCTId field -- never
  invented, never guessed from context.

FUSING LITERATURE INTO mechanism_or_findings:
- A literature excerpt is about YOUR trial ONLY if it explicitly names this
  trial's NCT id, OR its reported intervention combination distinctively
  matches this trial's own `interventions` -- the FULL combination, not one
  shared drug. Two different trials commonly share ONE overlapping agent
  (e.g. pembrolizumab appears across many unrelated trials as a backbone
  therapy) without being the same trial -- matching on that one common name
  alone is a FALSE match, and fusing one trial's efficacy numbers into a
  different trial's row because they happen to share a popular drug is
  exactly the kind of cross-trial data mixing the grounding discipline
  exists to prevent. Look for an uncommon/distinctive agent name, the full
  combination, or an explicit NCT id -- not partial overlap on a widely-used
  drug. Topical similarity alone (e.g. "also about lung cancer") is not
  enough either.
- When an excerpt genuinely matches, weave its reported efficacy metrics
  (ORR, PFS, OS), safety/adverse-event findings, or Kaplan-Meier results into
  mechanism_or_findings alongside the registry design details, and attribute
  it to literature (e.g. "per a conference poster on this trial, ORR was
  58.4%...") so a reader can tell registry design facts apart from reported
  results.
- When no excerpt matches, say nothing about literature at all -- do not
  force a connection. A row built from the registry alone is a complete,
  correct answer.

FUSING FDA RECORDS INTO mechanism_or_findings:
- An FDA record matches YOUR trial ONLY if one of its `BrandNames` or
  `ActiveIngredients` names the SAME drug as one of this trial's own
  `interventions` -- match on the specific ingredient/brand name, not on
  therapeutic class or indication. The same false-match risk as literature
  applies: a widely-used backbone agent appearing in an FDA record does not
  mean that record is about this specific trial's regimen, only about that
  one shared drug's own regulatory status.
- When an FDA record genuinely matches, weave its regulatory status into
  mechanism_or_findings -- e.g. "per FDA records, [drug] has an approved
  application (No. [ApplicationNumber]) sponsored by [SponsorName]" -- and
  attribute it to FDA data so a reader can tell registry/literature facts
  apart from regulatory status. Do not claim a drug is "FDA-approved" beyond
  what the record states (an application on file is not the same as an
  approved indication for this trial's specific use) -- report the fields as
  given, not an inference about approval scope.
- When no FDA record matches, say nothing about FDA status at all -- do not
  force a connection. A row built from the registry (plus literature, where
  applicable) alone is a complete, correct answer.

Each trial record carries structured pharmacology: `interventions` (a list of
{type, name}), `conditions`, and `studyType`. Use those fields as the
authoritative source for which agents are being tested -- name the specific
interventions rather than describing them generically, and do not rely on
parsing drug names out of the narrative BriefSummary."""

# --- Reduce stage prompt: prose only, from already-extracted rows ----------
REDUCER_SYSTEM = """You are writing the final analyst answer for a clinical
trials comparison.

This is the Reduce stage of a Map-Reduce pipeline: every trial has already
been independently extracted into a schema-validated row by a separate Map
worker. You are given ONLY that array of rows -- not raw database text, not
the original tool output. Treat every field already in a row as vetted; your
job is prose, not extraction.

Write the narrative_summary answering the user's question using ONLY the
provided rows.
- Do not add mechanism, target, or modality claims beyond what a row's own
  mechanism_or_findings / mechanism_described fields state. A row with
  mechanism_described = false means that trial's mechanism is unknown FOR OUR
  PURPOSES -- do not infer or supply one.
- Every factual claim must be followed by the NCTId in brackets, e.g.
  "targets PD-1 [NCT01234567]".
- Every nct_id you cite must appear in the provided rows. Never invent or
  reference an identifier that is not there.
- If the rows do not support an answer to the question, say so plainly.

Group related mechanisms of action where that aids the reader. Be specific
about drug targets and modalities exactly as the rows describe them."""

INTENT_SYSTEM = """You are a fast input classifier guarding a clinical trials
intelligence agent. Classify whether the user's question is in-domain:
clinical trials, oncology, pharmaceutical drugs or biologics, trial sponsors,
trial phases, or mechanisms of action -- including questions about drugs or
compounds you don't recognise, since our database may still cover them.

Answer False for anything else: recipes, general chit-chat, coding help,
unrelated science, or any topic that is not about clinical trials or pharma --
even if it superficially uses a medical-sounding word."""


def build_llm(model: str, max_tokens: int = MAX_TOKENS, timeout: int = 180):
    """ChatAnthropic bound to the tool schema, by default.

    temperature/top_p are deliberately NOT set for Anthropic -- claude-opus-5
    rejects them with a 400. Steer behaviour through the prompt instead.

    LLM_PROVIDER=nvidia is a TEMPORARY escape hatch, not a second production
    path: it reroutes every LLM in the graph (main reasoning, intent gate,
    extraction, synthesis) through NVIDIA's OpenAI-compatible endpoint via
    the exact client construction already verified working in
    evaluate_agent.py's build_judge() (ChatOpenAI against
    integrate.api.nvidia.com, not ChatNVIDIA -- see that function's own
    comment on why: ChatNVIDIA's aiohttp client has no configurable
    socket-read timeout and reliably died on a long call). Added so this
    graph can still be exercised end-to-end when Anthropic billing is
    blocked; NOT verified to have equivalent bind_tools()/
    with_structured_output() fidelity to Claude -- tool-calling and
    structured-output behavior genuinely differs across providers/models,
    so this is for unblocking a live test, not a claim the two are
    interchangeable in production.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "nvidia":
        from langchain_openai import ChatOpenAI

        key = os.getenv("NVIDIA_API_KEY")
        if not key:
            raise SystemExit(
                "[build_llm] LLM_PROVIDER=nvidia but NVIDIA_API_KEY is not set.\n"
                "            Add it to .env — free key at https://build.nvidia.com"
            )
        nvidia_model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        return ChatOpenAI(
            model=nvidia_model,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=key,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, max_tokens=max_tokens, timeout=timeout)


def make_graph(model: str, verbose: bool = True):
    llm = build_llm(model)
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    # --- Map stage: one structured call per trial, run in parallel ---------
    # No include_raw/retry here (unlike the Reducer below) -- a single-trial
    # extraction is a much smaller, lower-risk generation than the old
    # monolithic "extract everything AND write prose" call was. extract_trial
    # wraps this in a try/except instead: one flaky worker among N must not
    # be allowed to sink the whole parallel batch.
    #
    # NOTE (cost at scale): this reuses the same `model` as the rest of the
    # graph (e.g. claude-opus-5). That is fine at today's corpus size, but
    # once retrieval genuinely returns "hundreds of documents" per the
    # motivation for this refactor, hundreds of full-price Opus calls per
    # request is a real cost line to plan for -- worth revisiting with a
    # cheaper/faster model here, the same reasoning INTENT_MODEL already
    # applies to the intent gate -- not done here to keep this change scoped
    # to what was asked.
    extraction_llm = llm.with_structured_output(TrialRow)

    # --- Reduce stage: prose only -- table_data is already fixed by the Map
    # stage, so this call carries far less risk than the old Synthesis call
    # did, but keeps the same validate-or-retry shape for consistency.
    narrative_llm = llm.with_structured_output(NarrativeSummary, include_raw=True)

    # Separate, cheap/fast model for the input guardrail -- see INTENT_MODEL.
    # max_tokens=1536, not the smaller value this had before: verified
    # directly under LLM_PROVIDER=nvidia (see build_llm's docstring) that
    # NVIDIA's Nemotron model reliably burns several hundred tokens on its
    # own internal reasoning before it ever emits the tiny IntentClassification
    # payload -- 512 hit openai.LengthFinishReasonError consistently, not
    # intermittently, on ordinary in-domain questions. Harmless headroom for
    # Claude, which was never close to that ceiling for this task.
    intent_llm = build_llm(INTENT_MODEL, max_tokens=1536, timeout=60) \
        .with_structured_output(IntentClassification)

    # --- node: IntentClassifier --------------------------------------------
    def intent_classifier_node(state: AgentState) -> dict:
        question = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        verdict: IntentClassification = intent_llm.invoke([
            SystemMessage(content=INTENT_SYSTEM),
            HumanMessage(content=question),
        ])
        if verbose:
            _trace_intent(verdict)
        return {"is_in_domain": verdict.is_in_domain}

    # --- node: OutOfDomain (deterministic, no LLM call) --------------------
    def out_of_domain_node(state: AgentState) -> dict:
        if verbose:
            print(f"\n{'─' * 78}\n▶ NODE: OutOfDomain  "
                  f"(deterministic refusal, no LLM call, no Qdrant hit)\n{'─' * 78}")
            print(f"  {OUT_OF_DOMAIN_MESSAGE}")
        result = SmartTableResponse(narrative_summary=OUT_OF_DOMAIN_MESSAGE, table_data=[])
        return {"messages": [AIMessage(content=OUT_OF_DOMAIN_MESSAGE)], "result": result}

    # --- node: Agent ---------------------------------------------------------
    def agent_node(state: AgentState) -> dict:
        msgs = [SystemMessage(content=AGENT_SYSTEM)] + state["messages"]

        # Cap the ReACT loop: past the limit, stop offering tools so the
        # agent must conclude and routing falls through to synthesis.
        model_to_use = llm if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS else llm_with_tools
        reply = model_to_use.invoke(msgs)

        if verbose:
            _trace_agent(reply, state.get("tool_rounds", 0))
        return {"messages": [reply]}

    # --- node: Tool Node (wrapped for tracing + grounding aggregation) -----
    def tools_node(state: AgentState) -> dict:
        out = tool_node.invoke(state)
        round_grounded = False
        new_trials: list[dict] = []
        new_literature: list[dict] = []
        new_fda: list[dict] = []
        for m in out["messages"]:
            if verbose:
                _trace_tool_result(m)
            try:
                payload = json.loads(m.content)
                # Any one tool's grounding is enough -- a query that only
                # search_pdf_literature (or search_fda_records) can ground
                # should still be treated as "something real was found," not
                # routed to NoResultsFallback just because the OTHER tools
                # came up empty on their own corpus.
                if payload.get("has_results"):
                    round_grounded = True
                new_trials.extend(payload.get("trials", []))
                new_literature.extend(payload.get("chunks", []))
                new_fda.extend(payload.get("fda_records", []))
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # Sticky-OR across rounds: once ANY round finds grounded results,
        # stay True even if a later refinement search comes back empty --
        # otherwise a single unlucky follow-up query would kill an
        # already-successful earlier round. An ungrounded FIRST round (no
        # prior True to inherit) still correctly triggers NoResultsFallback.
        prior = state.get("has_results")
        has_results = bool(prior) or round_grounded
        return {"messages": out["messages"],
                "tool_rounds": state.get("tool_rounds", 0) + 1,
                "has_results": has_results,
                "retrieved_trials": new_trials,
                "retrieved_literature": new_literature,
                "retrieved_fda": new_fda}

    # --- node: NoResultsFallback (deterministic, no LLM call) ---------------
    def no_results_fallback_node(state: AgentState) -> dict:
        if verbose:
            print(f"\n{'─' * 78}\n▶ NODE: NoResultsFallback  "
                  f"(deterministic, no LLM call — bypassing Synthesis)\n{'─' * 78}")
            print(f"  {NO_RESULTS_MESSAGE}")
        result = SmartTableResponse(narrative_summary=NO_RESULTS_MESSAGE, table_data=[])
        return {"messages": [AIMessage(content=NO_RESULTS_MESSAGE)], "result": result}

    # --- node: extract_trial (Map stage -- one parallel worker per trial) --
    def extract_trial_node(state: AgentState) -> dict:
        trial = state["single_trial"]
        literature = state.get("literature") or []
        fda_records = state.get("fda_records") or []
        nct_id = trial.get("NCTId", "?")
        started = time.time()
        if verbose:
            print(f"\n{'─' * 78}\n▶ NODE: extract_trial  worker={nct_id}  "
                  f"(Map stage, parallel, {len(literature)} literature "
                  f"excerpt(s), {len(fda_records)} FDA record(s) available)"
                  f"\n{'─' * 78}")

        prompt = (
            f"TRIAL RECORD (structured registry -- the primary source for "
            f"this row):\n{json.dumps(trial, indent=2)}\n\n"
        )
        if literature:
            prompt += (
                f"LITERATURE EXCERPTS (shared pool from search_pdf_literature "
                f"-- fuse in ONLY what genuinely matches THIS trial, per the "
                f"system instructions):\n{json.dumps(literature, indent=2)}\n\n"
            )
        else:
            prompt += "LITERATURE EXCERPTS: (none retrieved this run)\n\n"

        if fda_records:
            prompt += (
                f"FDA RECORDS (shared pool from search_fda_records -- fuse in "
                f"ONLY what genuinely matches THIS trial's drug(s), per the "
                f"system instructions):\n{json.dumps(fda_records, indent=2)}"
            )
        else:
            prompt += "FDA RECORDS: (none retrieved this run)"

        try:
            row: TrialRow = extraction_llm.invoke([
                SystemMessage(content=EXTRACTION_SYSTEM),
                HumanMessage(content=prompt),
            ])
        except Exception as exc:  # one flaky worker must not sink the batch
            if verbose:
                print(f"  ✗ extraction failed for {nct_id} "
                      f"({time.time() - started:.1f}s): {exc} — dropping this row")
            return {"extracted_rows": []}

        if verbose:
            print(f"  ✓ {row.nct_id}  phase={row.phase!r}  "
                  f"mechanism_described={row.mechanism_described}  "
                  f"({time.time() - started:.1f}s)")
        return {"extracted_rows": [row]}

    # --- node: synthesize_table (Reduce stage) -------------------------------
    def synthesize_table_node(state: AgentState) -> dict:
        question = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        rows = state.get("extracted_rows", [])
        retries = state.get("synthesis_retries", 0)

        if verbose:
            label = f"  (retry {retries}/{MAX_SYNTHESIS_RETRIES})" if retries else ""
            print(f"\n{'─' * 78}\n▶ NODE: synthesize_table{label}  (Reduce stage — "
                  f"writing narrative from {len(rows)} extracted row(s); raw Qdrant "
                  f"text is not visible here)\n{'─' * 78}")

        if not rows:
            empty = SmartTableResponse(
                narrative_summary="No trials were retrieved from the database, "
                                  "so there is no evidence to answer from.",
                table_data=[],
            )
            return {"messages": [AIMessage(content=empty.narrative_summary)],
                    "result": empty}

        prompt = (
            f"USER QUESTION:\n{question}\n\n"
            f"EXTRACTED TRIAL ROWS (the only permitted source -- already "
            f"validated, structured records produced by independent Map-stage "
            f"workers; you do not have access to raw retrieval text):\n"
            + json.dumps([r.model_dump() for r in rows], indent=2)
        )
        prior_error = state.get("synthesis_error")
        if prior_error:
            # This is the "inject the error context back into state" step:
            # the failed attempt's own validation error becomes part of the
            # next prompt, so the model sees exactly what it got wrong.
            prompt += (
                f"\n\nYOUR PREVIOUS ATTEMPT FAILED SCHEMA VALIDATION:\n{prior_error}\n"
                f"Correct the structure this time — match the schema exactly."
            )

        outcome: dict = narrative_llm.invoke([
            SystemMessage(content=REDUCER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        parsed: NarrativeSummary | None = outcome.get("parsed")
        error = outcome.get("parsing_error")

        if parsed is not None and error is None:
            result = SmartTableResponse(
                narrative_summary=parsed.narrative_summary, table_data=rows
            )
            if verbose:
                print(f"  ✓ narrative synthesized — {len(rows)} table rows "
                      f"carried through unchanged from the Map stage")
            return {"messages": [AIMessage(content=result.narrative_summary)],
                    "result": result, "synthesis_error": None}

        # --- validation failed: retry or fail closed ------------------------
        err_text = str(error) if error else "model did not return the expected structure"
        retries += 1
        if verbose:
            print(f"  ✗ ValidationError (attempt {retries}/{MAX_SYNTHESIS_RETRIES}): "
                  f"{err_text[:200]}")

        if retries > MAX_SYNTHESIS_RETRIES:
            fallback = SmartTableResponse(
                narrative_summary="The agent was unable to produce a validly "
                                  "structured response after multiple attempts. "
                                  "Please rephrase your question and try again.",
                table_data=rows,
            )
            if verbose:
                print(f"  ✗ retries exhausted — returning deterministic fallback "
                      f"(extracted rows preserved)")
            return {"messages": [AIMessage(content=fallback.narrative_summary)],
                    "result": fallback}

        # No "result" key set -> route_after_synthesis loops back to retry.
        return {"synthesis_retries": retries, "synthesis_error": err_text}

    # --- conditional edges ---------------------------------------------------
    def route_intent(state: AgentState) -> str:
        return "agent" if state.get("is_in_domain") else "out_of_domain"

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "synthesis"

    def continue_to_extraction(state: AgentState):
        """Conditional edge from `tools` -- the Mapper.

        Folds in the pre-existing NoResultsFallback guardrail (a plain node
        name, no LLM call) so an ungrounded search still fails closed exactly
        as before. Otherwise, dedupes retrieved trials by NCTId (a trial can
        legitimately appear more than once if the agent issues overlapping
        tool calls in one round), dedupes retrieved literature by
        (SourceFile, ChunkIndex), and returns one Send per distinct trial --
        this is what dynamically spawns N parallel extract_trial branches at
        runtime, N being however many trials Qdrant actually returned. The
        deduped literature pool is broadcast into EVERY Send -- each worker
        independently judges which excerpts (if any) belong to its trial.
        """
        if state.get("has_results") is False:
            return "no_results_fallback"

        seen: set[str] = set()
        trials: list[dict] = []
        for t in state.get("retrieved_trials", []):
            nct = t.get("NCTId")
            if nct and nct not in seen:
                seen.add(nct)
                trials.append(t)

        if not trials:
            # has_results is True (checked above) but there is no trial to
            # spawn a Map worker for -- either a generic query grounded on an
            # empty token set (pre-existing edge case), or, newly possible
            # now that retrieval is federated across two tools: the agent
            # called ONLY search_pdf_literature this round, so has_results
            # came from literature alone and search_clinical_trials never
            # ran. A literature excerpt cannot back a table row on its own
            # (nct_id must come from a trial record) -- route straight to
            # the Reducer rather than returning an empty Send list, which
            # would silently dead-end the graph (no further supersteps run,
            # "result" is never set). synthesize_table already handles an
            # empty extracted_rows list via its own deterministic message.
            if verbose:
                print(f"\n{'─' * 78}\n▶ MAPPER: continue_to_extraction\n{'─' * 78}")
                print("  has_results=True but no distinct trial retrieved "
                      "(literature alone cannot back a table row) — routing "
                      "straight to synthesize_table")
            return "synthesize_table"

        seen_lit: set[tuple] = set()
        literature: list[dict] = []
        for c in state.get("retrieved_literature", []):
            key = (c.get("SourceFile"), c.get("ChunkIndex"))
            if key not in seen_lit:
                seen_lit.add(key)
                literature.append(c)

        seen_fda: set[str] = set()
        fda_records: list[dict] = []
        for r in state.get("retrieved_fda", []):
            app_no = r.get("ApplicationNumber")
            if app_no and app_no not in seen_fda:
                seen_fda.add(app_no)
                fda_records.append(r)

        if verbose:
            print(f"\n{'─' * 78}\n▶ MAPPER: continue_to_extraction\n{'─' * 78}")
            print(f"  {len(trials)} distinct trial(s), {len(literature)} distinct "
                  f"literature excerpt(s), {len(fda_records)} distinct FDA "
                  f"record(s) — fanning out to {len(trials)} parallel "
                  f"extract_trial worker(s) via Send (literature + FDA pools "
                  f"shared across all of them)")
            for t in trials:
                print(f"    • Send(\"extract_trial\", single_trial={t.get('NCTId')}, "
                      f"literature={len(literature)} excerpt(s), "
                      f"fda_records={len(fda_records)})")

        return [Send("extract_trial",
                     {"single_trial": t, "literature": literature,
                      "fda_records": fda_records})
                for t in trials]

    def route_after_synthesis(state: AgentState) -> str:
        return "done" if state.get("result") is not None else "retry"

    g = StateGraph(AgentState)
    g.add_node("intent_classifier", intent_classifier_node)
    g.add_node("out_of_domain", out_of_domain_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("no_results_fallback", no_results_fallback_node)
    g.add_node("extract_trial", extract_trial_node)
    g.add_node("synthesize_table", synthesize_table_node)

    g.add_edge(START, "intent_classifier")
    g.add_conditional_edges("intent_classifier", route_intent,
                            {"agent": "agent", "out_of_domain": "out_of_domain"})
    g.add_edge("out_of_domain", END)

    g.add_conditional_edges("agent", route, {"tools": "tools", "synthesis": "synthesize_table"})
    # `tools` fans out via Send (Map), falls through to NoResultsFallback, or
    # (federated-retrieval edge case: literature grounded the round but no
    # trial was retrieved to attach it to) goes straight to the Reducer --
    # continue_to_extraction returns whichever fits; path_map only needs to
    # cover the plain-string branches, Send objects are used directly.
    g.add_conditional_edges("tools", continue_to_extraction,
                            {"no_results_fallback": "no_results_fallback",
                             "synthesize_table": "synthesize_table"})
    g.add_edge("no_results_fallback", END)
    g.add_edge("extract_trial", "synthesize_table")  # Map workers join here (Reduce)

    g.add_conditional_edges("synthesize_table", route_after_synthesis,
                            {"retry": "synthesize_table", "done": END})

    return g.compile()


# =============================================================================
# tracing helpers -- make the thought process visible
# =============================================================================
def _text_of(msg) -> str:
    """AIMessage.content may be a string or a list of blocks."""
    c = msg.content
    if isinstance(c, str):
        return c
    parts = []
    for block in c or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _trace_intent(verdict: IntentClassification) -> None:
    print(f"\n{'─' * 78}\n▶ NODE: IntentClassifier  (model={INTENT_MODEL})\n{'─' * 78}")
    print(f"  is_in_domain = {verdict.is_in_domain}")
    print(f"  reason       = {verdict.reason}")


def _trace_agent(reply, round_no: int) -> None:
    print(f"\n{'─' * 78}\n▶ NODE: Agent  (tool round {round_no})\n{'─' * 78}")
    reasoning = _text_of(reply).strip()
    if reasoning:
        for line in textwrap.wrap(reasoning, 74)[:8]:
            print(f"  {line}")
    for call in getattr(reply, "tool_calls", None) or []:
        print(f"\n  ⚙ TOOL CALL → {call['name']}")
        for k, v in (call.get("args") or {}).items():
            print(f"      {k} = {v!r}")
    if not getattr(reply, "tool_calls", None):
        print("\n  (no tool calls — routing to Synthesis)")


def _trace_tool_result(msg) -> None:
    # Three different tools land here, each with its own payload shape --
    # printed with a tool-specific label so the trace makes it unambiguous
    # which corpus was actually hit, not just that "a" tool ran. Checked by
    # tool_name explicitly (not by which JSON keys are present) because
    # query_knowledge_graph's payload ALSO has a "trials" key -- same shape
    # deliberately, so it flows through the Map-Reduce pipeline unchanged --
    # and would otherwise be misdetected as search_clinical_trials.
    tool_name = getattr(msg, "name", None) or "?"
    print(f"\n{'─' * 78}\n▶ NODE: Tool Node  (tool={tool_name})\n{'─' * 78}")
    try:
        payload = json.loads(msg.content)
    except Exception:
        print(f"  {str(msg.content)[:300]}")
        return
    if "error" in payload:
        print(f"  ERROR: {payload['error']}")
        return

    if tool_name == "query_knowledge_graph":
        print(f"  entity           : {payload.get('entity')!r}")
        print(f"  resolved_concepts: {payload.get('resolved_concepts')}  "
              f"(exact graph match -- brand/generic resolved via Neo4j, no vector search)")
        print(f"  returned         : {payload.get('returned')} trial(s)")
        print(f"  has_results      : {payload.get('has_results')} (exact traversal -- "
              f"no lexical grounding heuristic needed, unlike kNN)")
        for t in payload.get("trials", []):
            title = textwrap.shorten(t.get("BriefTitle") or "", 50, placeholder="…")
            mc = t.get("MappedConcept") or {}
            print(f"    • {t.get('NCTId')}  via {mc.get('matched_drug_name')!r} "
                  f"-> {mc.get('standard_name')!r} (CUI {mc.get('cui')})  {title}")
    elif "trials" in payload:
        print(f"  query        : {payload.get('query')!r}")
        print(f"  phase_filter : {payload.get('phase_filter')!r}")
        print(f"  returned     : {payload.get('returned')} trials (kNN, always up to limit)")
        print(f"  has_results  : {payload.get('has_results')} (lexical grounding check)")
        for t in payload.get("trials", []):
            title = textwrap.shorten(t.get("BriefTitle") or "", 58, placeholder="…")
            print(f"    • {t.get('NCTId')}  score={t.get('score')}  "
                  f"[{'/'.join(t.get('Phase') or [])}]  {title}")
    elif "chunks" in payload:
        print(f"  query        : {payload.get('query')!r}")
        print(f"  returned     : {payload.get('returned')} literature excerpt(s) "
              f"(kNN, always up to limit)")
        print(f"  has_results  : {payload.get('has_results')} (lexical grounding check)")
        for c in payload.get("chunks", []):
            preview = textwrap.shorten((c.get("Text") or "").replace("\n", " "),
                                       58, placeholder="…")
            print(f"    • {c.get('SourceFile')}#{c.get('ChunkIndex')}  "
                  f"score={c.get('score')}  {preview}")
    elif "fda_records" in payload:
        print(f"  query        : {payload.get('query')!r}")
        print(f"  returned     : {payload.get('returned')} FDA record(s) "
              f"(kNN, always up to limit)")
        print(f"  has_results  : {payload.get('has_results')} (lexical grounding check)")
        for r in payload.get("fda_records", []):
            brands = ", ".join(r.get("BrandNames") or []) or "(no brand name)"
            print(f"    • {r.get('ApplicationNumber')}  score={r.get('score')}  "
                  f"{textwrap.shorten(brands, 40, placeholder='…')}  "
                  f"sponsor={r.get('SponsorName')!r}")


# =============================================================================
def preflight() -> int:
    """Fail with actionable messages rather than a stack trace."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[preflight] ANTHROPIC_API_KEY is not set.\n"
              "            Add it to .env:  ANTHROPIC_API_KEY=sk-ant-...",
              file=sys.stderr)
        return 1
    try:
        c = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        if not c.collection_exists(COLLECTION_NAME):
            print(f"[preflight] Qdrant collection '{COLLECTION_NAME}' is missing.\n"
                  "            Run: python fetch_and_embed_trials.py", file=sys.stderr)
            return 1
        n = c.get_collection(COLLECTION_NAME).points_count
    except Exception as exc:
        print(f"[preflight] cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT} -> {exc}\n"
              "            docker run -d -p 6333:6333 -p 6334:6334 \\\n"
              "              -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest",
              file=sys.stderr)
        return 1
    print(f"[preflight] Qdrant OK — {n} points in '{COLLECTION_NAME}'")
    return 0


# Smart Table verification query
VERIFICATION_QUESTION = (
    "Build a comparative table of Phase 3 oncology trials showing drug names, "
    "sponsors, and key mechanisms."
)

# Previous milestones' questions, kept for regression runs:
#   --question "What are the mechanisms of action for the Phase 3 oncology
#               trials in our database?"
#   --question "What is the best recipe for chocolate chip cookies?"          (AC2: OutOfDomain)
#   --question "Compare Phase 3 trials for XYZ-Fake-Drug-999."                (AC3: NoResultsFallback)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=VERIFICATION_QUESTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quiet", action="store_true", help="hide the trace")
    args = parser.parse_args()

    print("=" * 78)
    print("medical-rag :: multi-agent RAG research agent (deterministic orchestrator)")
    print("=" * 78)
    if preflight() != 0:
        return 1
    print(f"[preflight] model: {args.model}  |  intent gate: {INTENT_MODEL}")

    graph = make_graph(args.model, verbose=not args.quiet)

    print(f"\n{'=' * 78}\nQUESTION\n{'=' * 78}")
    for line in textwrap.wrap(args.question, 74):
        print(f"  {line}")

    final = graph.invoke(
        {"messages": [HumanMessage(content=args.question)],
         "tenant_id": None,  # no authenticated caller on the CLI path -- see auth.py
         "tool_rounds": 0, "result": None,
         "is_in_domain": None, "has_results": None,
         "synthesis_retries": 0, "synthesis_error": None,
         "retrieved_trials": [], "extracted_rows": [], "retrieved_literature": [],
         "retrieved_fda": []},
        config={"recursion_limit": 25},
    )

    result: SmartTableResponse | None = final.get("result")
    if result is None:
        print("\n[error] graph produced no SmartTableResponse", file=sys.stderr)
        return 1

    # --- 1. narrative_summary as prose ------------------------------------
    print(f"\n{'=' * 78}\nNARRATIVE SUMMARY\n{'=' * 78}\n")
    for para in result.narrative_summary.split("\n"):
        print("\n".join(textwrap.wrap(para, 76)) if para.strip() else "")

    # --- 2. table_data as formatted JSON ----------------------------------
    print(f"\n{'=' * 78}\nTABLE DATA  ({len(result.table_data)} rows)"
          f"  — JSON payload for the frontend grid\n{'=' * 78}")
    print(json.dumps([r.model_dump() for r in result.table_data],
                     indent=2, ensure_ascii=False))

    # --- 3. human-readable grid preview -----------------------------------
    if result.table_data:
        print(f"\n{'=' * 78}\nGRID PREVIEW\n{'=' * 78}")
        hdr = f"{'NCT ID':<13}{'PHASE':<16}{'MECH?':<7}{'SPONSOR':<26}{'INTERVENTIONS'}"
        print(hdr + "\n" + "-" * 78)
        for r in result.table_data:
            ivs = ", ".join(r.interventions[:3]) + ("…" if len(r.interventions) > 3 else "")
            print(f"{r.nct_id:<13}{r.phase:<16}"
                  f"{('yes' if r.mechanism_described else 'NO'):<7}"
                  f"{textwrap.shorten(r.sponsor, 24, placeholder='…'):<26}"
                  f"{textwrap.shorten(ivs, 26, placeholder='…')}")

    # --- 4. guardrail + validation report -----------------------------------
    import re as _re
    retrieved = sorted({
        t["NCTId"]
        for m in final["messages"] if isinstance(m, ToolMessage)
        for t in (json.loads(m.content).get("trials", [])
                  if m.content.strip().startswith("{") else [])
        if t.get("NCTId")
    })
    cited = sorted(set(_re.findall(r"NCT\d{8}", result.narrative_summary)))
    row_ids = [r.nct_id for r in result.table_data]

    bad_cites = [c for c in cited if c not in retrieved]
    bad_rows = [i for i in row_ids if i not in retrieved]
    dupes = sorted({i for i in row_ids if row_ids.count(i) > 1})

    print(f"\n{'=' * 78}\nGRAPH PATH TAKEN\n{'=' * 78}")
    print(f"  is_in_domain      : {final.get('is_in_domain')}")
    print(f"  has_results       : {final.get('has_results')}")
    print(f"  tool_rounds       : {final.get('tool_rounds')}")
    print(f"  extracted_rows    : {len(final.get('extracted_rows') or [])}  "
          f"(parallel extract_trial workers spawned via Send)")
    print(f"  synthesis_retries : {final.get('synthesis_retries')}")

    if final.get("is_in_domain") is False:
        print("  path              : IntentClassifier → OutOfDomain → END")
        print(f"\n{'=' * 78}\nGUARDRAIL VALIDATION\n{'=' * 78}")
        ok = (result.narrative_summary == OUT_OF_DOMAIN_MESSAGE
              and not result.table_data and not retrieved)
        print("  ✓ PASS — deterministic refusal, zero tool calls"
              if ok else "  ✗ FAIL — expected the exact OutOfDomain message with no retrieval")
        return 0 if ok else 1

    if final.get("has_results") is False:
        print("  path              : IntentClassifier → Agent → Tools → NoResultsFallback → END")
        print(f"\n{'=' * 78}\nGUARDRAIL VALIDATION\n{'=' * 78}")
        ok = (result.narrative_summary == NO_RESULTS_MESSAGE and not result.table_data)
        print("  ✓ PASS — deterministic no-results message, Synthesis bypassed, no hallucination"
              if ok else "  ✗ FAIL — expected the exact NoResultsFallback message")
        return 0 if ok else 1

    print("  path              : IntentClassifier → Agent → Tools → "
          "continue_to_extraction ─Send×N→ extract_trial (Map, parallel) → "
          "synthesize_table (Reduce) → END")

    print(f"\n{'=' * 78}\nSTRUCTURED OUTPUT VALIDATION\n{'=' * 78}")
    print(f"  type returned                : {type(result).__name__}")
    print(f"  NCTIds retrieved (any tool)  : {len(retrieved)}")
    print(f"  narrative citations          : {len(cited)}")
    print(f"  table_data rows              : {len(row_ids)}")
    if row_ids:
        print(f"  rows with >=1 intervention   : "
              f"{sum(1 for r in result.table_data if r.interventions)}/{len(row_ids)}")
        n_desc = sum(1 for r in result.table_data if r.mechanism_described)
        print(f"  mechanism_described = true   : {n_desc}/{len(row_ids)}")

    ok = True
    if not isinstance(result, SmartTableResponse):
        print("  ✗ FAIL — not a SmartTableResponse instance"); ok = False
    if bad_cites:
        print(f"  ✗ FAIL — narrative cites unretrieved ids: {bad_cites}"); ok = False
    if bad_rows:
        print(f"  ✗ FAIL — table rows with unretrieved ids: {bad_rows}"); ok = False
    if dupes:
        print(f"  ✗ FAIL — duplicate rows: {dupes}"); ok = False
    if not row_ids:
        print("  ✗ FAIL — table_data is empty"); ok = False
    if not cited:
        print("  ✗ FAIL — narrative_summary has no NCTId citations"); ok = False

    if ok:
        print("  ✓ PASS — validated SmartTableResponse; every nct_id in the "
              "narrative and the table came from a real tool result")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
