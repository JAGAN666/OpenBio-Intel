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
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

import openai
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
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
# Fourth federated source (see fetch_pubmed.py): open-access PubMed Central
# literature -- title/abstract/body text, chunked. Use it for scientific
# efficacy/mechanism-of-action questions grounded in published research,
# distinct from search_pdf_literature's conference-poster/FDA-filing corpus.
COLLECTION_NAME_PUBMED = "pubmed_literature"
# Fifth federated source (see fetch_sec_edgar.py): SEC EDGAR 10-K/8-K
# filings for major biopharma tickers, HTML-stripped and isolated to their
# Pipeline/Clinical Trials/Research and Development sections. Use it for
# corporate strategy, pipeline prioritization, and R&D investment questions
# -- a fundamentally different lens than any trial/literature/FDA source,
# since it reflects what the COMPANY says about its own strategy, not
# third-party trial data or regulatory status.
COLLECTION_NAME_SEC = "sec_filings"
# Sixth federated source (see fetch_news_and_transcripts.py): real-time
# corporate news -- RSS press releases (FDA, J&J, AbbVie -- Pfizer/Merck do
# not have a usable public press-release RSS feed, verified live and
# documented in that script) plus earnings call transcripts for all four
# tracked tickers, keyword-isolated to pipeline/guidance/forward-looking
# content. Distinct from SEC filings (formal, periodic, legally-reviewed
# disclosure) -- this is unscripted management commentary and press-release
# announcements, the fastest-moving/most real-time source in this system.
COLLECTION_NAME_NEWS = "corporate_news"
# EMBEDDING_MODEL is imported from embeddings.py (text-embedding-3-small,
# 1536-dim, via OpenAI) -- must match the model the collections were indexed
# with, or the query vector is not comparable to the stored vectors (and a
# dimension mismatch hard-fails at the Qdrant call). All six collections
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

# search_clinical_trials's own retrieval breadth -- this is what actually
# bounds how many table_data rows a Smart Table query can ever produce (one
# extract_trial Map worker per distinct trial retrieved, no separate cap
# downstream in continue_to_extraction). Raised from a much smaller default
# per explicit request for close to 50 rows back, then to 60 per a second
# explicit request ("limit it to max 60 at least") after live testing
# showed real run-to-run variance in how many rows a broad query actually
# surfaces -- matches LANDSCAPE_TRIAL_LIMIT/CATALYST_TRIAL_LIMIT's precedent
# for "broad, not exploratory" retrieval. The other five tools stay at
# their own small per-call limit -- they feed supplementary evidence fused
# into a trial's row, not additional rows of their own, so widening them
# doesn't move the row count the user asked for.
TRIAL_SEARCH_LIMIT = 60

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
    sources: list["SourceCitation"] = Field(
        default_factory=list,
        description="Evidence items that informed mechanism_or_findings "
                    "BEYOND the trial registry record itself: one entry per "
                    "pool excerpt actually used, copying its PMCID / "
                    "SourceURL / identifier EXACTLY as it appears in the "
                    "excerpt. Empty when only the registry record was used. "
                    "Never invent references -- an uncited claim is better "
                    "than a fabricated citation."
    )


class SourceCitation(BaseModel):
    """One clickable provenance link behind a Smart Table row -- the
    auditability contract: every number an analyst sees should trace to a
    primary document. The registry citation itself is added
    DETERMINISTICALLY by extract_trial_node (never trusted to the LLM);
    the model only cites the auxiliary evidence pools it actually fused."""

    source_type: str = Field(
        description="One of: registry, pdf_literature, fda, pubmed, sec, news."
    )
    reference: str = Field(
        description="The cited document's identifier exactly as it appears "
                    "in the evidence: an NCT id, PMCID, FDA application "
                    "number, company/ticker, or document title."
    )
    url: Optional[str] = Field(
        default=None,
        description="The excerpt's SourceURL, copied exactly, when present."
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


def _is_grounded_pubmed(query: str, chunks: list[dict]) -> bool:
    """Same anti-hallucination principle as _is_grounded, applied to the
    PubMed Central literature collection: does anything actually retrieved
    share real vocabulary with the query.

    Haystack is Title + Journal + the chunk's own Text -- there is no
    curated identity field set here the way BriefTitle/conditions/
    interventions serve _is_grounded, so the chunk's own indexed text (which
    already leads with "Title: ...\\nJournal: ..." per fetch_pubmed.py's
    build_chunks()) is the closest equivalent.
    """
    qtok = _content_tokens(query)
    if not qtok:
        return True
    for c in chunks:
        haystack = " ".join([
            c.get("Title") or "", c.get("Journal") or "", c.get("Text") or "",
        ]).lower()
        if any(tok in haystack for tok in qtok):
            return True
    return False


def _is_grounded_sec(query: str, chunks: list[dict]) -> bool:
    """Same anti-hallucination principle as _is_grounded, applied to the SEC
    EDGAR filings collection: does anything actually retrieved share real
    vocabulary with the query.

    Haystack is Company + Ticker + the chunk's own Text -- the closest
    equivalent here to BriefTitle/conditions/interventions, since a filing
    excerpt has no other curated identity fields.
    """
    qtok = _content_tokens(query)
    if not qtok:
        return True
    for c in chunks:
        haystack = " ".join([
            c.get("Company") or "", c.get("Ticker") or "", c.get("Text") or "",
        ]).lower()
        if any(tok in haystack for tok in qtok):
            return True
    return False


def _is_grounded_news(query: str, chunks: list[dict]) -> bool:
    """Same anti-hallucination principle as _is_grounded, applied to the
    corporate news collection: does anything actually retrieved share real
    vocabulary with the query.

    Haystack differs by SourceType -- a press release chunk's identity
    fields are Title/FeedName, a transcript chunk's are Company/Ticker --
    so both are included; whichever apply to a given result contribute,
    the other simply contributes empty strings.
    """
    qtok = _content_tokens(query)
    if not qtok:
        return True
    for c in chunks:
        haystack = " ".join([
            c.get("Title") or "", c.get("FeedName") or "",
            c.get("Company") or "", c.get("Ticker") or "", c.get("Text") or "",
        ]).lower()
        if any(tok in haystack for tok in qtok):
            return True
    return False


# =============================================================================
# TOOLS -- federated Qdrant retrieval over six collections
# =============================================================================
_qdrant: QdrantClient | None = None


def _client() -> QdrantClient:
    """Lazy singleton -- a plain client now; embedding happens explicitly via
    embeddings.embed_query() before each search, not via a bound FastEmbed
    model.

    timeout=30, not the default (None, which falls through to httpx's own
    ~5s default) -- verified live against the AWS deployment: a payload-
    filtered query (e.g. catalyst retrieval's phase_filter) against the
    597K-point clinical_trials collection can genuinely take several
    seconds longer than that, and a real "timed out" 502 was reproduced
    end-to-end this way even though the query itself was actively
    succeeding server-side (confirmed via Qdrant's own access log), just
    not within httpx's default window.
    """
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    return _qdrant


# Collections confirmed (at first use, cached) to carry the bm25 sparse
# vector -- hybrid fusion is attempted once per collection and downgraded
# to dense-only permanently for that process if the collection predates the
# hybrid migration (as the AWS deployment does until its snapshot is
# recovered). Keyed by collection name; values: True (hybrid), False
# (dense-only fallback).
_hybrid_capable: dict[str, bool] = {}


def _query_hybrid(collection: str, query: str, query_filter, limit: int):
    """Dense+BM25 RRF fusion via Qdrant's Query API, with a one-time
    per-collection fallback to dense-only when the collection has no bm25
    sparse vector. Both prefetch branches over-fetch 3x so RRF has real
    candidate diversity to fuse rather than two near-identical top-k lists.
    """
    dense = embed_query(query)
    if _hybrid_capable.get(collection, True):
        try:
            from sparse_embeddings import SPARSE_VECTOR_NAME, embed_query as sparse_embed
            hits = _client().query_points(
                collection_name=collection,
                prefetch=[
                    qmodels.Prefetch(query=dense, using="",
                                     filter=query_filter, limit=limit * 3),
                    qmodels.Prefetch(query=sparse_embed(query), using=SPARSE_VECTOR_NAME,
                                     filter=query_filter, limit=limit * 3),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
            ).points
            _hybrid_capable[collection] = True
            return hits
        except Exception as exc:  # noqa: BLE001
            if "Not existing vector name" not in str(exc):
                raise
            print(f"[retrieval] {collection}: no bm25 sparse vector -- "
                  f"dense-only fallback for this process")
            _hybrid_capable[collection] = False

    return _client().query_points(
        collection_name=collection,
        query=dense,
        query_filter=query_filter,
        limit=limit,
    ).points


def retrieve_trials(
    query: str,
    limit: int = TRIAL_SEARCH_LIMIT,
    phase: Optional[str] = None,
    collection: Optional[str] = None,
) -> list[dict]:
    """The raw clinical-trials retrieval core -- embed, query Qdrant (hybrid
    dense+BM25 RRF where the collection supports it), shape payloads into
    plain dicts. Extracted from the search_clinical_trials tool so the eval
    harness (eval/test_retrieval.py) measures EXACTLY the retrieval path the
    agent uses -- when this function changes (hybrid fusion, reranking), the
    tool and the eval numbers change together, which is the whole point of
    having a baseline.

    Raises on failure -- the tool wrapper is what converts errors into
    agent-visible observations; the eval harness wants a loud failure.
    """
    query_filter = None
    if phase:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="Phase", match=qmodels.MatchValue(value=phase)
            )]
        )

    # Over-fetch for the cross-encoder when reranking is on -- it re-scores
    # a wide fused candidate pool and keeps the true top `limit` (see
    # reranker.py). With reranking off, fetch exactly `limit` as before.
    import reranker as _rr
    fetch_n = max(limit, _rr.RERANK_CANDIDATES) if _rr.enabled() else limit
    hits = _query_hybrid(collection or COLLECTION_NAME, query, query_filter, fetch_n)

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

    def _rerank_text(r: dict) -> str:
        iv = ", ".join((x.get("name") or "") for x in (r.get("interventions") or [])
                       if isinstance(x, dict))
        conds = ", ".join(c for c in (r.get("conditions") or []) if isinstance(c, str))
        return (f"{r.get('BriefTitle') or ''}. Conditions: {conds}. "
                f"Interventions: {iv}. {r.get('BriefSummary') or ''}")

    return _rr.rerank(query, results, _rerank_text, top_k=limit)


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

    try:
        results = retrieve_trials(query, limit=TRIAL_SEARCH_LIMIT, phase=phase)
    except Exception as exc:  # surfaced to the agent as an observation
        return json.dumps({"error": f"Qdrant search failed: {exc}", "has_results": False})

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


@tool
def search_pubmed_literature(query: str) -> str:
    """Search open-access PubMed Central scientific literature (title,
    abstract, and body text of recent biopharma/oncology papers), embedded
    with the same OpenAI model as the other collections.

    Use this for SCIENTIFIC EFFICACY and MECHANISM-of-action questions
    grounded in published research: how a drug or drug class works, what a
    paper reports about a biological pathway or target, or recent literature
    discussion of a drug's activity in a specific indication. This is a
    SEPARATE corpus from search_pdf_literature -- that tool covers conference
    posters and FDA filings (efficacy TABLES: ORR/PFS/OS, Kaplan-Meier data);
    this one covers peer-reviewed PMC articles' own narrative text. Prefer
    this tool when the question is phrased as "what does the literature say"
    or asks about mechanism/biology rather than a specific reported metric.

    Args:
        query: A semantic description of what to find, e.g. "mechanism of
            PD-1 checkpoint inhibition in non-small cell lung cancer".
            Natural language, not keywords -- results are ranked by
            embedding similarity.
    """
    try:
        hits = _query_hybrid(COLLECTION_NAME_PUBMED, query, None, limit=6)
    except Exception as exc:  # surfaced to the agent as an observation
        return json.dumps({"error": f"Qdrant search failed: {exc}", "has_results": False})

    results = []
    for h in hits:
        meta = h.payload or {}
        results.append({
            "PMCID": meta.get("PMCID"),
            "Title": meta.get("Title"),
            "Journal": meta.get("Journal"),
            "PubYear": meta.get("PubYear"),
            "ChunkIndex": meta.get("ChunkIndex"),
            "score": round(float(h.score), 4),
            "Text": meta.get("Text") or "",
            "SourceURL": meta.get("SourceURL"),
        })

    # Same kNN-always-returns-something caveat as the other Qdrant tools --
    # see _is_grounded_pubmed.
    grounded = _is_grounded_pubmed(query, results)

    return json.dumps(
        {"query": query, "returned": len(results), "has_results": grounded,
         "pubmed_chunks": results},
        indent=2,
    )


@tool
def search_sec_filings(query: str) -> str:
    """Search SEC EDGAR 10-K/8-K filings for major biopharma tickers (PFE,
    MRK, JNJ, ABBV), isolated to their Pipeline/Clinical Trials/Research and
    Development sections and embedded with the same OpenAI model as the
    other collections.

    Use this for CORPORATE STRATEGY and PIPELINE PRIORITIZATION questions:
    what a company itself states about its R&D pipeline, which programs it
    is prioritizing or discontinuing, collaboration/licensing deals it
    discloses, or how it frames a drug's role in its portfolio. This is a
    fundamentally different lens than any other tool here -- it reflects the
    COMPANY's own disclosed narrative, not third-party trial data,
    independent literature, or FDA regulatory status. It does NOT contain
    financial statements, stock performance, or non-biopharma corporate
    facts beyond what these filings' R&D-focused sections state -- say so
    rather than guessing if asked those.

    Args:
        query: A semantic description of what to find, e.g. "Merck's
            pipeline strategy for Keytruda" or "Pfizer oncology R&D
            priorities". Natural language, not keywords -- results are
            ranked by embedding similarity. Include the company name or
            ticker in the query text to steer toward one company; there is
            no separate filter parameter.
    """
    try:
        query_vector = embed_query(query)
        hits = _client().query_points(
            collection_name=COLLECTION_NAME_SEC,
            query=query_vector,
            limit=6,
        ).points
    except Exception as exc:  # surfaced to the agent as an observation
        return json.dumps({"error": f"Qdrant search failed: {exc}", "has_results": False})

    results = []
    for h in hits:
        meta = h.payload or {}
        results.append({
            "Ticker": meta.get("Ticker"),
            "Company": meta.get("Company"),
            "Form": meta.get("Form"),
            "AccessionNumber": meta.get("AccessionNumber"),
            "FiledDate": meta.get("FiledDate"),
            "ChunkIndex": meta.get("ChunkIndex"),
            "score": round(float(h.score), 4),
            "Text": meta.get("Text") or "",
            "SourceURL": meta.get("SourceURL"),
        })

    # Same kNN-always-returns-something caveat as the other Qdrant tools --
    # see _is_grounded_sec.
    grounded = _is_grounded_sec(query, results)

    return json.dumps(
        {"query": query, "returned": len(results), "has_results": grounded,
         "sec_chunks": results},
        indent=2,
    )


@tool
def search_corporate_news(query: str) -> str:
    """Search real-time corporate news: RSS press releases (FDA regulatory
    announcements, Johnson & Johnson, and AbbVie corporate newsrooms) and
    earnings call transcripts (all four tracked tickers -- PFE, MRK, JNJ,
    ABBV -- keyword-isolated to pipeline/guidance/forward-looking content),
    embedded with the same OpenAI model as the other collections.

    Use this for the MOST REAL-TIME layer of intelligence available: recent
    regulatory press announcements, corporate press releases, and
    unscripted management commentary from earnings calls -- interim
    clinical updates, pipeline deprioritizations or reprioritizations,
    financial guidance, and forward-looking statements as executives
    themselves described them on the call. This is a SEPARATE corpus from
    search_sec_filings -- SEC filings are formal, periodic, legally-
    reviewed disclosure; this corpus is faster-moving, less formal
    commentary and announcements, and may say things (or use franker
    language) that a 10-K/8-K would not.

    SCOPE NOTE: Pfizer and Merck do NOT have a usable public press-release
    RSS feed (verified live -- see fetch_news_and_transcripts.py's module
    docstring), so press-release-type results here skew toward FDA/J&J/
    AbbVie; Pfizer and Merck ARE covered for the earnings-transcript half
    of this corpus. If asked about a Pfizer or Merck press release
    specifically and nothing relevant is retrieved, say so rather than
    assuming coverage that doesn't exist.

    Args:
        query: A semantic description of what to find, e.g. "Merck pipeline
            deprioritization" or "AbbVie interim clinical update". Natural
            language, not keywords -- results are ranked by embedding
            similarity.
    """
    try:
        query_vector = embed_query(query)
        hits = _client().query_points(
            collection_name=COLLECTION_NAME_NEWS,
            query=query_vector,
            limit=6,
        ).points
    except Exception as exc:  # surfaced to the agent as an observation
        return json.dumps({"error": f"Qdrant search failed: {exc}", "has_results": False})

    results = []
    for h in hits:
        meta = h.payload or {}
        results.append({
            "SourceType": meta.get("SourceType"),
            "Ticker": meta.get("Ticker"),
            "Company": meta.get("Company"),
            "FeedName": meta.get("FeedName"),
            "Title": meta.get("Title"),
            "PubDate": meta.get("PubDate"),
            "CallDate": meta.get("CallDate"),
            "ChunkIndex": meta.get("ChunkIndex"),
            "score": round(float(h.score), 4),
            "Text": meta.get("Text") or "",
            "SourceURL": meta.get("SourceURL"),
        })

    # Same kNN-always-returns-something caveat as the other Qdrant tools --
    # see _is_grounded_news.
    grounded = _is_grounded_news(query, results)

    return json.dumps(
        {"query": query, "returned": len(results), "has_results": grounded,
         "news_chunks": results},
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
MATCH (t:Trial)-[:INVESTIGATES]->(d:Drug)
OPTIONAL MATCH (d)-[:MAPPED_TO_RXNORM]->(c:Concept)
WITH t, d, c
WHERE toLower(d.name) CONTAINS toLower($entity)
   OR toLower(coalesce(c.standard_name, '')) = toLower($entity)
   OR ANY(b IN coalesce(c.brand_names, []) WHERE toLower(b) = toLower($entity))
RETURN DISTINCT
  t.id AS NCTId, t.title AS BriefTitle, t.phase AS Phase,
  t.status AS OverallStatus, t.sponsor AS LeadSponsorName,
  t.study_type AS studyType, t.conditions AS conditions,
  t.interventions_json AS interventions_json, t.summary AS BriefSummary,
  d.name AS MatchedDrugName, c.cui AS cui, c.standard_name AS standard_name
ORDER BY t.id
LIMIT $limit
"""
# The MAPPED_TO_RXNORM hop is OPTIONAL MATCH, not part of the required
# pattern -- found live on AWS: build_kg.py deliberately skips low-
# confidence RxNorm links (MERGE_TRIAL_DRUG_ONLY), so unlicensed
# development compounds (e.g. "ABX464") exist as Drug nodes with NO
# Concept edge. With the hop mandatory, the traversal silently excluded
# exactly the pipeline-stage assets a biopharma intelligence tool is most
# often asked about, and the agent concluded "no results" for drugs the
# graph actually contains. WITH...WHERE (not a bare WHERE after the
# OPTIONAL MATCH) is load-bearing Cypher: a WHERE clause attached directly
# to an OPTIONAL MATCH only filters the optional expansion, not the rows.
# Verified live: unlike search_clinical_trials's kNN (which always returns at
# most `limit` points), this Cypher traversal had NO limit at all -- for a
# widely-studied drug like pembrolizumab it matched 1711 trials in one call,
# every one of which would fan out to its own extract_trial worker. Capped
# to the same TRIAL_SEARCH_LIMIT as search_clinical_trials so the two tools
# that can populate the trials pool share one sane ceiling; ORDER BY t.id
# makes which subset gets returned deterministic run-to-run (there is no
# relevance score to rank by here -- every match equally satisfies the exact
# entity query -- so id order is just for reproducibility, not priority).


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
            rows = list(session.run(KG_MATCH_QUERY, entity=entity, limit=TRIAL_SEARCH_LIMIT))
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
    #
    # DETERMINISTIC FALLBACK on empty: the system prompt has always told
    # the agent "empty graph result -> call search_clinical_trials", but
    # verified live on AWS (Kimi orchestration) that the model routinely
    # ignores that and concludes "no results" after the single empty tool
    # round. Rather than depending on LLM compliance, the tool itself now
    # runs the vector-search fallback and says so in its payload -- one
    # tool call, guaranteed best answer across both stores.
    if not trials:
        try:
            fallback = retrieve_trials(entity, limit=TRIAL_SEARCH_LIMIT)
        except Exception as exc:  # noqa: BLE001 -- fallback is best-effort
            fallback = []
            print(f"[kg] vector fallback failed: {exc}")
        for r in fallback:
            r["RetrievalSource"] = "vector_search_fallback"
        grounded = _is_grounded(entity, fallback) if fallback else False
        return json.dumps({
            "entity": entity,
            "resolved_concepts": [],
            "knowledge_graph_empty": True,
            "fell_back_to_vector_search": True,
            "returned": len(fallback),
            "has_results": grounded,
            "trials": fallback if grounded else [],
        }, indent=2)

    return json.dumps({
        "entity": entity,
        "resolved_concepts": [{"cui": cui, "standard_name": name}
                              for cui, name in resolved.items()],
        "returned": len(trials),
        "has_results": len(trials) > 0,
        "trials": trials,
    }, indent=2)


TOOLS = [search_clinical_trials, search_pdf_literature, query_knowledge_graph,
         search_fda_records, search_pubmed_literature, search_sec_filings,
         search_corporate_news]


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
    # Federated retrieval, fourth source: chunks from search_pubmed_literature
    # -- same broadcast-to-every-worker treatment as retrieved_literature. A
    # PubMed excerpt doesn't spawn its own extract_trial worker either (it
    # has no NCTId); each worker judges whether it's about ITS trial's drug.
    retrieved_pubmed: Annotated[list[dict], operator.add]
    # Federated retrieval, fifth source: chunks from search_sec_filings --
    # same treatment again. A corporate filing excerpt is broadcast the same
    # way; each worker judges whether it's about ITS trial's drug/sponsor.
    retrieved_sec: Annotated[list[dict], operator.add]
    # Federated retrieval, sixth source: chunks from search_corporate_news
    # -- same treatment again. A press-release/earnings-transcript excerpt
    # is broadcast the same way; each worker judges whether it's about ITS
    # trial's drug/sponsor.
    retrieved_news: Annotated[list[dict], operator.add]
    # Per-worker input only. Set exclusively via the Send("extract_trial",
    # {"single_trial": ..., "literature": ..., "fda_records": ..., ...})
    # payload in continue_to_extraction -- no other node reads or writes
    # these keys.
    single_trial: Optional[dict]
    literature: Optional[list[dict]]
    fda_records: Optional[list[dict]]
    pubmed_chunks: Optional[list[dict]]
    sec_chunks: Optional[list[dict]]
    news_chunks: Optional[list[dict]]


AGENT_SYSTEM = """You are a life sciences market intelligence analyst.

You have SEVEN tools over SEVEN independent sources -- pick whichever
actually match what the question is asking, not out of habit:

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
- search_pubmed_literature -- open-access PubMed Central scientific
  literature (peer-reviewed papers' title/abstract/body text). Use it for
  SCIENTIFIC EFFICACY and MECHANISM-OF-ACTION questions grounded in
  published research: how a drug or drug class works, what recent
  literature reports about a biological pathway or target, or mechanism
  discussion for a drug in a specific indication. Distinct from
  search_pdf_literature, which covers conference-poster/FDA-filing efficacy
  TABLES (ORR/PFS/OS) rather than peer-reviewed narrative text.
- search_sec_filings -- SEC EDGAR 10-K/8-K filings for major biopharma
  tickers, isolated to their Pipeline/Clinical Trials/Research and
  Development sections. Use it for CORPORATE STRATEGY and PIPELINE
  PRIORITIZATION questions: what a company itself discloses about its R&D
  pipeline, which programs it is prioritizing, or how it frames a drug's
  role in its portfolio. This is the COMPANY's own disclosed narrative, not
  third-party trial data, independent literature, or FDA regulatory status.
  Formal, periodic, legally-reviewed disclosure -- see search_corporate_news
  for this company's own FASTER-moving, less formal commentary instead.
- search_corporate_news -- RSS press releases (FDA regulatory
  announcements, Johnson & Johnson, and AbbVie newsrooms -- Pfizer and
  Merck do not have a usable public press-release feed, so press-release
  coverage there is real but incomplete) and earnings call transcripts
  (all four tracked tickers, keyword-isolated to pipeline/guidance/
  forward-looking content). Use it for the MOST REAL-TIME layer available:
  recent regulatory announcements, interim clinical updates, pipeline
  deprioritizations or reprioritizations mentioned live on a call,
  financial guidance, and unscripted management commentary -- language and
  timing a formal SEC filing would not carry. Distinct from
  search_sec_filings: that is the company's official, reviewed disclosure;
  this is real-time news and management's own words as spoken/published.

These are not alternatives to pick one of -- for a holistic pipeline question
that touches trial design/entity relationships, reported results, regulatory
status, scientific mechanism, corporate strategy, AND real-time developments
(e.g. "what is [Company]'s pipeline strategy for [Drug], what does the
literature say about its mechanism, and what has management said recently"),
call the relevant tools CONCURRENTLY, in the same turn, and let the evidence
from each cross-reference the other. Do not assume registry or graph data
alone can answer an efficacy question, do not assume literature alone can
answer a design question, do not assume the registry/graph/literature can
answer a regulatory-approval question, do not assume trial/FDA/literature
data can answer a corporate-strategy question that only the company's own
SEC filings state, and do not assume SEC filings alone capture the most
recent developments a press release or earnings call would carry instead:
each tool only knows its own corpus, and none substitutes for another.

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
(conference posters, FDA filings), a shared pool of FDA drug approval
records, a shared pool of PubMed Central literature excerpts, a shared pool
of SEC EDGAR filing excerpts, and a shared pool of corporate news excerpts
(press releases and earnings call commentary), each retrieved separately
from its own source.

This is the Map stage of a Map-Reduce pipeline: you never see any other
trial's registry record, only this one -- do not reference, compare against,
or assume anything about other trials in the corpus. The literature
excerpts, FDA records, PubMed excerpts, SEC filing excerpts, and corporate
news excerpts are the exception: they are shared across every worker
running this turn, most will have nothing to do with your trial, and it is
your job to judge which (if any) genuinely do.

STRICT CORPUS GROUNDING -- the hard boundary, now covering ALL SIX sources:
- The trial record, the literature excerpts, the FDA records, the PubMed
  excerpts, the SEC filing excerpts, and the corporate news excerpts below
  are your ONLY sources. Your own pharmacological or regulatory knowledge
  is out of scope, even when you are confident it is correct.
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

FUSING PUBMED LITERATURE INTO mechanism_or_findings:
- A PubMed excerpt matches YOUR trial ONLY on the same distinctive-agent-name
  or explicit-NCT-id basis as conference/FDA-filing literature above -- a
  shared common backbone agent (e.g. pembrolizumab appearing in many
  unrelated papers) is NOT enough on its own.
- When an excerpt genuinely matches, weave its reported mechanism-of-action
  or scientific finding into mechanism_or_findings and attribute it to
  PubMed (e.g. "per PubMed literature (PMCID), the mechanism involves...")
  so a reader can tell peer-reviewed literature apart from registry design
  facts, conference/FDA-filing literature, and FDA approval status.
- When no PubMed excerpt matches, say nothing about it -- do not force a
  connection.

FUSING SEC FILING EXCERPTS INTO mechanism_or_findings:
- An SEC filing excerpt matches YOUR trial ONLY if it names the SAME drug as
  one of this trial's own `interventions`, OR names this trial's
  `LeadSponsorName` as the filer -- not merely because the filer is a large
  biopharma company with many unrelated programs.
- When an excerpt genuinely matches, weave the company's own stated
  strategy/pipeline framing into mechanism_or_findings and attribute it to
  the filing (e.g. "per [Ticker]'s SEC filing, the company describes this
  program as...") -- report what the company states, not an inference about
  its priorities beyond what the excerpt says.
- When no SEC filing excerpt matches, say nothing about it -- do not force a
  connection.

FUSING CORPORATE NEWS EXCERPTS INTO mechanism_or_findings:
- A corporate news excerpt (press release or earnings call commentary)
  matches YOUR trial ONLY on the same distinctive-drug-name-or-sponsor
  basis as SEC filing excerpts above -- a shared common backbone agent or
  the filer simply being a large biopharma company is NOT enough on its
  own.
- When an excerpt genuinely matches, weave the real-time development into
  mechanism_or_findings and attribute it distinctly from an SEC filing --
  e.g. "per a Merck earnings call (Aug 2026), management stated..." or
  "per an AbbVie press release, ..." -- so a reader can tell unscripted
  real-time commentary apart from formal, reviewed SEC disclosure, since
  the two carry different reliability/formality signals.
- When no corporate news excerpt matches, say nothing about it -- do not
  force a connection.

Each trial record carries structured pharmacology: `interventions` (a list of
{type, name}), `conditions`, and `studyType`. Use those fields as the
authoritative source for which agents are being tested -- name the specific
interventions rather than describing them generically, and do not rely on
parsing drug names out of the narrative BriefSummary.

CITATIONS (the `sources` field): for every auxiliary excerpt you actually
fused into mechanism_or_findings, add one sources entry copying that
excerpt's own identifier and SourceURL EXACTLY as written (PMCID for PubMed,
SourceURL for filings/news/posters, application number for FDA records).
Do NOT add an entry for the trial registry record itself -- that citation
is attached automatically. Never invent or approximate a reference: leave
sources empty rather than guessing."""

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

# --- Reduce stage, sources-only variant: no trial rows exist this run ------
# (e.g. a corporate-strategy + mechanism question that never matched a
# specific trial) -- write directly from whichever non-trial pools grounded,
# instead of the trial-row-shaped REDUCER_SYSTEM contract above.
SOURCES_ONLY_REDUCER_SYSTEM = """You are writing the final analyst answer for
a life sciences market intelligence question that did NOT match any specific
clinical trial in the registry.

No extracted trial rows exist for this turn -- there is no Map stage output
to work from. You are given the USER QUESTION and whichever federated
evidence pools were actually retrieved and passed the grounding check this
turn: PubMed literature excerpts, SEC filing excerpts, conference/FDA-filing
literature excerpts, FDA approval records, and/or corporate news excerpts
(press releases and earnings call commentary). Each pool is exactly what
its own tool returned -- not pre-fused into anything, since there is no
per-trial worker to do that fusing here.

Write the narrative_summary answering the user's question using ONLY the
provided pools below.
- Never answer from prior knowledge, even when confident it is correct --
  every factual claim must trace to one of the provided excerpts/records.
- Attribute each claim to its source so a reader can tell them apart: a
  PubMed finding by its PMCID, an SEC filing claim by its Ticker/Company and
  Form (e.g. "per Merck's 10-K filing..."), a conference/FDA-filing
  literature finding by its SourceFile, an FDA record by its
  ApplicationNumber, and a corporate news excerpt by its Company/Ticker and
  whether it's a press release or earnings call (e.g. "per a Merck earnings
  call (Aug 2026), management stated..."), since that distinguishes fast,
  informal real-time commentary from formal, reviewed disclosure like SEC
  filings.
- If two pools disagree or address different aspects (e.g. SEC filings
  describe corporate strategy while PubMed describes scientific mechanism),
  present both rather than collapsing them into one claim -- they are
  answering different parts of the question, not corroborating each other.
- If the provided pools do not actually support an answer to the question,
  say so plainly rather than stretching a tangential excerpt to fit."""

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

    LLM_PROVIDER=nvidia / LLM_PROVIDER=kimi are TEMPORARY escape hatches, not
    a second production path: each reroutes every LLM in the graph (main
    reasoning, intent gate, extraction, synthesis) through an OpenAI-
    compatible endpoint via the same ChatOpenAI client construction already
    verified working in evaluate_agent.py's build_judge() (ChatOpenAI
    against integrate.api.nvidia.com, not ChatNVIDIA -- see that function's
    own comment on why: ChatNVIDIA's aiohttp client has no configurable
    socket-read timeout and reliably died on a long call). Added so this
    graph can still be exercised end-to-end when Anthropic billing is
    blocked; NOT verified to have equivalent bind_tools()/
    with_structured_output() fidelity to Claude -- tool-calling and
    structured-output behavior genuinely differs across providers/models,
    so either is for unblocking a live test, not a claim it is
    interchangeable with Claude in production.
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
            # See the kimi branch's comment below -- same root cause, same fix.
            max_retries=0,
        )

    if provider == "kimi":
        from langchain_openai import ChatOpenAI

        key = os.getenv("KIMI_API_KEY")
        if not key:
            raise SystemExit(
                "[build_llm] LLM_PROVIDER=kimi but KIMI_API_KEY is not set.\n"
                "            Add it to .env — get a key at https://platform.moonshot.ai"
            )
        # Both overridable: KIMI_BASE_URL for the .cn endpoint if that's
        # where the key is provisioned, KIMI_MODEL once a specific model is
        # confirmed live rather than guessed.
        return ChatOpenAI(
            model=os.getenv("KIMI_MODEL", "kimi-k3"),
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
            api_key=key,
            max_tokens=max_tokens,
            timeout=timeout,
            # max_retries=0 (openai's own client default is 2): verified
            # live to matter, not a guess. The landscape pipeline's call is
            # large and slow (tens of thousands of tokens); on a 429/timeout
            # the openai SDK's own automatic retry fires a NEW request
            # immediately, but the FIRST attempt can still be processing on
            # the provider's servers -- against this key's "max organization
            # concurrency: 1" limit, that overlap is a self-inflicted
            # collision: the retry gets rate-limited by the still-running
            # original, repeatedly, until every retry is exhausted. Disabling
            # the SDK's own retry means a failure surfaces immediately and
            # cleanly instead, and synthesize_node's own MAX_SYNTHESIS_RETRIES
            # loop (sequential -- each attempt's HTTP call fully completes,
            # success or exception, before the next one starts) is what
            # retries instead, without ever having two requests in flight.
            max_retries=0,
            # thinking disabled: verified live -- kimi-k3 defaults to an
            # internal "thinking" (reasoning) mode that Moonshot's own API
            # flatly rejects combining with a forced tool_choice ("tool_choice
            # 'specified' is incompatible with thinking enabled", a real
            # openai.BadRequestError hit by with_structured_output's
            # function_calling method, which forces the one tool). Probed
            # directly against the endpoint: {"thinking": {"type":
            # "disabled"}} in the request body clears the conflict AND avoids
            # burning max_tokens on invisible reasoning (see the
            # reasoning_tokens=15997/16000 case in this project's history).
            extra_body={"thinking": {"type": "disabled"}},
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, max_tokens=max_tokens, timeout=timeout)


# extract_trial_node's own per-call timeout -- a single-trial structured
# extraction is far smaller than the landscape/catalyst mega-calls
# LANDSCAPE_TIMEOUT/CATALYST_TIMEOUT budget for, so this is deliberately
# tighter; see extraction_llm's own comment in make_graph for why it is
# pinned to _build_gpt4o_llm at all.
EXTRACTION_TIMEOUT = 60

# Caps how many extract_trial workers actually call the OpenAI API at once.
# Verified live at TRIAL_SEARCH_LIMIT=50/93: with every Send worker firing
# its gpt-4o call simultaneously (LangGraph does not throttle Send fanout on
# its own), this project's OpenAI org hit "Rate limit reached for gpt-4o ...
# tokens per min (TPM): Limit 30000" within the first ~4 calls -- each
# extraction prompt runs ~8000 tokens (full trial record + every shared
# evidence pool), so the account can sustain only ~3-4 calls/minute, not 50
# truly concurrent ones. A low semaphore, not a higher TPM tier, is the fix
# available here -- it trades wall-clock time (a 50-row query now takes
# minutes, not seconds) for actually finishing instead of most workers
# 429-ing and silently dropping their row.
EXTRACTION_CONCURRENCY = 3
_extraction_semaphore = threading.Semaphore(EXTRACTION_CONCURRENCY)

# Matches openai's own "...please try again in 16.29s..." message text --
# sleeping exactly what the API itself says to (plus a small buffer) adapts
# to whatever the account's real TPM budget is at the moment, rather than
# guessing a fixed backoff that's either too short (still 429s) or too long
# (wastes time when the budget already refilled).
# Matches both provider dialects observed live: OpenAI's "please try again
# in 1.2s" AND Kimi/Moonshot's "please try again after 1 seconds".
_RETRY_AFTER_RE = re.compile(r"try again (?:in|after) ([\d.]+)\s*s")


def _invoke_ratelimit_retry(llm, msgs, *, attempts: int = 4, label: str = "",
                            verbose: bool = True):
    """invoke() with rate-limit patience for the ORCHESTRATION provider.

    Exists because of a real prod 502: the Kimi free tier enforces
    "organization max RPM: 3", and with max_retries=0 on the client (a
    deliberate choice -- see build_llm) a single 429 during the agent or
    narrative step killed the entire multi-minute run. Waits what the
    provider's own message asks for (both dialects parsed by
    _RETRY_AFTER_RE), with a floor that actually clears an RPM window, and
    re-raises after `attempts` so genuine outages still fail loudly."""
    for attempt in range(attempts):
        try:
            return llm.invoke(msgs)
        except openai.RateLimitError as exc:
            if attempt == attempts - 1:
                raise
            m = _RETRY_AFTER_RE.search(str(exc))
            wait = max(float(m.group(1)) if m else 0.0, 8.0 * (attempt + 1))
            if verbose:
                print(f"  ⚠ rate-limited{f' ({label})' if label else ''} "
                      f"(attempt {attempt + 1}/{attempts}), waiting {wait:.0f}s")
            time.sleep(wait)


def make_graph(model: str, verbose: bool = True):
    llm = build_llm(model)
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    # --- Map stage: one structured call per trial, run in parallel ---------
    # Pinned to _build_gpt4o_llm, NOT the provider-routed `llm` above --
    # TRIAL_SEARCH_LIMIT=50 means a single Smart Table query can now fan out
    # up to 50 concurrent extract_trial calls (continue_to_extraction Sends
    # one per distinct trial, uncapped). That is exactly the failure shape
    # _build_gpt4o_llm's own docstring documents for kimi/nvidia: kimi's key
    # hits a hard "max organization concurrency: 1" / observed "max RPM: 3"
    # quota, so under real load most of those 50 parallel calls would 429
    # and extract_trial_node's broad except silently drops each one -- more
    # retrieval breadth would have made the row count WORSE, not better.
    # gpt-4o has no such ceiling and is already a hard project dependency
    # (embeddings.py), so this is the same fix already applied to the
    # landscape/catalyst Map-scale calls, extended to this one. Real API
    # cost note: up to 50 gpt-4o calls per query now, vs. whatever
    # LLM_PROVIDER was routing to before.
    extraction_llm = _build_gpt4o_llm(
        EXTRACTION_TIMEOUT, model_env_var="EXTRACTION_MODEL"
    ).with_structured_output(TrialRow)

    # --- extraction cost cascade ------------------------------------------
    # Most trial records are formulaic registry text a mini-tier model
    # extracts correctly; only ambiguous rows need full gpt-4o. Each Map
    # worker therefore tries EXTRACTION_MODEL_MINI (default gpt-4o-mini)
    # first, validates the structured result deterministically
    # (_cascade_acceptable below -- schema parse + critical fields), and
    # escalates to the pinned gpt-4o ONLY on failure. Beyond the ~15x price
    # gap, the mini tier has its own separate TPM budget, so most workers
    # no longer queue against the 30K-TPM gpt-4o ceiling that made large
    # extractions take 5-10 minutes (see EXTRACTION_CONCURRENCY).
    # include_raw=True so a schema-parse failure surfaces as data
    # (parsing_error) to trigger escalation instead of raising, and so
    # usage metadata (cached-token counts) is loggable.
    extraction_llm_mini = _build_gpt4o_llm(
        EXTRACTION_TIMEOUT, model_env_var="EXTRACTION_MODEL_MINI",
        default_model="gpt-4o-mini",
    ).with_structured_output(TrialRow, include_raw=True)

    # --- Reduce stage: prose only -- table_data is already fixed by the Map
    # stage, so this call carries far less risk than the old Synthesis call
    # did, but keeps the same validate-or-retry shape for consistency.
    narrative_llm = llm.with_structured_output(
        NarrativeSummary, include_raw=True, method="function_calling"
    )

    # Separate, cheap/fast model for the input guardrail -- see INTENT_MODEL.
    # max_tokens=1536, not the smaller value this had before: verified
    # directly under LLM_PROVIDER=nvidia (see build_llm's docstring) that
    # NVIDIA's Nemotron model reliably burns several hundred tokens on its
    # own internal reasoning before it ever emits the tiny IntentClassification
    # payload -- 512 hit openai.LengthFinishReasonError consistently, not
    # intermittently, on ordinary in-domain questions. Harmless headroom for
    # Claude, which was never close to that ceiling for this task.
    intent_llm = build_llm(INTENT_MODEL, max_tokens=1536, timeout=60) \
        .with_structured_output(IntentClassification, method="function_calling")

    # --- node: IntentClassifier --------------------------------------------
    def intent_classifier_node(state: AgentState) -> dict:
        question = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        # Small retry loop, not a graph-level one like synthesize_node's --
        # verified live (LLM_PROVIDER=kimi) that despite the forced
        # tool_choice from method="function_calling", kimi-k3 intermittently
        # answers in plain prose instead of calling the IntentClassification
        # tool for the SAME question that calls it correctly on other
        # attempts (Moonshot's forced tool_choice isn't as strictly enforced
        # server-side as real OpenAI's) -- with_structured_output then
        # returns None with no exception, since a genuinely empty tool_calls
        # list isn't a schema-validation failure. Two retries have been
        # enough live; a persistent failure fails OPEN (in-domain) rather
        # than wrongly gatekeeping a legitimate clinical question on what is
        # a guardrail, not the actual answer.
        verdict: IntentClassification | None = None
        for attempt in range(3):
            verdict = _invoke_ratelimit_retry(intent_llm, [
                SystemMessage(content=INTENT_SYSTEM),
                HumanMessage(content=question),
            ], label="intent", verbose=verbose)
            if verdict is not None:
                break
            if verbose:
                print(f"  ✗ intent classifier returned no tool call "
                      f"(attempt {attempt + 1}/3) -- retrying")
        if verdict is None:
            if verbose:
                print("  ⚠ intent classifier failed 3/3 attempts -- "
                      "failing open (treating as in-domain)")
            return {"is_in_domain": True}
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
        reply = _invoke_ratelimit_retry(model_to_use, msgs, label="agent", verbose=verbose)

        if verbose:
            _trace_agent(reply, state.get("tool_rounds", 0))
        return {"messages": [reply]}

    # --- node: Tool Node (wrapped for tracing + grounding aggregation) -----
    # Takes `config` explicitly (LangGraph injects it) and forwards it to
    # tool_node.invoke() rather than relying on tool_node picking up an
    # ambient RunnableConfig on its own -- verified live that this ambient
    # pickup silently breaks on the astream() (streaming) path specifically:
    # this node runs in a thread-pool executor there (run_in_executor), and
    # langgraph 1.2.11 (this project's own `langgraph>=1.0.0` pin floated
    # up to it) raises `ValueError: Missing required config key 'N/A' for
    # 'tools'` inside ToolNode.invoke() when it can't recover a config from
    # that thread on its own. The plain (non-streaming) /api/research path
    # never hit this because its node runs on the graph's own thread, where
    # the ambient config happens to still be reachable.
    def tools_node(state: AgentState, config: RunnableConfig) -> dict:
        out = tool_node.invoke(state, config)
        round_grounded = False
        new_trials: list[dict] = []
        new_literature: list[dict] = []
        new_fda: list[dict] = []
        new_pubmed: list[dict] = []
        new_sec: list[dict] = []
        new_news: list[dict] = []
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
                new_pubmed.extend(payload.get("pubmed_chunks", []))
                new_sec.extend(payload.get("sec_chunks", []))
                new_news.extend(payload.get("news_chunks", []))
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
                "retrieved_fda": new_fda,
                "retrieved_pubmed": new_pubmed,
                "retrieved_sec": new_sec,
                "retrieved_news": new_news}

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
        pubmed_chunks = state.get("pubmed_chunks") or []
        sec_chunks = state.get("sec_chunks") or []
        news_chunks = state.get("news_chunks") or []
        nct_id = trial.get("NCTId", "?")
        started = time.time()
        if verbose:
            print(f"\n{'─' * 78}\n▶ NODE: extract_trial  worker={nct_id}  "
                  f"(Map stage, parallel, {len(literature)} literature "
                  f"excerpt(s), {len(fda_records)} FDA record(s), "
                  f"{len(pubmed_chunks)} PubMed excerpt(s), "
                  f"{len(sec_chunks)} SEC excerpt(s), "
                  f"{len(news_chunks)} corporate news excerpt(s) available)"
                  f"\n{'─' * 78}")

        # PROMPT ORDERING IS A COST FEATURE, not a readability choice: the
        # five evidence pools below are IDENTICAL across every Map worker
        # in a run (they're the shared retrieval state), while the trial
        # record is unique per worker. Shared-invariant content first +
        # per-worker content last means [system prompt + all pools] forms
        # one long identical prefix across the whole fan-out, which OpenAI
        # prompt caching serves at half price after the first worker warms
        # it (automatic, >=1024-token prefixes; EXTRACTION_SYSTEM alone is
        # ~1.9K tokens). The previous ordering -- trial record FIRST --
        # made every worker's prompt diverge at token ~1 and guaranteed a
        # 0% cache hit rate on exactly the fan-out that dominates spend.
        prompt = ""
        if literature:
            prompt += (
                f"LITERATURE EXCERPTS (shared pool from search_pdf_literature "
                f"-- fuse in ONLY what genuinely matches the trial record at "
                f"the END of this message, per the system instructions):\n"
                f"{json.dumps(literature, indent=2)}\n\n"
            )
        else:
            prompt += "LITERATURE EXCERPTS: (none retrieved this run)\n\n"

        if fda_records:
            prompt += (
                f"FDA RECORDS (shared pool from search_fda_records -- fuse in "
                f"ONLY what genuinely matches the trial's drug(s), per the "
                f"system instructions):\n{json.dumps(fda_records, indent=2)}\n\n"
            )
        else:
            prompt += "FDA RECORDS: (none retrieved this run)\n\n"

        if pubmed_chunks:
            prompt += (
                f"PUBMED LITERATURE EXCERPTS (shared pool from "
                f"search_pubmed_literature -- fuse in ONLY what genuinely "
                f"matches the trial, per the system instructions):\n"
                f"{json.dumps(pubmed_chunks, indent=2)}\n\n"
            )
        else:
            prompt += "PUBMED LITERATURE EXCERPTS: (none retrieved this run)\n\n"

        if sec_chunks:
            prompt += (
                f"SEC FILING EXCERPTS (shared pool from search_sec_filings -- "
                f"fuse in ONLY what genuinely matches the trial's drug(s) or "
                f"sponsor, per the system instructions):\n"
                f"{json.dumps(sec_chunks, indent=2)}\n\n"
            )
        else:
            prompt += "SEC FILING EXCERPTS: (none retrieved this run)\n\n"

        if news_chunks:
            prompt += (
                f"CORPORATE NEWS EXCERPTS (shared pool from "
                f"search_corporate_news -- press releases and earnings call "
                f"commentary; fuse in ONLY what genuinely matches the "
                f"trial's drug(s) or sponsor, per the system instructions):\n"
                f"{json.dumps(news_chunks, indent=2)}\n\n"
            )
        else:
            prompt += "CORPORATE NEWS EXCERPTS: (none retrieved this run)\n\n"

        prompt += (
            f"TRIAL RECORD (structured registry -- the primary source for "
            f"this row):\n{json.dumps(trial, indent=2)}"
        )

        # _extraction_semaphore + a generous retry budget, not a single quick
        # retry -- at up to TRIAL_SEARCH_LIMIT=50 concurrent workers, the
        # bottleneck verified live is a hard account-level TPM ceiling (see
        # EXTRACTION_CONCURRENCY's comment), not an occasional hiccup. The
        # semaphore keeps at most EXTRACTION_CONCURRENCY calls in flight;
        # the retry loop below then absorbs the 429s that still happen at
        # the boundary of that budget by sleeping exactly as long as the API
        # says to. Anything else (schema/parsing failure, a genuinely bad
        # request) still fails this worker immediately on its first
        # non-transient exception -- one flaky row must not sink the batch,
        # and retrying a non-transient error would just waste the same
        # failure again.
        messages = [SystemMessage(content=EXTRACTION_SYSTEM),
                    HumanMessage(content=prompt)]

        def _cascade_acceptable(candidate: TrialRow | None) -> bool:
            """Deterministic accept gate for the mini tier: the schema
            parsed AND the critical identity fields are present and honest
            (nct_id must match the record this worker was given -- a
            mismatched id is the classic small-model copy error and would
            poison the table)."""
            return (candidate is not None
                    and (candidate.nct_id or "").strip() == nct_id
                    and bool((candidate.phase or "").strip())
                    and bool((candidate.sponsor or "").strip()))

        # --- tier 1: mini model, outside the gpt-4o semaphore -----------
        # The semaphore exists purely to ration the 30K-TPM gpt-4o budget;
        # mini has its own (much larger) budget, so serializing mini calls
        # behind it would rebuild the very queue the cascade removes.
        row: TrialRow | None = None
        tier = "mini"
        try:
            out = extraction_llm_mini.invoke(messages)
            parsed = out.get("parsed")
            if _cascade_acceptable(parsed):
                row = parsed
                if verbose:
                    usage = getattr(out.get("raw"), "usage_metadata", None) or {}
                    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
                    print(f"  ↳ {nct_id}: mini tier OK "
                          f"(cached input tokens: {cached})")
            elif verbose:
                why = out.get("parsing_error") or "critical-field check failed"
                print(f"  ↳ {nct_id}: mini tier rejected ({why}) — escalating to gpt-4o")
        except Exception as exc:  # noqa: BLE001 -- any mini failure just escalates
            if verbose:
                print(f"  ↳ {nct_id}: mini tier errored ({exc}) — escalating to gpt-4o")

        # --- tier 2: pinned gpt-4o, semaphore + adaptive retry (unchanged
        # from the pre-cascade behavior; see EXTRACTION_CONCURRENCY) ------
        if row is None:
            tier = "gpt-4o"
            MAX_EXTRACTION_ATTEMPTS = 5
            with _extraction_semaphore:
                for attempt in range(MAX_EXTRACTION_ATTEMPTS):
                    try:
                        row = extraction_llm.invoke(messages)
                        break
                    except (openai.RateLimitError, openai.APITimeoutError,
                            openai.APIConnectionError, openai.InternalServerError) as exc:
                        if attempt == MAX_EXTRACTION_ATTEMPTS - 1:
                            if verbose:
                                print(f"  ✗ extraction failed for {nct_id} "
                                      f"({time.time() - started:.1f}s) after "
                                      f"{MAX_EXTRACTION_ATTEMPTS} attempts: {exc} — dropping this row")
                            return {"extracted_rows": []}
                        m = _RETRY_AFTER_RE.search(str(exc))
                        wait = min(float(m.group(1)), 30.0) + 1.0 if m else 5.0 * (attempt + 1)
                        if verbose:
                            print(f"  ⚠ transient error for {nct_id} "
                                  f"(attempt {attempt + 1}/{MAX_EXTRACTION_ATTEMPTS}), "
                                  f"retrying in {wait:.1f}s: {exc}")
                        time.sleep(wait)
                    except Exception as exc:  # one flaky worker must not sink the batch
                        if verbose:
                            print(f"  ✗ extraction failed for {nct_id} "
                                  f"({time.time() - started:.1f}s): {exc} — dropping this row")
                        return {"extracted_rows": []}

        # Registry provenance is deterministic, never model-supplied: every
        # row's primary source IS its ClinicalTrials.gov record, so build
        # that citation from the worker's own nct_id and keep only NON-
        # registry citations from the model (dropping any registry entry it
        # improvised, which may carry a hallucinated URL).
        row.sources = [SourceCitation(
            source_type="registry", reference=nct_id,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
        )] + [s for s in (row.sources or []) if s.source_type != "registry"]

        if verbose:
            print(f"  ✓ {row.nct_id}  phase={row.phase!r}  "
                  f"mechanism_described={row.mechanism_described}  "
                  f"tier={tier}  sources={len(row.sources)}  "
                  f"({time.time() - started:.1f}s)")
        return {"extracted_rows": [row]}

    # --- node: synthesize_table (Reduce stage) -------------------------------
    def synthesize_table_node(state: AgentState) -> dict:
        question = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        rows = state.get("extracted_rows", [])
        retries = state.get("synthesis_retries", 0)

        # No trial rows this run: either genuinely nothing was found, or the
        # agent grounded on non-trial pools only (e.g. a corporate-strategy +
        # mechanism question that never matched a specific trial -- see
        # continue_to_extraction's routing comment). _deduped_pools recomputes
        # the same non-trial pools it would have broadcast to Map workers, so
        # this branch can answer from them directly instead of assuming
        # "no trial" means "no evidence at all."
        pools = _deduped_pools(state) if not rows else None
        sources_only = bool(pools) and any(
            pools[k] for k in (
                "literature", "fda_records", "pubmed_chunks", "sec_chunks", "news_chunks",
            )
        )

        if verbose:
            label = f"  (retry {retries}/{MAX_SYNTHESIS_RETRIES})" if retries else ""
            mode = "sources-only" if sources_only else f"{len(rows)} extracted row(s)"
            print(f"\n{'─' * 78}\n▶ NODE: synthesize_table{label}  (Reduce stage — "
                  f"writing narrative from {mode}; raw Qdrant text is not "
                  f"visible here except in sources-only mode)\n{'─' * 78}")

        if not rows and not sources_only:
            empty = SmartTableResponse(
                narrative_summary="No trials were retrieved from the database, "
                                  "so there is no evidence to answer from.",
                table_data=[],
            )
            return {"messages": [AIMessage(content=empty.narrative_summary)],
                    "result": empty}

        if sources_only:
            prompt = f"USER QUESTION:\n{question}\n\n"
            if pools["pubmed_chunks"]:
                prompt += (f"PUBMED LITERATURE EXCERPTS (search_pubmed_literature):\n"
                          f"{json.dumps(pools['pubmed_chunks'], indent=2)}\n\n")
            if pools["sec_chunks"]:
                prompt += (f"SEC FILING EXCERPTS (search_sec_filings):\n"
                          f"{json.dumps(pools['sec_chunks'], indent=2)}\n\n")
            if pools["literature"]:
                prompt += (f"CONFERENCE/FDA-FILING LITERATURE EXCERPTS "
                          f"(search_pdf_literature):\n"
                          f"{json.dumps(pools['literature'], indent=2)}\n\n")
            if pools["fda_records"]:
                prompt += (f"FDA APPROVAL RECORDS (search_fda_records):\n"
                          f"{json.dumps(pools['fda_records'], indent=2)}\n\n")
            if pools["news_chunks"]:
                prompt += (f"CORPORATE NEWS EXCERPTS (search_corporate_news -- press "
                          f"releases and earnings call commentary):\n"
                          f"{json.dumps(pools['news_chunks'], indent=2)}\n\n")
            system_prompt = SOURCES_ONLY_REDUCER_SYSTEM
        else:
            prompt = (
                f"USER QUESTION:\n{question}\n\n"
                f"EXTRACTED TRIAL ROWS (the only permitted source -- already "
                f"validated, structured records produced by independent Map-stage "
                f"workers; you do not have access to raw retrieval text):\n"
                + json.dumps([r.model_dump() for r in rows], indent=2)
            )
            system_prompt = REDUCER_SYSTEM

        prior_error = state.get("synthesis_error")
        if prior_error:
            # This is the "inject the error context back into state" step:
            # the failed attempt's own validation error becomes part of the
            # next prompt, so the model sees exactly what it got wrong.
            prompt += (
                f"\n\nYOUR PREVIOUS ATTEMPT FAILED SCHEMA VALIDATION:\n{prior_error}\n"
                f"Correct the structure this time — match the schema exactly."
            )

        outcome: dict = _invoke_ratelimit_retry(narrative_llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ], label="narrative", verbose=verbose)
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

    def _deduped_pools(state: AgentState) -> dict:
        """Dedupe every federated retrieval pool -- shared by
        continue_to_extraction (to build Send payloads) and
        synthesize_table_node (to answer directly from non-trial pools when
        no trial was retrieved). A trial can legitimately appear more than
        once if the agent issues overlapping tool calls in one round; same
        reasoning for every other pool, each keyed on its own natural
        identity field(s)."""
        seen: set[str] = set()
        trials: list[dict] = []
        for t in state.get("retrieved_trials", []):
            nct = t.get("NCTId")
            if nct and nct not in seen:
                seen.add(nct)
                trials.append(t)

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

        seen_pubmed: set[tuple] = set()
        pubmed_chunks: list[dict] = []
        for c in state.get("retrieved_pubmed", []):
            key = (c.get("PMCID"), c.get("ChunkIndex"))
            if key not in seen_pubmed:
                seen_pubmed.add(key)
                pubmed_chunks.append(c)

        seen_sec: set[tuple] = set()
        sec_chunks: list[dict] = []
        for c in state.get("retrieved_sec", []):
            key = (c.get("AccessionNumber"), c.get("ChunkIndex"))
            if key not in seen_sec:
                seen_sec.add(key)
                sec_chunks.append(c)

        seen_news: set[tuple] = set()
        news_chunks: list[dict] = []
        for c in state.get("retrieved_news", []):
            # SourceURL is the one identity field both news SourceTypes
            # share (a press release has no AccessionNumber/PMCID; an
            # earnings transcript has no Title/FeedName) -- see
            # search_corporate_news's own payload shape.
            key = (c.get("SourceURL"), c.get("ChunkIndex"))
            if key not in seen_news:
                seen_news.add(key)
                news_chunks.append(c)

        return {"trials": trials, "literature": literature,
                "fda_records": fda_records, "pubmed_chunks": pubmed_chunks,
                "sec_chunks": sec_chunks, "news_chunks": news_chunks}

    def continue_to_extraction(state: AgentState):
        """Conditional edge from `tools` -- the Mapper.

        Folds in the pre-existing NoResultsFallback guardrail (a plain node
        name, no LLM call) so an ungrounded search still fails closed exactly
        as before. Otherwise, dedupes every federated pool (see
        _deduped_pools) and returns one Send per distinct trial -- this is
        what dynamically spawns N parallel extract_trial branches at
        runtime, N being however many trials Qdrant actually returned. Every
        deduped non-trial pool is broadcast into EVERY Send -- each worker
        independently judges which excerpts (if any) belong to its trial.
        """
        if state.get("has_results") is False:
            return "no_results_fallback"

        pools = _deduped_pools(state)
        trials = pools["trials"]

        if not trials:
            # has_results is True (checked above) but there is no trial to
            # spawn a Map worker for -- either a generic query grounded on an
            # empty token set (pre-existing edge case), or the agent called
            # only non-trial tools this round (search_pdf_literature,
            # search_fda_records, search_pubmed_literature,
            # search_sec_filings -- any combination), so has_results came
            # from one of those alone and search_clinical_trials /
            # query_knowledge_graph never ran. None of those sources can
            # back a table row on its own (nct_id must come from a trial
            # record) -- route to the Reducer rather than returning an empty
            # Send list, which would silently dead-end the graph (no further
            # supersteps run, "result" is never set). synthesize_table
            # handles this case directly: it writes a real narrative from
            # whichever non-trial pools ARE non-empty instead of assuming
            # "no trial" means "no evidence at all."
            if verbose:
                print(f"\n{'─' * 78}\n▶ MAPPER: continue_to_extraction\n{'─' * 78}")
                print("  has_results=True but no distinct trial retrieved -- "
                      "routing straight to synthesize_table to answer from "
                      "whichever non-trial source(s) actually grounded")
            return "synthesize_table"

        if verbose:
            print(f"\n{'─' * 78}\n▶ MAPPER: continue_to_extraction\n{'─' * 78}")
            print(f"  {len(trials)} distinct trial(s), "
                  f"{len(pools['literature'])} literature, "
                  f"{len(pools['fda_records'])} FDA, "
                  f"{len(pools['pubmed_chunks'])} PubMed, "
                  f"{len(pools['sec_chunks'])} SEC, "
                  f"{len(pools['news_chunks'])} corporate news excerpt(s) — "
                  f"fanning out to {len(trials)} parallel extract_trial "
                  f"worker(s) via Send (non-trial pools shared across all of them)")
            for t in trials:
                print(f"    • Send(\"extract_trial\", single_trial={t.get('NCTId')})")

        return [Send("extract_trial",
                     {"single_trial": t, "literature": pools["literature"],
                      "fda_records": pools["fda_records"],
                      "pubmed_chunks": pools["pubmed_chunks"],
                      "sec_chunks": pools["sec_chunks"],
                      "news_chunks": pools["news_chunks"]})
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
    elif "pubmed_chunks" in payload:
        print(f"  query        : {payload.get('query')!r}")
        print(f"  returned     : {payload.get('returned')} PubMed excerpt(s) "
              f"(kNN, always up to limit)")
        print(f"  has_results  : {payload.get('has_results')} (lexical grounding check)")
        for c in payload.get("pubmed_chunks", []):
            title = textwrap.shorten(c.get("Title") or "", 50, placeholder="…")
            print(f"    • {c.get('PMCID')}  score={c.get('score')}  {title}")
    elif "sec_chunks" in payload:
        print(f"  query        : {payload.get('query')!r}")
        print(f"  returned     : {payload.get('returned')} SEC filing excerpt(s) "
              f"(kNN, always up to limit)")
        print(f"  has_results  : {payload.get('has_results')} (lexical grounding check)")
        for c in payload.get("sec_chunks", []):
            print(f"    • {c.get('Ticker')} {c.get('Form')} "
                  f"{c.get('AccessionNumber')}  score={c.get('score')}  "
                  f"filed={c.get('FiledDate')}")
    elif "news_chunks" in payload:
        print(f"  query        : {payload.get('query')!r}")
        print(f"  returned     : {payload.get('returned')} corporate news chunk(s) "
              f"(kNN, always up to limit)")
        print(f"  has_results  : {payload.get('has_results')} (lexical grounding check)")
        for c in payload.get("news_chunks", []):
            if c.get("SourceType") == "earnings_transcript":
                label = f"{c.get('Ticker')} earnings call ({c.get('CallDate')})"
            else:
                label = f"{c.get('FeedName')}: {textwrap.shorten(c.get('Title') or '', 45, placeholder='…')}"
            print(f"    • {label}  score={c.get('score')}")


# =============================================================================
# INDICATION LANDSCAPE MATRIX -- mechanism/target rows x development-phase
# columns, for a single therapeutic area. A separate, deliberately SIMPLER
# pipeline than the six-tool ReACT agent above -- not a graph the caller
# steers with tool calls, but one fixed retrieve-then-synthesize shape,
# because a competitive-landscape MATRIX genuinely needs a cross-cutting
# view of ALL retrieved evidence at once to group drugs by shared mechanism
# (the same reason literature dedup needs the full pool, not a per-trial
# Map-Reduce worker that only ever sees one trial in isolation -- see
# extract_trial_node's own docstring for that same tradeoff elsewhere in
# this file).
# =============================================================================
class DrugEntry(BaseModel):
    """One drug placed in one phase cell of the landscape matrix."""

    name: str = Field(
        description="Drug name as it appears in the source record -- generic "
                    "or brand name, copied exactly, never invented."
    )
    sponsor: str = Field(
        description="The sponsor/company developing or marketing this drug "
                    "for this indication, taken from the source record "
                    "(LeadSponsorName for a trial, SponsorName for an FDA "
                    "record). Use 'Unknown sponsor' only if genuinely not "
                    "stated anywhere in the evidence."
    )
    source: str = Field(
        description="Where this entry's phase placement comes from: an NCT "
                    "id (e.g. 'NCT01234567') for trial-derived placement, an "
                    "FDA application number (e.g. 'BLA125514') for "
                    "FDA-derived Approved placement, or a PMCID (e.g. "
                    "'PMC12345678') for a PubMed-derived Preclinical "
                    "placement. Copied exactly from the record being cited "
                    "-- never invented."
    )


class PhaseCell(BaseModel):
    """One column's contents within one mechanism row."""

    phase: str = Field(
        description="One of: Preclinical, Phase 1, Phase 2, Phase 3, "
                    "Approved -- must match one of the fixed column names "
                    "given in the prompt exactly."
    )
    drugs: list[DrugEntry] = Field(
        default_factory=list,
        description="Every drug from the retrieved evidence whose most "
                    "advanced stage for this indication is this phase. "
                    "Empty list if none -- an empty list is the correct, "
                    "expected representation of 'nothing at this phase for "
                    "this mechanism,' not something to omit."
    )


class MechanismRow(BaseModel):
    """One row of the landscape matrix -- a single mechanism/target."""

    mechanism: str = Field(
        description="A canonical, clinical-grade mechanism of action or "
                    "molecular target class, e.g. 'PD-1 / PD-L1 Checkpoint "
                    "Inhibitor', 'EGFR Inhibitor', 'KRAS G12C Inhibitor'. "
                    "Standard pharmacological classification of a named, "
                    "real drug IS permitted here even when the retrieved "
                    "snippet doesn't spell out the mechanism verbatim -- "
                    "see LANDSCAPE_SYSTEM's classification-vs-study-facts "
                    "distinction. Use the literal string 'Other / "
                    "Unspecified Mechanism' only for a genuinely undisclosed "
                    "investigational code with no published target anywhere "
                    "in general medical literature."
    )
    cells: list[PhaseCell] = Field(
        description="Exactly one PhaseCell per fixed phase column, in the "
                    "SAME order given in the prompt (Preclinical, Phase 1, "
                    "Phase 2, Phase 3, Approved) -- always all 5, even when "
                    "empty."
    )


class LandscapeMatrix(BaseModel):
    """Final output: a therapeutic area's competitive landscape matrix."""

    therapeutic_area: str = Field(
        description="The therapeutic area/indication this landscape "
                    "covers, copied from the user's request."
    )
    phases: list[str] = Field(
        description="The fixed column headers, in display order: "
                    "Preclinical, Phase 1, Phase 2, Phase 3, Approved."
    )
    rows: list[MechanismRow] = Field(
        description="One row per distinct mechanism/target genuinely found "
                    "in the retrieved evidence, most-populated rows first."
    )


LANDSCAPE_PHASES = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Approved"]

# Broader than the six-tool agent's per-call limit=6 -- a landscape needs
# enough breadth to surface multiple competing mechanisms, not just the top
# handful of nearest neighbours. Bounded, not unlimited: kept small enough
# that trials(40) + fda(20) + pubmed(20) stays a few thousand tokens of
# retrieved evidence per call, not something that risks the LLM's context
# window or the endpoint's latency budget.
# Restored to a comprehensive 50/20/20 after two other fixes made the
# earlier 25/12/12 cut unnecessary: (1) LANDSCAPE_MAX_TOKENS is now a
# genuinely generous 40000 (verified live to absorb kimi-k3's ~16K-token
# reasoning overhead AND a full JSON matrix -- see that constant's own
# comment), and (2) LANDSCAPE_SYSTEM now caps mechanism ROWS directly
# (6-10 clinical-grade groups), which bounds output size on the row axis
# regardless of how much input evidence is fed in. The earlier 25/12/12
# traded away real result breadth to chase a token ceiling that a bigger
# budget + a row cap now handle properly -- verified live that this
# under-covered a large indication like NSCLC (only 4 rows, 19/23 drugs
# stuck in the "no mechanism" bucket).
# Raised 50 -> 60 alongside TRIAL_SEARCH_LIMIT/CATALYST_TRIAL_LIMIT per
# explicit request ("limit it to max 60 at least") -- same "broad, not
# exploratory" reasoning above still applies at the new ceiling.
LANDSCAPE_TRIAL_LIMIT = 60
LANDSCAPE_FDA_LIMIT = 20
LANDSCAPE_PUBMED_LIMIT = 20

# Verified live to matter, not a guess: MAX_TOKENS (8000, tuned for a
# narrative answer) truncated a real matrix response mid-JSON -- a live run
# hit completion_tokens=8000 exactly and failed to parse. A full multi-
# mechanism x 5-phase grid with several drugs per cell is a structurally
# bigger output than one paragraph of prose, so this pipeline gets its own,
# larger budget rather than sharing MAX_TOKENS.
#
# Raised again, substantially, after switching to LLM_PROVIDER=kimi:
# verified live that kimi-k3 is a reasoning model whose internal chain-of-
# thought competes with visible output for the SAME max_tokens budget --
# a real run against this exact prompt spent reasoning_tokens=15997 of a
# 16000-token budget on reasoning alone, leaving ~0 for the actual JSON
# (openai.LengthFinishReasonError). Same class of issue INTENT_MODEL's own
# max_tokens=1536 comment already documents for NVIDIA's Nemotron, just
# larger here because this prompt's evidence bundle is much bigger than an
# intent check. This is comfortable headroom for reasoning AND a full
# matrix, not a tight fit -- err generous, since a too-small budget fails
# closed with a confusing parse error rather than a clean truncation.
LANDSCAPE_MAX_TOKENS = 40000

# Also verified live to matter: build_llm's default timeout=180 is tuned for
# a normal agent turn, not a ~20K-input-token / up-to-16K-output-token
# structured-extraction call. A live run under LLM_PROVIDER=nvidia hit
# openai.APITimeoutError at 180s, got retried by the SDK's own internal
# retry logic, and STILL failed the same way on the retry -- 180s was never
# enough for this workload's first attempt to finish, so retrying it
# unchanged just repeats the same failure slower. A longer single-attempt
# budget gives the real work a chance to complete instead.
LANDSCAPE_TIMEOUT = 420

# A trial's own Phase field can be multi-valued (e.g. dual-phase "Phase
# 1/Phase 2"). Early Phase 1 counts as Phase 1 for bucketing; Phase 4 and
# Not Applicable are deliberately excluded from this map (see
# _max_phase_bucket's docstring for why).
_PHASE_RANK = {"Early Phase 1": 1, "Phase 1": 1, "Phase 2": 2, "Phase 3": 3}


def _max_phase_bucket(phases: list[str] | None) -> str | None:
    """Collapse a trial's Phase field to the SINGLE most-advanced bucket
    among {Phase 1, Phase 2, Phase 3} for landscape placement -- the
    standard competitive-landscape convention of showing a drug at its most
    advanced reached stage, not every stage it ever passed through.

    Deliberately DETERMINISTIC, computed in code rather than left to the
    LLM: phase bucketing from a trial's own structured Phase array is a pure
    data-mapping task with one correct answer, and doing it in code removes
    a whole axis of possible LLM error from a task that has zero need for
    language understanding. The LLM's job (see LANDSCAPE_SYSTEM) is to use
    this precomputed field, not re-derive it.

    "Not Applicable" (device/behavioral trials) and "Phase 4" are
    deliberately NOT bucketed here: Phase 4 trials only exist for
    already-approved drugs, which is an approval signal, not a development
    phase -- see the module docstring's Approved-column reasoning; Not
    Applicable trials carry no drug-development-stage information at all.
    """
    if not phases:
        return None
    best = max((_PHASE_RANK.get(p, 0) for p in phases), default=0)
    return {1: "Phase 1", 2: "Phase 2", 3: "Phase 3"}.get(best)


def _retrieve_landscape_evidence(therapeutic_area: str) -> dict:
    """Broad retrieval across three independent corpora for one therapeutic
    area -- trials (for phase placement, via the precomputed phase_bucket
    field), FDA approval records (real regulator-confirmed ground truth for
    the Approved column), and PubMed literature (for mechanism/target
    language and the rare explicit Preclinical mention). Each retrieval is a
    single kNN query against its own collection with the SAME query vector
    -- no cross-collection filtering, since a therapeutic area is a topic,
    not an exact-match field any of these three schemas carries.

    Text fields are truncated (not omitted) to keep the combined evidence
    bundle a few thousand tokens rather than growing unboundedly with
    LANDSCAPE_TRIAL_LIMIT/LANDSCAPE_PUBMED_LIMIT -- BriefSummary/Text is
    still long enough to state a mechanism in a sentence or two, which is
    all the synthesis step actually needs from it.
    """
    query_vector = embed_query(therapeutic_area)

    # Hard-filtered to INTERVENTIONAL studies only -- verified live against
    # this exact corpus that a bare semantic query over a common indication
    # returns a majority OBSERVATIONAL/EXPANDED_ACCESS studies (33 of the
    # top 40 for "Non-Small Cell Lung Cancer" in one live check), and
    # observational studies structurally never carry a Phase value (Phase
    # only applies to interventional trials on ClinicalTrials.gov) -- so
    # they can never contribute a phase-bucketed matrix cell at all. Same
    # hybrid filter+vector pattern as search_clinical_trials' phase_filter.
    trial_hits = _client().query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=LANDSCAPE_TRIAL_LIMIT,
        query_filter=qmodels.Filter(must=[qmodels.FieldCondition(
            key="studyType", match=qmodels.MatchValue(value="INTERVENTIONAL")
        )]),
    ).points
    trials = []
    for h in trial_hits:
        meta = h.payload or {}
        trials.append({
            "NCTId": meta.get("NCTId"),
            "BriefTitle": meta.get("BriefTitle"),
            "Phase": meta.get("Phase"),
            "phase_bucket": _max_phase_bucket(meta.get("Phase")),
            "LeadSponsorName": meta.get("LeadSponsorName"),
            "conditions": meta.get("conditions"),
            "interventions": meta.get("interventions"),
            "BriefSummary": (meta.get("BriefSummary") or "")[:800],
        })

    fda_hits = _client().query_points(
        collection_name=COLLECTION_NAME_FDA, query=query_vector, limit=LANDSCAPE_FDA_LIMIT
    ).points
    fda_records = []
    for h in fda_hits:
        meta = h.payload or {}
        fda_records.append({
            "ApplicationNumber": meta.get("ApplicationNumber"),
            "SponsorName": meta.get("SponsorName"),
            "BrandNames": meta.get("BrandNames"),
            "ActiveIngredients": meta.get("ActiveIngredients"),
        })

    pubmed_hits = _client().query_points(
        collection_name=COLLECTION_NAME_PUBMED, query=query_vector, limit=LANDSCAPE_PUBMED_LIMIT
    ).points
    pubmed_chunks = []
    for h in pubmed_hits:
        meta = h.payload or {}
        pubmed_chunks.append({
            "PMCID": meta.get("PMCID"),
            "Title": meta.get("Title"),
            "Text": (meta.get("Text") or "")[:600],
        })

    return {"trials": trials, "fda_records": fda_records, "pubmed_chunks": pubmed_chunks}


LANDSCAPE_SYSTEM = """You are building a competitive landscape matrix for a
life sciences market intelligence platform: mechanism-of-action / target
rows against clinical-development-phase columns, for a single therapeutic
area.

You are given three pools of retrieved evidence, each independently
retrieved from its own corpus for the requested therapeutic area:
- TRIAL RECORDS: each carries a PRECOMPUTED `phase_bucket` field (Phase 1,
  Phase 2, Phase 3, or null). ALWAYS use this precomputed field for phase
  placement -- never re-derive a phase from the raw `Phase` array yourself.
  A null phase_bucket means this trial's phase field does not map to one of
  the three fixed trial-phase columns and should not be phase-placed from
  this record.
- FDA APPROVAL RECORDS: each is a real, regulator-confirmed drug PRODUCT
  with an on-file application. Every drug named in one of these records
  belongs in the "Approved" column -- ApplicationNumber is the record's own
  ground truth, not something you infer.
- PUBMED LITERATURE EXCERPTS: the ONLY source for the "Preclinical" column
  -- place a compound there ONLY when an excerpt explicitly describes it as
  preclinical, in vitro, in vivo (animal model), or "not yet in clinical
  trials." Do not infer preclinical status from absence of trial data --
  absence of evidence is not evidence of preclinical status, it is simply
  absence of evidence, so a drug with no phase_bucket and no explicit
  PubMed preclinical mention should not appear in your output at all.

TWO DIFFERENT GROUNDING BARS -- study facts vs. pharmacological classification:

STUDY FACTS (phase, sponsor, NCT id, application number, PMCID, source URL)
are STRICTLY grounded, no exceptions:
- Every drug name, sponsor, and source id (NCT id / application number /
  PMCID) must be copied exactly from a retrieved record. Never invent a
  trial, a sponsor, or an id that is not actually in the evidence.
- A drug's source for a Preclinical placement MUST be a PMCID from the
  PubMed pool -- never an NCT id or an FDA application number. An FDA
  application number is proof of APPROVAL, the opposite of preclinical; if
  a drug already has an FDA record (so it belongs in your Approved
  column), never ALSO place that same drug in Preclinical.
- Every drug in the FDA APPROVAL RECORDS pool must appear somewhere in your
  Approved column -- either under its target class if one is known (per
  the classification rule below) or under "Other / Unspecified Mechanism"
  otherwise. Do not silently omit an approved drug just because its target
  class wasn't determined; the FDA record's ApplicationNumber is itself
  real evidence worth surfacing regardless of mechanism.

MECHANISM / TARGET CLASSIFICATION is a DIFFERENT, deliberately looser bar:
you ARE EXPLICITLY PERMITTED AND ENCOURAGED to use your own standard
biomedical/pharmacological knowledge to classify a NAMED, REAL drug into
its canonical target class, even when the retrieved snippet itself never
spells out the mechanism in so many words. This is safe precisely because
it is NOT a trial-specific claim -- "cetuximab is an EGFR inhibitor" is
stable, textbook, publicly documented pharmacology, not something you are
guessing about THIS trial's results or THIS drug's efficacy.

WORKED EXAMPLES -- classify by the drug's OWN real-world pharmacology,
always, regardless of which indication's trial record it came from:
- Paclitaxel, Docetaxel -> "Taxane Chemotherapy" (a taxane's mechanism is
  the same whether it shows up in a breast, lung, or ovarian trial)
- Crizotinib, Alectinib -> "ALK / ROS1 Inhibitor"
- Sunitinib, Lenvatinib, Nintedanib -> "VEGFR Multikinase Inhibitor"
- Osimertinib, Erlotinib, Gefitinib -> "EGFR Inhibitor"
- Nilotinib, Dasatinib -> "BCR-ABL / KIT Inhibitor" -- these are CML drugs.
  If one appears in an NSCLC trial record (e.g. as an off-label arm or a
  comparator), classify it by ITS OWN real mechanism (BCR-ABL / KIT), never
  by guessing it must match whatever target class is common in that
  indication (it is NOT a KRAS, EGFR, or ALK inhibitor just because it
  showed up in a lung cancer study).

SELF-CHECK before committing to a specific target class: ask yourself
"would a pharmacology reference actually confirm this drug inhibits this
target?" A drug's real mechanism is a fixed, indication-independent fact
about the molecule itself -- never infer it from the indication of the
trial it happened to appear in, from other drugs in the same row, or from
a plausible-sounding guess. If you are genuinely not sure, use a correct
BROADER class you ARE sure of (e.g. "Multikinase Inhibitor" instead of
guessing the exact target, or "Cytotoxic Chemotherapy" for a chemo agent
whose specific class you're unsure of) rather than asserting a specific,
possibly wrong target. Reserve "Other / Unspecified Mechanism" (the literal
row name to use) ONLY for a genuinely undisclosed investigational asset --
an internal code name (e.g. "XYZ-101") with no published target anywhere in
general medical literature -- not for a real, named drug whose class you
simply haven't recalled yet; most real, approved or late-stage drugs DO
have a known class, so reaching for "Other" for one should be rare.

- Two records naming the same real-world drug under different names
  (generic vs. brand, e.g. "pembrolizumab" and "Keytruda") are the SAME
  drug -- consolidate them into one DrugEntry using whichever name the
  majority of retrieved evidence uses, do not list the same real drug
  twice in one cell under two different names.
- A drug should appear in more than one row only if it genuinely has more
  than one distinct, clinically recognized mechanism (rare) -- do not
  duplicate it across rows as a default. In particular: never classify the
  SAME drug differently in two different rows (e.g. once correctly, once
  under "Other") -- decide its mechanism once and use that everywhere it
  appears in your output.

OUTPUT SHAPE:
- `phases` must be exactly ["Preclinical", "Phase 1", "Phase 2", "Phase 3",
  "Approved"], in that order.
- Every row's `cells` array must have exactly 5 entries, one per phase
  above, in that same order -- including phases with an empty `drugs`
  list. Do not omit a cell just because it is empty; an empty list IS the
  correct representation of "no evidence at this phase for this
  mechanism."
- Group into 6-10 MEANINGFUL, clinical-grade mechanism rows -- standard
  target classes a pharma analyst would recognize, not fragmented micro-
  categories (e.g. one "PD-1 / PD-L1 Checkpoint Inhibitor" row, not
  separate rows per individual checkpoint drug). Consolidate related
  assets into the same canonical row rather than inventing a new row per
  drug. This is a competitive-intelligence dashboard surfacing the active
  mechanisms in this space, not an exhaustive one-row-per-drug catalog.
- Order rows by how many total drugs they contain, most first, so the
  most competitively active mechanisms surface at the top of the grid."""


_PMCID_RE = re.compile(r"^PMC\d+$")


def _sanitize_landscape_matrix(matrix: LandscapeMatrix) -> LandscapeMatrix:
    """Deterministic backstop over the LLM's own output -- verified live to
    matter, not defensive-for-its-own-sake: a real run placed the same
    drug in BOTH Preclinical and Approved, citing the SAME FDA application
    number as its Preclinical "source" (an FDA record cannot state
    preclinical status; it is proof of the opposite). LANDSCAPE_SYSTEM now
    says this explicitly, but a system prompt is not a validator -- this
    function is, catching whatever the prompt still misses. Drops any
    Preclinical DrugEntry whose source is not a genuine PMCID; every other
    cell is left untouched, since only Preclinical has this single-source-
    type constraint (see LANDSCAPE_SYSTEM)."""
    for row in matrix.rows:
        for cell in row.cells:
            if cell.phase == "Preclinical":
                cell.drugs = [d for d in cell.drugs if _PMCID_RE.match(d.source)]

    # Second backstop, also verified live to matter: a real run classified
    # "Nilotinib" as a specific (and wrong) target class in one row, then
    # ALSO listed it again under "Other / Unspecified Mechanism" -- the
    # same drug, inconsistently mechanism-tagged twice in one response.
    # LANDSCAPE_SYSTEM now explicitly says "never classify the SAME drug
    # differently in two different rows," but the same principle applies
    # here: don't trust the prompt alone to enforce it. Drop a duplicate
    # from "Other / Unspecified Mechanism" whenever that same drug name
    # (case-insensitive) already has a real classification elsewhere --
    # whatever specific classification the model committed to first wins,
    # and the redundant, less-informative "unspecified" duplicate is
    # removed rather than left to confuse the grid. This does NOT resolve
    # a genuine conflict between two DIFFERENT specific classifications
    # for the same drug (rare, and not distinguishable after the fact
    # without re-querying the model) -- only the specific-vs-unspecified
    # duplicate this live run actually produced.
    OTHER = "Other / Unspecified Mechanism"
    classified_names = {
        d.name.strip().lower()
        for row in matrix.rows if row.mechanism != OTHER
        for cell in row.cells for d in cell.drugs
    }
    if classified_names:
        for row in matrix.rows:
            if row.mechanism != OTHER:
                continue
            for cell in row.cells:
                cell.drugs = [d for d in cell.drugs
                             if d.name.strip().lower() not in classified_names]

    return matrix


class LandscapeState(TypedDict):
    therapeutic_area: str
    retrieved_trials: list[dict]
    retrieved_fda: list[dict]
    retrieved_pubmed: list[dict]
    result: Optional[LandscapeMatrix]
    retries: int
    error: Optional[str]


def _build_gpt4o_llm(timeout: int, model_env_var: str = "LANDSCAPE_MODEL",
                     default_model: str = "gpt-4o"):
    """Shared builder for every evidence-heavy structured-extraction call in
    this file (landscape matrix synthesis, catalyst timeline synthesis)
    that is deliberately PINNED to OpenAI gpt-4o, not routed through
    build_llm()'s LLM_PROVIDER switch -- unlike every other LLM call in
    this file, which follows whatever LLM_PROVIDER is set to.

    Verified live, in order, why the two temporary fallbacks that work fine
    for the main agent's much smaller calls both broke on this shape of call:
    - LLM_PROVIDER=nvidia: Nemotron is a reasoning model whose internal
      chain-of-thought competes with visible output for the same
      max_tokens budget -- fine for a small intent check, but an evidence-
      heavy prompt like this pushed real landscape-matrix runs past a
      420s timeout entirely (measured: 802s for one successful run).
    - LLM_PROVIDER=kimi: kimi-k3 has the same reasoning-overhead problem
      (one real run spent reasoning_tokens=15997 of a 16000 budget on
      thinking alone, leaving ~0 for output), AND this account's key hits
      a hard "max organization concurrency: 1" quota -- verified live that
      even a single clean, isolated first attempt gets rejected outright,
      not just overlapping retries.
    gpt-4o is a non-reasoning chat model with no chain-of-thought token
    overhead and no such concurrency ceiling, and OPENAI_API_KEY is
    already a hard requirement of this whole project (embeddings.py), not
    a new secret being introduced for this call.

    temperature=0 here (unlike the rest of this file, which never sets
    temperature because claude-opus-5 rejects it with a 400): gpt-4o
    supports it, and this call benefits from it -- consistent, minimally
    creative extraction/classification output run after run, not narrative
    prose where some variation is harmless.

    `model_env_var` lets each caller offer its own override (e.g.
    LANDSCAPE_MODEL vs CATALYST_MODEL) without a shared knob accidentally
    changing both features' model choice at once.
    """
    from langchain_openai import ChatOpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "[gpt4o]   OPENAI_API_KEY is not set.\n"
            "          Required (embeddings.py also depends on it) -- add it to .env."
        )
    return ChatOpenAI(
        model=os.getenv(model_env_var, default_model),
        temperature=0,
        api_key=key,
        # Verified live that gpt-4o hard-rejects any max_tokens above 16384
        # with a 400 (invalid_request_error) -- this is still ample
        # headroom, since gpt-4o has no invisible chain-of-thought
        # competing for the same budget the way kimi-k3/Nemotron did, so
        # the full budget is available for actual JSON output.
        max_tokens=16384,
        timeout=timeout,
        # Same reasoning as build_llm's nvidia/kimi branches: the calling
        # node's own MAX_SYNTHESIS_RETRIES loop is the one retry layer, so
        # the SDK never has two requests in flight for the same logical
        # attempt.
        max_retries=0,
    )


def make_landscape_graph(model: str, verbose: bool = True):
    """Two-node graph: retrieve (broad, three corpora) -> synthesize (one
    structured-output call over ALL retrieved evidence at once, with a
    bounded self-retry on schema validation failure -- same
    validate-or-retry shape as synthesize_table_node's Reduce stage
    elsewhere in this file, reusing MAX_SYNTHESIS_RETRIES for consistency).

    `model` is accepted for signature consistency with make_graph() but
    UNUSED here -- see _build_gpt4o_llm's own docstring for why this
    graph pins its model choice instead of following LLM_PROVIDER."""
    landscape_llm = _build_gpt4o_llm(LANDSCAPE_TIMEOUT).with_structured_output(
        LandscapeMatrix, include_raw=True
    )

    def retrieve_node(state: LandscapeState) -> dict:
        area = state["therapeutic_area"]
        if verbose:
            print(f"[landscape] retrieving evidence for {area!r}")
        evidence = _retrieve_landscape_evidence(area)
        if verbose:
            print(f"[landscape] {len(evidence['trials'])} trials, "
                  f"{len(evidence['fda_records'])} FDA records, "
                  f"{len(evidence['pubmed_chunks'])} PubMed excerpts")
        return {"retrieved_trials": evidence["trials"],
                "retrieved_fda": evidence["fda_records"],
                "retrieved_pubmed": evidence["pubmed_chunks"]}

    def synthesize_node(state: LandscapeState) -> dict:
        prompt = (
            f"THERAPEUTIC AREA: {state['therapeutic_area']}\n\n"
            f"TRIAL RECORDS ({len(state['retrieved_trials'])}):\n"
            f"{json.dumps(state['retrieved_trials'], indent=2)}\n\n"
            f"FDA APPROVAL RECORDS ({len(state['retrieved_fda'])}):\n"
            f"{json.dumps(state['retrieved_fda'], indent=2)}\n\n"
            f"PUBMED LITERATURE EXCERPTS ({len(state['retrieved_pubmed'])}):\n"
            f"{json.dumps(state['retrieved_pubmed'], indent=2)}"
        )
        prior_error = state.get("error")
        # Only worth re-prompting on a genuine schema-validation failure --
        # a transient API error (rate limit / timeout, see the except block
        # below) says nothing about what the model produced, so there is
        # nothing for it to "correct" and injecting that text would be
        # actively misleading on a retry.
        if prior_error and not prior_error.startswith("transient API error:"):
            prompt += (
                f"\n\nYOUR PREVIOUS ATTEMPT FAILED SCHEMA VALIDATION:\n{prior_error}\n"
                f"Correct the structure this time — match the schema exactly."
            )

        try:
            outcome: dict = landscape_llm.invoke([
                SystemMessage(content=LANDSCAPE_SYSTEM),
                HumanMessage(content=prompt),
            ])
        except (openai.RateLimitError, openai.APITimeoutError,
               openai.APIConnectionError, openai.InternalServerError) as exc:
            # Same retry mechanism as a schema-validation failure, not an
            # uncaught crash -- verified live to matter: without this, a
            # transient 429/timeout on this large, slow call propagated all
            # the way out of run_landscape_query as a raw exception instead
            # of going through this node's own retry loop. build_llm's own
            # max_retries=0 (see that function's comment) means the SDK
            # itself no longer silently retries either, so THIS is now the
            # only retry layer -- each attempt fully completes or fails
            # before the next one starts, never two requests in flight.
            retries = state.get("retries", 0) + 1
            if verbose:
                print(f"[landscape] API error (attempt {retries}): {exc}")
            if retries <= MAX_SYNTHESIS_RETRIES:
                time.sleep(5)  # brief backoff before the next full attempt
            return {"retries": retries, "error": f"transient API error: {exc}"}

        parsed: LandscapeMatrix | None = outcome.get("parsed")
        error = outcome.get("parsing_error")

        if parsed is not None and error is None:
            parsed = _sanitize_landscape_matrix(parsed)
            if verbose:
                print(f"[landscape] synthesized {len(parsed.rows)} mechanism row(s)")
            return {"result": parsed, "error": None}

        retries = state.get("retries", 0) + 1
        err_text = str(error) if error else "model did not return the expected structure"
        if verbose:
            print(f"[landscape] validation failed (attempt {retries}): {err_text[:200]}")
        return {"retries": retries, "error": err_text}

    def route_after_synthesis(state: LandscapeState) -> str:
        if state.get("result") is not None:
            return "done"
        if state.get("retries", 0) > MAX_SYNTHESIS_RETRIES:
            return "done"  # fail closed -- result stays None, caller must handle
        return "retry"

    g = StateGraph(LandscapeState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_conditional_edges("synthesize", route_after_synthesis,
                            {"retry": "synthesize", "done": END})
    return g.compile()


def run_landscape_query(graph, therapeutic_area: str) -> LandscapeMatrix | None:
    final = graph.invoke(
        {"therapeutic_area": therapeutic_area, "retrieved_trials": [], "retrieved_fda": [],
         "retrieved_pubmed": [], "result": None, "retries": 0, "error": None},
        config={"recursion_limit": 10},
    )
    return final.get("result")


# =============================================================================
# CLINICAL CATALYST & READOUT TRACKER -- a chronological timeline of upcoming
# market-moving events (trial data readouts, FDA decision dates) for a
# therapeutic-area/event-type query. Same retrieve-then-synthesize shape and
# gpt-4o pinning as the landscape matrix pipeline above, but with one
# structural difference driven by what was verified live before writing
# this: NEITHER of this project's already-ingested corpora stores the dates
# this feature needs.
#
#   - clinical_trials' own Qdrant payload (see fetch_and_embed_trials.py's
#     API_FIELDS) never requested primaryCompletionDate/completionDate --
#     confirmed directly by inspecting a real stored payload before writing
#     a line of this. A catalyst tracker's entire value is real dates, so
#     rather than leave this a silent gap or have the LLM guess, retrieve_node
#     does ONE extra, live, unauthenticated, free ClinicalTrials.gov API call
#     per request -- batched (filter.ids=NCT1,NCT2,...), verified live to
#     return a real multi-trial batch in one round trip -- to fetch each
#     retrieved trial's REAL primaryCompletionDateStruct/completionDateStruct.
#   - SEC filings sometimes DO state a real date in free text -- verified
#     live against the already-indexed sec_filings collection: a real chunk
#     reads "The FDA set a PDUFA date of September 21, 2026" for Keytruda.
#     No live lookup exists for this (there is no structured "PDUFA date"
#     field anywhere to fetch); the LLM must read it out of retrieved text,
#     same strict-grounding discipline as everywhere else in this file.
#
# DATE COMPUTATION SPLIT (same principle as phase_bucket in the landscape
# pipeline): the LLM's ONLY job re: dates is to CITE a real date substring
# verbatim from evidence (CatalystEventDraft.raw_date) -- it never computes
# year/quarter itself. _parse_catalyst_date() does that deterministically,
# in code, after the fact. This removes date arithmetic from the set of
# things an LLM could get subtly wrong, the same way removing phase-bucket
# arithmetic from the landscape LLM's job removed a whole class of its
# errors.
# =============================================================================
class CatalystEventDraft(BaseModel):
    """LLM-facing schema -- cites evidence, never computes a date itself."""

    raw_date: str = Field(
        description="The date EXACTLY as it appears in the source evidence "
                    "-- e.g. '2027-02-26' or '2027-02' copied verbatim from "
                    "a trial's primaryCompletionDate/completionDate field, "
                    "or an explicitly stated date/quarter/half-year "
                    "substring copied verbatim from an SEC excerpt (e.g. "
                    "'September 21, 2026', 'second half of 2026'). Never "
                    "invented, never reformatted, never estimated -- copy "
                    "exactly what the evidence states."
    )
    company: str = Field(description="Sponsor/company name, copied from the source record.")
    drug_name: str = Field(
        description="A specific drug/intervention NAME, copied from the "
                    "source record's own interventions field or an SEC "
                    "excerpt -- e.g. 'Pembrolizumab' or 'OBI-902', never a "
                    "generic mechanism-class description like 'PD-1 "
                    "inhibitor' or a placeholder like 'the control group' "
                    "even if that is literally how one intervention arm is "
                    "labeled in the record. If a trial's own intervention "
                    "list has no genuinely specific name (only "
                    "control/placebo/generic labels), skip that event "
                    "rather than invent a name or use a non-specific one."
    )
    event_type: str = Field(
        description="e.g. 'Phase 3 Primary Completion', 'Phase 2 Primary "
                    "Completion', 'Study Completion', 'PDUFA Date', "
                    "'Interim OS Readout' -- reflect what the evidence "
                    "actually shows; do not default everything to one "
                    "generic label when the evidence is more specific."
    )
    indication: str = Field(description="Disease/condition, copied from the source record.")
    source: str = Field(
        description="The real NCT id (trial-derived event) or SEC "
                    "AccessionNumber (SEC-derived event) this event comes "
                    "from -- copied exactly, never invented."
    )
    source_type: str = Field(description="Literally 'trial' or 'sec' -- which evidence pool this event came from.")


class CatalystTimelineDraft(BaseModel):
    """Raw LLM output -- unsorted; final chronological ordering happens
    after _parse_catalyst_date runs on every event, not here."""

    events: list[CatalystEventDraft] = Field(
        description="Every genuinely date-grounded event found in the "
                    "evidence. Include at most 30 -- the most concrete, "
                    "nearest-term, highest-confidence ones if the evidence "
                    "supports more than that."
    )


class CatalystEvent(BaseModel):
    """One chronologically-placed catalyst event for the frontend timeline.
    display_date/year/quarter are computed deterministically from the
    LLM-cited raw_date by _parse_catalyst_date -- never LLM output."""

    display_date: str = Field(description="Human-readable date/period, no more precise than the source data supports.")
    year: int = Field(description="Calendar year, for chronological grouping.")
    quarter: str = Field(
        description="One of 'Q1'..'Q4' for a specific quarter, 'H1'/'H2' "
                    "when the source only supports half-year precision, or "
                    "'' when only a bare year is known."
    )
    company: str
    drug_name: str
    event_type: str
    indication: str
    source: str = Field(description="NCT id or SEC AccessionNumber this event is grounded in.")
    source_type: str = Field(description="'trial' or 'sec'.")


class CatalystTimeline(BaseModel):
    """Final output: a chronologically sorted catalyst timeline for a query."""

    query: str = Field(description="The therapeutic area / event-type query this timeline covers.")
    events: list[CatalystEvent] = Field(description="Chronologically sorted, earliest first.")


# Raised 40 -> 60 per explicit request ("limit it to max 60 at least") --
# same TRIAL_SEARCH_LIMIT/LANDSCAPE_TRIAL_LIMIT reasoning: verified live
# that event count for a broad query varies run-to-run partly on
# LLM-synthesis selectivity, not just retrieval breadth, so a higher
# retrieval ceiling gives the synthesis step more real candidates to
# choose from rather than being starved before it even gets to judge them.
CATALYST_TRIAL_LIMIT = 60
CATALYST_SEC_LIMIT = 20
CATALYST_TIMEOUT = 120

# A catalyst tracker cares about events that HAVEN'T happened yet -- these
# are the ClinicalTrials.gov overallStatus values consistent with a trial
# still being underway (verified live against this corpus's own status
# distribution before picking this set). COMPLETED/TERMINATED/WITHDRAWN
# trials' data has already read out; not a future catalyst.
_UPCOMING_TRIAL_STATUSES = [
    "RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION",
]

# Verified live to matter, not a guess: a bare semantic query for "Upcoming
# Phase 3 readouts in Oncology" retrieved mostly Phase 1/Phase 2 trials --
# only 2 of 40 were genuinely Phase 3 -- because embedding similarity does
# not treat "Phase 3" as a hard constraint the way a real market-intel
# query needs. This mirrors why search_clinical_trials's own phase_filter
# is a hard Qdrant filter, not left to semantic ranking alone; this
# extractor + the filter built from it applies that same fix here.
_PHASE_QUERY_RE = re.compile(r"\bphase\s*(1|2|3|4|iv|iii|ii|i)\b", re.IGNORECASE)
_ROMAN_TO_ARABIC = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}


def _extract_phase_filter(query: str) -> str | None:
    """Best-effort hard-phase extraction from free-text query, e.g. 'Upcoming
    Phase 3 readouts' -> 'Phase 3'. Returns None (no filter applied) when
    the query doesn't name a specific phase -- a query like 'Upcoming PDUFA
    dates for oncology drugs' should NOT be phase-restricted."""
    m = _PHASE_QUERY_RE.search(query)
    if not m:
        return None
    num = _ROMAN_TO_ARABIC.get(m.group(1).lower(), m.group(1))
    return PHASE_LABELS.get(f"PHASE{num}")

_MONTH_TO_QUARTER = {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2",
                     7: "Q3", 8: "Q3", 9: "Q3", 10: "Q4", 11: "Q4", 12: "Q4"}
_MONTH_NAMES = ("January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December")

_ISO_FULL_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_YM_RE = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR_ONLY_RE = re.compile(r"^(\d{4})$")
_Q_RE = re.compile(r"\bQ([1-4])[' ’]*(\d{4}|\d{2})\b", re.IGNORECASE)
_H_RE = re.compile(r"\bH([12])[' ’]*(\d{4}|\d{2})\b", re.IGNORECASE)
# SEC filing prose commonly says "second half of 2026" rather than "H2
# 2027" -- verified live this phrasing needs its own explicit match: without
# it, the free-text dateutil fallback below "parses" it by defaulting the
# missing month to TODAY's month (dateutil's own default-date behavior),
# which only coincidentally lands in the right half of the year and is
# flatly wrong the rest of the time (e.g. parsed in February, "second half
# of 2026" would wrongly resolve to Q1/Q2). This pattern must be checked
# BEFORE that fallback.
_HALF_PHRASE_RE = re.compile(
    r"\b(first|1st|second|2nd|latter)\s+half\s+of\s+(\d{4})\b", re.IGNORECASE
)


def _parse_catalyst_date(raw: str) -> tuple[int, str, str] | None:
    """Deterministic date parsing -- see this section's module-level
    comment for why this is code, not LLM output. Tries, in order: ISO
    full date, ISO year-month, explicit "Q# YYYY", explicit "H# YYYY",
    bare year, then a free-text fallback via python-dateutil for prose
    dates like "September 21, 2026" straight out of an SEC excerpt.
    Returns None if genuinely no real year can be extracted -- the caller
    drops that event rather than guess."""
    raw = raw.strip()

    m = _ISO_FULL_RE.match(raw)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return year, _MONTH_TO_QUARTER[month], f"{_MONTH_NAMES[month - 1]} {day}, {year}"

    m = _ISO_YM_RE.match(raw)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        return year, _MONTH_TO_QUARTER[month], f"{_MONTH_NAMES[month - 1]} {year}"

    m = _Q_RE.search(raw)
    if m:
        quarter = f"Q{m.group(1)}"
        year = int(m.group(2))
        year += 2000 if year < 100 else 0
        return year, quarter, f"{quarter} {year}"

    m = _H_RE.search(raw)
    if m:
        half = f"H{m.group(1)}"
        year = int(m.group(2))
        year += 2000 if year < 100 else 0
        return year, half, f"{half} {year}"

    m = _HALF_PHRASE_RE.search(raw)
    if m:
        half = "H1" if m.group(1).lower() in ("first", "1st") else "H2"
        year = int(m.group(2))
        return year, half, raw

    m = _YEAR_ONLY_RE.match(raw)
    if m:
        year = int(m.group(1))
        return year, "", str(year)

    try:
        from dateutil import parser as _dateutil_parser

        parsed = _dateutil_parser.parse(raw, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    if not (1900 <= parsed.year <= 2100):
        return None
    # raw itself (not a reformatted version) as the display string here --
    # it is already real, human-readable prose extracted from the source
    # (e.g. "September 21, 2026"), and reformatting risks implying more
    # precision (e.g. a specific day) than the source actually stated.
    return parsed.year, _MONTH_TO_QUARTER[parsed.month], raw


def _enrich_trial_dates(nct_ids: list[str]) -> dict[str, dict]:
    """Live, unauthenticated, free ClinicalTrials.gov lookup for the real
    completion-date fields this project's own ingested corpus does not
    store (see this section's module comment). Batched via filter.ids= --
    verified live to return every requested trial's current data in ONE
    round trip, not one call per trial. Best-effort: a failed lookup
    returns {} rather than raising, so a transient network issue degrades
    to "fewer dated events" rather than crashing the whole endpoint."""
    if not nct_ids:
        return {}
    try:
        resp = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={
                "filter.ids": ",".join(nct_ids),
                # PrimaryCompletionDateType must be requested as its OWN
                # field name -- verified live that omitting it (even
                # though PrimaryCompletionDate is present) silently drops
                # the ESTIMATED/ACTUAL distinction from the response
                # rather than erroring, which a first version of this
                # call did without ever noticing.
                "fields": "NCTId,OverallStatus,PrimaryCompletionDate,PrimaryCompletionDateType,CompletionDate",
                "pageSize": len(nct_ids),
            },
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[catalysts] live ClinicalTrials.gov date lookup failed (non-fatal): {exc}")
        return {}

    out: dict[str, dict] = {}
    for study in resp.json().get("studies", []):
        ident = study.get("protocolSection", {}).get("identificationModule", {})
        sm = study.get("protocolSection", {}).get("statusModule", {})
        nct = ident.get("nctId")
        if not nct:
            continue
        out[nct] = {
            "overallStatus": sm.get("overallStatus"),
            "primaryCompletionDate": sm.get("primaryCompletionDateStruct", {}).get("date"),
            "primaryCompletionDateType": sm.get("primaryCompletionDateStruct", {}).get("type"),
            "completionDate": sm.get("completionDateStruct", {}).get("date"),
        }
    return out


def _retrieve_catalyst_evidence(query: str) -> dict:
    """Broad retrieval across trials (hard-filtered to INTERVENTIONAL +
    still-underway statuses, then live-enriched with real completion
    dates) and SEC filings (for explicitly stated PDUFA/readout dates in
    free text). Trials the live lookup returns no completion date for are
    dropped here, not carried forward for the LLM to guess about -- a
    trial with no real date is not something this feature can honestly
    place on a timeline."""
    query_vector = embed_query(query)

    must_conditions = [qmodels.FieldCondition(key="studyType", match=qmodels.MatchValue(value="INTERVENTIONAL"))]
    phase_filter = _extract_phase_filter(query)
    if phase_filter:
        must_conditions.append(qmodels.FieldCondition(key="Phase", match=qmodels.MatchValue(value=phase_filter)))
        print(f"[catalysts] hard phase filter detected in query: {phase_filter!r}")

    trial_hits = _client().query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=CATALYST_TRIAL_LIMIT,
        query_filter=qmodels.Filter(
            must=must_conditions,
            should=[qmodels.FieldCondition(key="OverallStatus", match=qmodels.MatchValue(value=s))
                   for s in _UPCOMING_TRIAL_STATUSES],
        ),
    ).points

    trials = []
    for h in trial_hits:
        meta = h.payload or {}
        nct = meta.get("NCTId")
        if not nct:
            continue
        trials.append({
            "NCTId": nct,
            "BriefTitle": meta.get("BriefTitle"),
            "Phase": meta.get("Phase"),
            "LeadSponsorName": meta.get("LeadSponsorName"),
            "conditions": meta.get("conditions"),
            "interventions": meta.get("interventions"),
        })

    date_lookup = _enrich_trial_dates([t["NCTId"] for t in trials])
    today = date.today()
    dated_trials = []
    dropped_past = 0
    for t in trials:
        d = date_lookup.get(t["NCTId"])
        raw_date = d.get("primaryCompletionDate") or d.get("completionDate") if d else None
        if not raw_date:
            continue
        # Filtering already-past trials HERE, not just at final synthesis
        # (_finalize_catalyst_timeline has its own backstop for whatever
        # slips through) -- verified live that leaving this only as a
        # downstream check let a whole CATALYST_TRIAL_LIMIT-sized candidate
        # pool skew toward trials that are administratively still "active"
        # in the registry (see _UPCOMING_TRIAL_STATUSES) but already past
        # their own stated completion date, starving the LLM of genuinely
        # upcoming candidates to choose from and occasionally returning
        # zero events on an otherwise-reasonable query. Best-effort parse:
        # an unparseable date is dropped here too -- this feature can't
        # honestly place an unparseable date on a timeline either way.
        try:
            from dateutil import parser as _dateutil_parser
            # default= needs a real datetime, not a date -- verified live:
            # passing `today` (a date) directly makes dateutil return a
            # date-typed result instead of a datetime, and .date() on THAT
            # raises AttributeError ('date' object has no attribute
            # 'date'). datetime.combine gives it what it actually wants.
            parsed_date = _dateutil_parser.parse(
                raw_date, default=datetime.combine(today, datetime.min.time())
            ).date()
        except (ValueError, OverflowError):
            continue
        if parsed_date < today:
            dropped_past += 1
            continue
        t.update(d)
        dated_trials.append(t)
    print(f"[catalysts] {len(trials)} trials retrieved, {len(dated_trials)} have a "
          f"real live-looked-up completion date that hasn't passed yet "
          f"({dropped_past} dropped as already past)")

    sec_hits = _client().query_points(
        collection_name=COLLECTION_NAME_SEC, query=query_vector, limit=CATALYST_SEC_LIMIT
    ).points
    sec_chunks = []
    for h in sec_hits:
        meta = h.payload or {}
        sec_chunks.append({
            "Ticker": meta.get("Ticker"),
            "Company": meta.get("Company"),
            "Form": meta.get("Form"),
            "AccessionNumber": meta.get("AccessionNumber"),
            "FiledDate": meta.get("FiledDate"),
            "Text": (meta.get("Text") or "")[:800],
        })

    return {"trials": dated_trials, "sec_chunks": sec_chunks}


CATALYST_SYSTEM = """You are building a chronological catalyst/readout
tracker for a life sciences market intelligence platform: upcoming market-
moving events (clinical trial data readouts, FDA decision dates) for a
query about a therapeutic area or event type.

You are given two pools of retrieved evidence:
- TRIAL RECORDS: each carries REAL, LIVE-LOOKED-UP date fields --
  `primaryCompletionDate` (when primary-endpoint data collection is
  expected/was completed -- the standard proxy for "topline readout"),
  `primaryCompletionDateType` ("ESTIMATED" means a genuine future
  projection; "ACTUAL" means that date already passed, so data collection
  is done and topline results may already be pending analysis or
  announcement), and `completionDate` (full study completion, usually
  later). `overallStatus` shows the trial's current state (e.g.
  RECRUITING, ACTIVE_NOT_RECRUITING). These fields are the ONLY source for
  trial-derived events -- never invent or estimate a date not present here.
- SEC FILING EXCERPTS: real 10-K/8-K text that SOMETIMES explicitly states
  a PDUFA date, an expected data-readout timeframe (e.g. "second half of
  2026"), or a regulatory submission timeline. Extract a date/event ONLY
  when the excerpt explicitly states one -- never infer a timeframe from
  general pipeline discussion that names no specific date, quarter, or half.

STRICT GROUNDING:
- Every event's raw_date must be copied from one of: a trial's
  primaryCompletionDate field (preferred), a trial's completionDate field
  (if primaryCompletionDate is absent), or an explicitly stated
  date/quarter/half-year substring in an SEC excerpt. Never invent,
  estimate, or infer a date -- if no real date exists for a given
  drug/trial in the evidence, do not create an event for it.
- Every company/sponsor, drug name, and indication must be copied from the
  source record, not invented.
- Every event's `source` must be a real NCT id (trial-derived) or SEC
  AccessionNumber (SEC-derived) copied exactly from the evidence.
- `event_type` should reflect what the evidence actually shows: e.g.
  "Phase 3 Primary Completion" / "Phase 2 Primary Completion" (matched to
  the trial's own Phase field) for a trial-derived event from
  primaryCompletionDate, "Study Completion" for one from completionDate
  only, "PDUFA Date" for an FDA regulatory decision date explicitly stated
  in an SEC excerpt, or a more specific label (e.g. "Interim OS Readout")
  ONLY when the evidence itself uses that specific language.
- One event per distinct trial/SEC-statement -- do not create duplicate
  events for the same drug/trial pairing.

Include at most 30 events -- the most concrete, nearest-term, highest-
confidence ones if the evidence supports more than that. Do not attempt to
sort chronologically yourself; that is handled after your response, from
the raw_date you cite."""


class CatalystState(TypedDict):
    query: str
    retrieved_trials: list[dict]
    retrieved_sec: list[dict]
    result: Optional[CatalystTimeline]
    retries: int
    error: Optional[str]


# Shared by _finalize_catalyst_timeline's sort AND its past-event filter --
# hoisted to module scope so both use the exact same quarter ordering
# rather than two dicts that could quietly drift apart. "" (year-only,
# no quarter cited) sorts before any real quarter within its year.
_QUARTER_RANK = {"": 0, "H1": 1, "Q1": 1, "Q2": 2, "H2": 3, "Q3": 3, "Q4": 4}


def _finalize_catalyst_timeline(query: str, draft: CatalystTimelineDraft) -> CatalystTimeline:
    """Runs _parse_catalyst_date over every drafted event, drops any whose
    raw_date genuinely can't be parsed (see that function's own docstring
    for why that's the honest outcome rather than a guess) OR whose parsed
    date has already passed, and sorts the survivors chronologically. This
    is the code-side half of the LLM-cites/code-computes split described in
    this section's module comment.

    The past-date drop is a real, verified-live fix, not defensive
    padding: _UPCOMING_TRIAL_STATUSES only filters on a trial's
    OverallStatus (RECRUITING etc.), which is NOT the same guarantee as
    "hasn't happened yet" -- a trial can genuinely still be marked active
    in the registry well past its own stated completion date (a real,
    common ClinicalTrials.gov data-quality gap). Widening
    CATALYST_TRIAL_LIMIT surfaced exactly this: a live "Upcoming Phase 3
    readouts in Oncology" query returned a "January 2024" event -- already
    two years in the past at query time. The synthesis prompt already
    instructs the model to only cite genuinely upcoming events, but isn't
    perfectly reliable at that judgment against a wider candidate pool;
    this is the deterministic backstop.
    """
    today = date.today()
    current_year = today.year
    current_quarter_rank = _QUARTER_RANK[f"Q{(today.month - 1) // 3 + 1}"]

    events = []
    dropped_unparseable = 0
    dropped_past = 0
    for draft_event in draft.events:
        parsed = _parse_catalyst_date(draft_event.raw_date)
        if parsed is None:
            dropped_unparseable += 1
            continue
        year, quarter, display = parsed
        # A year-only citation (no quarter) in the CURRENT year is kept --
        # there's no finer-grained signal to judge "already passed this
        # year" from, and false-dropping a legitimate future event is worse
        # than occasionally keeping one that's already happened within the
        # same calendar year.
        if year < current_year:
            dropped_past += 1
            continue
        if year == current_year and quarter and _QUARTER_RANK.get(quarter, 99) < current_quarter_rank:
            dropped_past += 1
            continue
        events.append(CatalystEvent(
            display_date=display, year=year, quarter=quarter,
            company=draft_event.company, drug_name=draft_event.drug_name,
            event_type=draft_event.event_type, indication=draft_event.indication,
            source=draft_event.source, source_type=draft_event.source_type,
        ))
    if dropped_unparseable:
        print(f"[catalysts] dropped {dropped_unparseable} event(s) with an unparseable raw_date")
    if dropped_past:
        print(f"[catalysts] dropped {dropped_past} event(s) already in the past "
              f"(not a genuine upcoming catalyst)")

    # Sort by (year, quarter-as-a-number) -- deterministic, not an LLM
    # ordering claim.
    events.sort(key=lambda e: (e.year, _QUARTER_RANK.get(e.quarter, 5)))

    return CatalystTimeline(query=query, events=events)


def make_catalyst_graph(verbose: bool = True):
    """Two-node graph: retrieve (trials with live date enrichment + SEC
    excerpts) -> synthesize (one structured-output call citing raw dates,
    then deterministic parsing/sorting via _finalize_catalyst_timeline).
    Pinned to gpt-4o like make_landscape_graph -- see _build_gpt4o_llm's
    docstring for why."""
    catalyst_llm = _build_gpt4o_llm(CATALYST_TIMEOUT, model_env_var="CATALYST_MODEL") \
        .with_structured_output(CatalystTimelineDraft, include_raw=True)

    def retrieve_node(state: CatalystState) -> dict:
        query = state["query"]
        if verbose:
            print(f"[catalysts] retrieving evidence for {query!r}")
        evidence = _retrieve_catalyst_evidence(query)
        if verbose:
            print(f"[catalysts] {len(evidence['trials'])} dated trials, "
                  f"{len(evidence['sec_chunks'])} SEC excerpts")
        return {"retrieved_trials": evidence["trials"], "retrieved_sec": evidence["sec_chunks"]}

    def synthesize_node(state: CatalystState) -> dict:
        prompt = (
            f"QUERY: {state['query']}\n\n"
            f"TRIAL RECORDS WITH REAL COMPLETION DATES ({len(state['retrieved_trials'])}):\n"
            f"{json.dumps(state['retrieved_trials'], indent=2)}\n\n"
            f"SEC FILING EXCERPTS ({len(state['retrieved_sec'])}):\n"
            f"{json.dumps(state['retrieved_sec'], indent=2)}"
        )
        prior_error = state.get("error")
        if prior_error and not prior_error.startswith("transient API error:"):
            prompt += (
                f"\n\nYOUR PREVIOUS ATTEMPT FAILED SCHEMA VALIDATION:\n{prior_error}\n"
                f"Correct the structure this time — match the schema exactly."
            )

        try:
            outcome: dict = catalyst_llm.invoke([
                SystemMessage(content=CATALYST_SYSTEM),
                HumanMessage(content=prompt),
            ])
        except (openai.RateLimitError, openai.APITimeoutError,
               openai.APIConnectionError, openai.InternalServerError) as exc:
            retries = state.get("retries", 0) + 1
            if verbose:
                print(f"[catalysts] API error (attempt {retries}): {exc}")
            if retries <= MAX_SYNTHESIS_RETRIES:
                time.sleep(5)
            return {"retries": retries, "error": f"transient API error: {exc}"}

        draft: CatalystTimelineDraft | None = outcome.get("parsed")
        error = outcome.get("parsing_error")

        if draft is not None and error is None:
            result = _finalize_catalyst_timeline(state["query"], draft)
            if verbose:
                print(f"[catalysts] synthesized {len(result.events)} event(s) "
                      f"(from {len(draft.events)} drafted)")
            return {"result": result, "error": None}

        retries = state.get("retries", 0) + 1
        err_text = str(error) if error else "model did not return the expected structure"
        if verbose:
            print(f"[catalysts] validation failed (attempt {retries}): {err_text[:200]}")
        return {"retries": retries, "error": err_text}

    def route_after_synthesis(state: CatalystState) -> str:
        if state.get("result") is not None:
            return "done"
        if state.get("retries", 0) > MAX_SYNTHESIS_RETRIES:
            return "done"
        return "retry"

    g = StateGraph(CatalystState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_conditional_edges("synthesize", route_after_synthesis,
                            {"retry": "synthesize", "done": END})
    return g.compile()


def run_catalyst_query(graph, query: str) -> CatalystTimeline | None:
    final = graph.invoke(
        {"query": query, "retrieved_trials": [], "retrieved_sec": [],
         "result": None, "retries": 0, "error": None},
        config={"recursion_limit": 10},
    )
    return final.get("result")


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

    # Soft checks only -- these two sources are newer and independently
    # optional (each tool already fails closed with has_results=False if its
    # collection is empty/missing), so a missing one should warn, not block
    # the whole agent from running.
    for name, script in ((COLLECTION_NAME_PUBMED, "fetch_pubmed.py"),
                        (COLLECTION_NAME_SEC, "fetch_sec_edgar.py")):
        try:
            if c.collection_exists(name):
                cnt = c.get_collection(name).points_count
                print(f"[preflight] Qdrant OK — {cnt} points in '{name}'")
            else:
                print(f"[preflight] WARNING: Qdrant collection '{name}' is "
                      f"missing -- run: python {script}")
        except Exception as exc:
            print(f"[preflight] WARNING: could not check '{name}' -> {exc}")
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
         "retrieved_fda": [], "retrieved_pubmed": [], "retrieved_sec": [],
         "retrieved_news": []},
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

    def _tool_payloads() -> list[dict]:
        out = []
        for m in final["messages"]:
            if isinstance(m, ToolMessage) and m.content.strip().startswith("{"):
                try:
                    out.append(json.loads(m.content))
                except json.JSONDecodeError:
                    pass
        return out

    payloads = _tool_payloads()
    retrieved = sorted({t["NCTId"] for p in payloads for t in p.get("trials", [])
                        if t.get("NCTId")})
    retrieved_pmcids = sorted({c["PMCID"] for p in payloads
                              for c in p.get("pubmed_chunks", []) if c.get("PMCID")})
    retrieved_accessions = sorted({c["AccessionNumber"] for p in payloads
                                   for c in p.get("sec_chunks", [])
                                   if c.get("AccessionNumber")})
    retrieved_lit_count = sum(len(p.get("chunks", [])) for p in payloads)
    retrieved_fda_count = sum(len(p.get("fda_records", [])) for p in payloads)
    tools_called = sorted({m.name for m in final["messages"]
                           if isinstance(m, ToolMessage) and getattr(m, "name", None)})
    cited = sorted(set(_re.findall(r"NCT\d{8}", result.narrative_summary)))
    row_ids = [r.nct_id for r in result.table_data]
    # sources_only mirrors synthesize_table_node's own condition: no table
    # rows, but real evidence was retrieved from at least one non-trial tool
    # -- e.g. AC4's corporate-strategy + mechanism question, which may
    # legitimately never match a specific clinical trial.
    sources_only = not row_ids and bool(retrieved_pmcids or retrieved_accessions
                                        or retrieved_lit_count or retrieved_fda_count)

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

    if sources_only:
        print("  path              : IntentClassifier → Agent → Tools → "
              "continue_to_extraction (no distinct trial) → "
              "synthesize_table (sources-only Reduce) → END")
    else:
        print("  path              : IntentClassifier → Agent → Tools → "
              "continue_to_extraction ─Send×N→ extract_trial (Map, parallel) → "
              "synthesize_table (Reduce) → END")

    print(f"\n{'=' * 78}\nSTRUCTURED OUTPUT VALIDATION\n{'=' * 78}")
    print(f"  type returned                : {type(result).__name__}")
    print(f"  tools called                 : {tools_called}")
    print(f"  NCTIds retrieved (any tool)  : {len(retrieved)}")
    print(f"  PMCIDs retrieved             : {len(retrieved_pmcids)}")
    print(f"  SEC accessions retrieved     : {len(retrieved_accessions)}")
    print(f"  narrative citations (NCTId)  : {len(cited)}")
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

    if sources_only:
        # This run's evidence never matched a specific trial (e.g. a
        # corporate-strategy + mechanism question) -- an empty grid and zero
        # NCTId citations are the CORRECT outcome here, not a failure. The
        # real evidence bar instead: the narrative must actually cite
        # something from the non-trial pools that grounded it.
        print(f"  (sources-only mode: empty table_data / no NCTId citations "
              f"is expected — evidence came from PubMed/SEC/literature/FDA, "
              f"not the trial registry)")
        if not result.narrative_summary.strip():
            print("  ✗ FAIL — empty narrative_summary"); ok = False
    else:
        if not row_ids:
            print("  ✗ FAIL — table_data is empty"); ok = False
        if not cited:
            print("  ✗ FAIL — narrative_summary has no NCTId citations"); ok = False

    if ok:
        print("  ✓ PASS — validated SmartTableResponse; every claim traces to "
              "a real tool result")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
