"""Biomedical knowledge graph builder: Entity Resolution -> Neo4j.

Maps the raw drug names in our trial JSON corpus to standard RxNorm
concepts via SciSpaCy, and writes the result as a graph:

    (Trial {id: NCTId}) -[:INVESTIGATES]-> (Drug {name})
                                              -[:MAPPED_TO_RXNORM]-> (Concept {cui, standard_name})

WHY THIS SCRIPT RUNS IN ITS OWN VENV (.venv-kg, Python 3.11)
    scispacy's own PyPI metadata pins `spacy<3.8,>=3.7.0`, and its RxNorm
    linker depends on `nmslib-metabrainz` -- verified directly (`pip
    download --only-binary=:all: --python-version 3.14`) that
    nmslib-metabrainz ships NO wheel at all for cp314, and the spacy pin
    alone would force a from-source build of blis that fails to compile
    against Python 3.14's C API. This is not a version we pinned -- it's
    scispacy's own current ceiling, and Python 3.14 is new enough that nmslib
    (a native similarity-search library) hasn't caught up. Rather than
    downgrade the ENTIRE project (and re-validate every other pinned
    dependency -- qdrant-client==1.18.0, langgraph, etc. -- against an older
    Python), entity resolution is isolated to its own venv: it's an
    ingestion-time batch job, not something research_agent.py needs at
    request time. The agent only needs the `neo4j` driver (works fine on
    3.14) to query the graph THIS script already built.

        python3.11 -m venv .venv-kg
        .venv-kg/bin/pip install scispacy neo4j
        .venv-kg/bin/pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
        .venv-kg/bin/python build_kg.py

WHY RxNav IS ALSO USED, NOT JUST THE SCISPACY LINKER
    Verified empirically before writing this (see project notes): scispacy's
    packaged RxNorm knowledge base does per-span STRING-SIMILARITY candidate
    lookup, not full UMLS relationship traversal -- linking "Keytruda" and
    "Pembrolizumab" separately returns two DIFFERENT CUIs (C3855203 vs
    C3658706) with empty `aliases` on both entries in this KB snapshot. It
    cannot tell you Keytruda IS a trade name of pembrolizumab on its own.
    The free, public NIH RxNav REST API (no auth) exposes the actual RxNorm
    relationship graph -- /rxcui/{cui}/related.json?tty=BN on an ingredient's
    RxCUI returns its real, official brand names. That enrichment runs HERE,
    once per distinct resolved ingredient concept, and is stored durably on
    the Concept node (`brand_names`) -- so the agent's query-time Cypher can
    match a brand name with a single graph query and no live API call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import spacy
from neo4j import GraphDatabase
from scispacy.linking import EntityLinker  # noqa: F401  (registers scispacy_linker)

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:  # dotenv isn't in requirements-kg -- env vars work too
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

SPACY_MODEL = "en_core_sci_sm"
# Below this, a linker candidate is discarded rather than recorded -- a
# missing MAPPED_TO_RXNORM edge is an honest gap; a low-confidence match
# presented as a real mapping is a worse failure mode (same "gaps over
# unsupported attribution" principle used throughout this project's
# grounding rules).
MIN_LINKER_CONFIDENCE = 0.75

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"


# =============================================================================
# 1. LOAD raw trial JSON (the same Land-stage files fetch_and_embed_trials.py
#    writes to data/raw/ -- this script re-derives its own view of the
#    corpus rather than reading from Qdrant, so it can run independently)
# =============================================================================
def latest_raw_payload() -> Path:
    files = sorted(RAW_DATA_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"[load] no raw JSON files found in {RAW_DATA_DIR} -- "
                         f"run fetch_and_embed_trials.py first")
    return files[-1]


def load_studies(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    studies = [s for p in data.get("pages", []) for s in p.get("studies", [])]
    print(f"[load]    {path.name}: {len(studies)} studies")
    return studies


def drug_interventions(study: dict) -> list[dict]:
    """DRUG-type interventions only -- RxNorm covers drugs, not the
    PROCEDURE-type entries (biospecimen collection, CT scans, ...) that
    ClinicalTrials.gov mixes into the same interventions array."""
    proto = study.get("protocolSection", {})
    raw = proto.get("armsInterventionsModule", {}).get("interventions", []) or []
    return [iv for iv in raw if isinstance(iv, dict) and iv.get("type") == "DRUG"
            and (iv.get("name") or "").strip()]


# =============================================================================
# 2. ENTITY RESOLUTION -- scispacy NER + RxNorm linking, cached per unique
#    raw drug-name string (many trials repeat the same drugs)
# =============================================================================
def build_nlp():
    print(f"[nlp]     loading {SPACY_MODEL} + scispacy_linker(rxnorm)…")
    nlp = spacy.load(SPACY_MODEL)
    nlp.add_pipe("scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "rxnorm"})
    print(f"[nlp]     ready — RxNorm KB size: "
          f"{len(nlp.get_pipe('scispacy_linker').kb.cui_to_entity):,} concepts")
    return nlp


def resolve_drug_names(nlp, names: list[str]) -> dict[str, dict | None]:
    """One NER+linking pass per unique name, via nlp.pipe for batch
    efficiency. Returns {raw_name: {"cui", "standard_name", "score"} | None}
    -- None means no candidate cleared MIN_LINKER_CONFIDENCE."""
    linker = nlp.get_pipe("scispacy_linker")
    resolved: dict[str, dict | None] = {}
    for name, doc in zip(names, nlp.pipe(names)):
        best = None
        for ent in doc.ents:
            for cui, score in ent._.kb_ents:
                if best is None or score > best["score"]:
                    entry = linker.kb.cui_to_entity[cui]
                    best = {"cui": cui, "standard_name": entry.canonical_name,
                            "score": float(score)}
        resolved[name] = best if best and best["score"] >= MIN_LINKER_CONFIDENCE else None
    return resolved


# =============================================================================
# RxNav enrichment -- official RxNorm brand-name relationships, cached per
# distinct resolved ingredient concept (NOT per drug-name string or per
# trial).
#
# IMPORTANT: scispacy's "rxnorm" linker KB is built from a UMLS Metathesaurus
# subset filtered to RxNorm-sourced concepts (its data file is literally
# named umls_rxnorm_2022.jsonl) -- the concept_id it returns is a UMLS CUI
# ("C" + 7 digits, e.g. C3658706), NOT a native RxNorm RXCUI (a plain
# integer, e.g. 1547545). RxNav's /rxcui/{id}/related endpoint only accepts
# RXCUIs. The first version of this function passed the UMLS CUI straight
# through and got a 404 on every single lookup (verified: ALL 63 distinct
# concepts failed identically on a live run, which is what exposed this --
# a real bug, not "these drugs have no brand names"). The fix looks up the
# RXCUI by NAME instead, using the linker's own canonical_name -- the same
# name-based lookup already verified working for "Keytruda" itself.
# =============================================================================
def fetch_brand_names(standard_name: str, session: requests.Session) -> list[str]:
    try:
        rxcui_resp = session.get(f"{RXNAV_BASE}/rxcui.json",
                                 params={"name": standard_name}, timeout=10)
        rxcui_resp.raise_for_status()
        rxcuis = rxcui_resp.json().get("idGroup", {}).get("rxnormId") or []
        if not rxcuis:
            return []  # no RxNorm RXCUI for this name -- not every UMLS/RxNorm
                       # concept has one (e.g. procedure-adjacent entries)

        resp = session.get(f"{RXNAV_BASE}/rxcui/{rxcuis[0]}/related.json",
                           params={"tty": "BN"}, timeout=10)
        resp.raise_for_status()
        groups = resp.json().get("relatedGroup", {}).get("conceptGroup") or []
        names = []
        for g in groups:
            for prop in g.get("conceptProperties") or []:
                if prop.get("name"):
                    names.append(prop["name"])
        return sorted(set(names))
    except Exception as exc:  # RxNav being briefly unreachable shouldn't abort the run
        print(f"[rxnav]   WARNING: brand-name lookup failed for {standard_name!r}: {exc}")
        return []


# =============================================================================
# 3. INGEST into Neo4j
# =============================================================================
MERGE_TRIAL_DRUG_CONCEPT = """
MERGE (t:Trial {id: $nct_id})
SET t.title = $title, t.phase = $phase, t.status = $status,
    t.sponsor = $sponsor, t.study_type = $study_type,
    t.conditions = $conditions, t.summary = $summary,
    t.intervention_names = $intervention_names,
    t.interventions_json = $interventions_json

MERGE (d:Drug {name: $drug_name})
SET d.other_names = $other_names
MERGE (t)-[:INVESTIGATES]->(d)

WITH t, d
WHERE $cui IS NOT NULL
MERGE (c:Concept {cui: $cui})
SET c.standard_name = $standard_name, c.brand_names = $brand_names
MERGE (d)-[:MAPPED_TO_RXNORM]->(c)
"""

# A trial can have DRUG interventions that all failed linking (or none) --
# still record the trial/drug relationship without a dangling WITH clause
# expecting a Concept that will never exist.
MERGE_TRIAL_DRUG_ONLY = """
MERGE (t:Trial {id: $nct_id})
SET t.title = $title, t.phase = $phase, t.status = $status,
    t.sponsor = $sponsor, t.study_type = $study_type,
    t.conditions = $conditions, t.summary = $summary,
    t.intervention_names = $intervention_names,
    t.interventions_json = $interventions_json

MERGE (d:Drug {name: $drug_name})
SET d.other_names = $other_names
MERGE (t)-[:INVESTIGATES]->(d)
"""


# Duplicated from fetch_and_embed_trials.py rather than imported -- that
# module pulls in the main project's dependencies (qdrant-client etc.),
# which are not (and should not be) installed in this script's isolated
# .venv-kg. fetch_and_embed_trials.py keeps its own copy of this same table
# for the identical reason, so this mirrors an established pattern, not a
# new one. Kept in sync manually; it's ClinicalTrials.gov's fixed enum, not
# something that changes often.
PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not Applicable",
}


def trial_params(study: dict, all_intervention_names: list[str],
                 interventions_trimmed: list[dict]) -> dict:
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    raw_phases = proto.get("designModule", {}).get("phases", []) or []
    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle") or "(no title)",
        # A Neo4j list-of-strings property, normalised to the SAME human
        # form ("Phase 3") search_clinical_trials's Qdrant payload uses --
        # not a joined string of raw API enum tokens ("PHASE3"). Both tools'
        # trial dicts need matching shapes: query_knowledge_graph's results
        # flow into the exact same Map-Reduce pipeline (tools_node's generic
        # "trials" accumulation, extract_trial's TrialRow extraction) as
        # search_clinical_trials's, with no shape-specific branching.
        "phase": [PHASE_LABELS.get(p, p) for p in raw_phases],
        "status": proto.get("statusModule", {}).get("overallStatus"),
        "sponsor": proto.get("sponsorCollaboratorsModule", {})
                        .get("leadSponsor", {}).get("name"),
        "study_type": proto.get("designModule", {}).get("studyType"),
        "conditions": proto.get("conditionsModule", {}).get("conditions", []) or [],
        "summary": (proto.get("descriptionModule", {}).get("briefSummary") or "")[:2000],
        "intervention_names": all_intervention_names,
        # Neo4j properties can't hold a nested list-of-maps, so the
        # {type, name} shape search_clinical_trials returns (trimmed the
        # same way fetch_and_embed_trials.py's extract_interventions() trims
        # it -- type + name only, no description/armGroupLabels noise) is
        # JSON-serialised here and deserialised back in
        # query_knowledge_graph, rather than approximated or dropped.
        "interventions_json": json.dumps(interventions_trimmed),
    }


def ingest(driver, studies: list[dict], resolved: dict[str, dict | None],
          brand_names: dict[str, list[str]]) -> dict:
    # NOTE: these count MERGE calls issued (i.e. trial-drug OCCURRENCES
    # processed), not distinct relationships written -- MERGE dedupes, so a
    # drug mentioned in N trials produces only ONE (Drug)-[:MAPPED_TO_RXNORM]
    # edge no matter how many times this loop calls MERGE for it. main()
    # queries Neo4j directly after ingestion for the true graph-level counts
    # rather than trusting these as the final report.
    stats = {"trials": 0, "drug_edges": 0, "concept_edges": 0}
    with driver.session() as session:
        for study in studies:
            nct_id = study.get("protocolSection", {}) \
                          .get("identificationModule", {}).get("nctId")
            if not nct_id:
                continue
            drugs = drug_interventions(study)
            if not drugs:
                continue

            raw_interventions = (study.get("protocolSection", {})
                                      .get("armsInterventionsModule", {})
                                      .get("interventions", []) or [])
            all_names = [iv["name"] for iv in raw_interventions
                        if isinstance(iv, dict) and iv.get("name")]
            # Trimmed to {type, name} -- the exact shape
            # fetch_and_embed_trials.py's extract_interventions() produces
            # and search_clinical_trials returns, so query_knowledge_graph's
            # results are indistinguishable in shape from that tool's.
            interventions_trimmed = [
                {"type": (iv.get("type") or "UNKNOWN").strip(), "name": iv["name"]}
                for iv in raw_interventions
                if isinstance(iv, dict) and (iv.get("name") or "").strip()
            ]
            params = trial_params(study, all_names, interventions_trimmed)
            stats["trials"] += 1

            for iv in drugs:
                name = iv["name"]
                match = resolved.get(name)
                row = dict(params, drug_name=name,
                          other_names=iv.get("otherNames") or [])
                stats["drug_edges"] += 1
                if match:
                    row["cui"] = match["cui"]
                    row["standard_name"] = match["standard_name"]
                    row["brand_names"] = brand_names.get(match["cui"], [])
                    session.run(MERGE_TRIAL_DRUG_CONCEPT, **row)
                    stats["concept_edges"] += 1
                else:
                    session.run(MERGE_TRIAL_DRUG_ONLY, **row)
    return stats


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="raw trial JSON payload (default: latest in data/raw/)")
    parser.add_argument("--wipe", action="store_true",
                        help="delete all existing graph nodes/relationships first")
    args = parser.parse_args()

    started = time.time()
    print("=" * 74)
    print("medical-rag :: biomedical knowledge graph builder")
    print("=" * 74)

    path = args.input or latest_raw_payload()
    studies = load_studies(path)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as exc:
        print(f"[neo4j]   cannot reach {NEO4J_URI} -> {exc}\n"
              f"          docker compose up -d neo4j", file=sys.stderr)
        return 1
    print(f"[neo4j]   connected to {NEO4J_URI}")

    if args.wipe:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[neo4j]   wiped existing graph")

    # --- unique drug-name resolution (one NER+link pass per distinct name) -
    unique_names = sorted({iv["name"] for s in studies for iv in drug_interventions(s)})
    print(f"[resolve] {len(unique_names)} distinct drug name(s) across "
          f"{len(studies)} trials")

    nlp = build_nlp()
    resolved = resolve_drug_names(nlp, unique_names)
    n_matched = sum(1 for v in resolved.values() if v)
    print(f"[resolve] {n_matched}/{len(unique_names)} resolved to an RxNorm "
          f"concept (score >= {MIN_LINKER_CONFIDENCE})")
    for name, match in resolved.items():
        if match:
            print(f"    • {name!r:<40} -> CUI {match['cui']}  "
                  f"{match['standard_name']!r}  (score={match['score']:.3f})")
        else:
            print(f"    • {name!r:<40} -> no confident RxNorm match")

    # --- brand-name enrichment, once per distinct resolved CUI -------------
    # dict keyed by CUI (what ingest() looks up by), fetched by canonical
    # NAME (see fetch_brand_names docstring for why: the linker's CUI is a
    # UMLS id, RxNav's related-concepts endpoint needs an RxNorm RXCUI).
    cui_to_name = {m["cui"]: m["standard_name"] for m in resolved.values() if m}
    print(f"[rxnav]   fetching brand names for {len(cui_to_name)} distinct concept(s)…")
    brand_names: dict[str, list[str]] = {}
    with requests.Session() as http:
        for cui, std_name in sorted(cui_to_name.items()):
            names = fetch_brand_names(std_name, http)
            brand_names[cui] = names
            print(f"    • {std_name!r} (CUI {cui}) -> brand names: {names}")

    # --- ingest --------------------------------------------------------------
    stats = ingest(driver, studies, resolved, brand_names)

    # Confirm what's ACTUALLY stored, not just what the write loop attempted
    # -- MERGE dedupes, so stats['concept_edges'] above counts (trial, drug)
    # occurrences processed, not distinct relationships in the graph (a drug
    # mentioned in N trials still produces exactly one MAPPED_TO_RXNORM
    # edge). This mirrors fetch_and_embed_trials.py's own final check
    # (client.get_collection(...).points_count) rather than trusting an
    # in-process counter.
    with driver.session() as session:
        graph_counts = session.run("""
            OPTIONAL MATCH (t:Trial) WITH count(DISTINCT t) AS trials
            OPTIONAL MATCH (d:Drug) WITH trials, count(DISTINCT d) AS drugs
            OPTIONAL MATCH (c:Concept) WITH trials, drugs, count(DISTINCT c) AS concepts
            OPTIONAL MATCH ()-[i:INVESTIGATES]->() WITH trials, drugs, concepts, count(i) AS investigates
            OPTIONAL MATCH ()-[m:MAPPED_TO_RXNORM]->()
            RETURN trials, drugs, concepts, investigates, count(m) AS mapped
        """).single()
    driver.close()

    elapsed = time.time() - started
    print("-" * 74)
    print(f"trial-drug occurrences processed      : {stats['drug_edges']} "
          f"({stats['concept_edges']} resolved to a concept)")
    print(f"(:Trial) nodes                        : {graph_counts['trials']}")
    print(f"(:Drug) nodes                         : {graph_counts['drugs']}")
    print(f"(:Concept) nodes                      : {graph_counts['concepts']}")
    print(f"(Trial)-[:INVESTIGATES]->(Drug)       : {graph_counts['investigates']}")
    print(f"(Drug)-[:MAPPED_TO_RXNORM]->(Concept) : {graph_counts['mapped']}")
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
