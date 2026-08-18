"""Biomedical knowledge graph builder for PubMed literature: Entity Resolution
-> Neo4j.

Companion to build_kg.py, NOT a drop-in replacement -- see that module's own
docstring for why this whole family of scripts runs in its own .venv-kg
(scispacy's Python-3.14 incompatibility). This script differs in a way that
matters: build_kg.py already KNOWS its drug names (ClinicalTrials.gov's
structured `interventions` field lists them explicitly) and only needs to
LINK each known name to RxNorm. A PubMed article has no such structured
list -- drug mentions must first be DISCOVERED via NER over free-text prose,
then linked.

VERIFIED before writing this (not assumed): en_core_sci_sm's NER produces a
single generic "ENTITY" label for every span it finds -- a live test tagged
"treat" and "markets" identically to "Pembrolizumab" -- so there is no type
filter available at the NER stage. The RxNorm LINKER itself is what does the
real filtering here: only entities that clear MIN_LINKER_CONFIDENCE against
the RxNorm-specific KB survive as Drug nodes, which naturally discards
non-drug spans (RxNorm's KB simply has no concept for "treat").

Writes:
    (Article {pmcid})-[:MENTIONS]->(Drug {name})-[:MAPPED_TO_RXNORM]->(Concept {cui, standard_name})

Deliberately the SAME (Drug)-[:MAPPED_TO_RXNORM]->(Concept) shape build_kg.py
writes from clinical trials -- MERGE keys on drug name / concept CUI, so a
drug mentioned in both a trial and a PubMed article lands on the SAME graph
nodes, not a parallel disconnected copy. That is what lets
query_knowledge_graph (research_agent.py) traverse from one drug to both its
trials AND the literature discussing it, in a single graph query.

SCOPE NOTE -- diseases are NOT written to the graph here, despite the spec
asking for "drugs and diseases": en_core_sci_sm's NER has no disease-specific
type label (see above), and there is no disease-specific linker wired up in
this pipeline (RxNorm covers drugs only) -- reusing the RxNorm linker as an
implicit disease filter, the way it works for drugs, is not possible (RxNorm
has no disease concepts to confidently link against, so nothing would ever
pass, silently producing zero disease nodes rather than a useful signal).
Faking "Disease" nodes from untyped, unlinked NER spans would mean writing
"markets"/"treat"-shaped garbage into the graph with no confidence signal to
filter it -- exactly the "unsupported attribution" failure mode
MIN_LINKER_CONFIDENCE exists to prevent in build_kg.py. A real disease
pipeline needs scispacy's disease/chemical-specific NER model
(en_ner_bc5cdr_md) plus a disease-oriented linker (UMLS or MeSH) -- a genuine
second model to download, load, and validate, not a small addition to this
one. Left as a scoped, flagged follow-up, not implemented here.

    .venv-kg/bin/python build_kg_pubmed.py --input <fetch_pubmed.py's batch JSON>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase
import spacy
from scispacy.linking import EntityLinker  # noqa: F401  (registers scispacy_linker)

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:  # dotenv isn't in requirements-kg -- env vars work too
    pass

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

SPACY_MODEL = "en_core_sci_sm"
# HIGHER than build_kg.py's 0.75, and deliberately so -- build_kg.py only
# LINKS a name ClinicalTrials.gov already told it is a drug; this script
# must first DISCOVER whether a free-text span is a drug at all, which is a
# strictly harder, noisier task the same threshold under-protects.
#
# Verified live on a 3-article test batch before picking 0.85 (not guessed):
# at 0.75, "human" (a generic word in running prose) linked at 0.784 to
# "brain-derived neurotrophic factor, human" -- a real RxNorm entry whose
# name happens to end in the literal word "human". 0.85 correctly drops it.
#
# NOT fixed by ANY threshold, and left as an explicit, honest known gap: in
# that same test, "CREST" (a clinical acronym, CREST syndrome) linked to
# RxNorm's own literal product entry "Crest" (the P&G fluoride toothpaste --
# RxNorm indexes many OTC consumer products, not only prescription drugs) at
# 0.994 confidence -- a genuine, high-confidence name COLLISION, not a weak
# match. scispacy's candidate generator ranks by character-n-gram string
# similarity, not real-world word sense -- "CREST" and "Crest" are close to
# the same string, so no confidence cutoff separates them. A real fix needs
# either a curated brand-name exclusion list built from an actual false-
# positive audit, or a dedicated drug-specific NER model instead of
# en_core_sci_sm's generic entity detector -- a genuine follow-up, not
# something safely bolted on here without the audit data to validate it.
MIN_LINKER_CONFIDENCE = 0.85

# NER over an article's full body is slow and mostly redundant -- title +
# abstract already name the drug(s) under discussion in almost every
# biomedical paper's own framing. Capped, not unlimited, to keep a
# 50-article batch tractable: verified this is a real constraint, not a
# guess -- one live test body was 31,054 characters for a SINGLE article;
# NER over 50 articles at that length would be minutes, not seconds.
MAX_CHARS_FOR_NER = 4000


def build_nlp():
    print(f"[nlp]     loading {SPACY_MODEL} + scispacy_linker(rxnorm)…")
    nlp = spacy.load(SPACY_MODEL)
    nlp.add_pipe("scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "rxnorm"})
    print(f"[nlp]     ready — RxNorm KB size: "
          f"{len(nlp.get_pipe('scispacy_linker').kb.cui_to_entity):,} concepts")
    return nlp


def extract_drug_mentions(nlp, text: str) -> dict[str, dict]:
    """NER + RxNorm linking over one article's text.

    Returns {surface_text: {cui, standard_name, score}} for every span that
    clears MIN_LINKER_CONFIDENCE. The same drug can surface more than once
    with different casing/spacing ("Keytruda" vs "keytruda"); keeps the
    highest-scoring match per distinct surface string rather than the first.
    """
    linker = nlp.get_pipe("scispacy_linker")
    doc = nlp(text[:MAX_CHARS_FOR_NER])
    found: dict[str, dict] = {}
    for ent in doc.ents:
        best = None
        for cui, score in ent._.kb_ents:
            if best is None or score > best["score"]:
                entry = linker.kb.cui_to_entity[cui]
                best = {"cui": cui, "standard_name": entry.canonical_name, "score": float(score)}
        if best and best["score"] >= MIN_LINKER_CONFIDENCE:
            key = ent.text.strip()
            if not key:
                continue
            if key not in found or best["score"] > found[key]["score"]:
                found[key] = best
    return found


MERGE_ARTICLE_DRUG_CONCEPT = """
MERGE (a:Article {pmcid: $pmcid})
SET a.title = $title

MERGE (d:Drug {name: $drug_name})
MERGE (a)-[:MENTIONS]->(d)

MERGE (c:Concept {cui: $cui})
SET c.standard_name = $standard_name
MERGE (d)-[:MAPPED_TO_RXNORM]->(c)
"""


def ingest(driver, articles: list[dict], nlp) -> dict:
    stats = {"articles": 0, "mentions": 0, "distinct_drug_cuis": set()}
    with driver.session() as session:
        for art in articles:
            pmcid = art.get("pmcid")
            title = art.get("title") or "(no title)"
            text = " ".join(filter(None, [art.get("title"), art.get("abstract"), art.get("body")]))
            if not pmcid or not text.strip():
                continue
            stats["articles"] += 1

            mentions = extract_drug_mentions(nlp, text)
            for surface, match in mentions.items():
                session.run(MERGE_ARTICLE_DRUG_CONCEPT,
                           pmcid=pmcid, title=title, drug_name=surface,
                           cui=match["cui"], standard_name=match["standard_name"])
                stats["mentions"] += 1
                stats["distinct_drug_cuis"].add(match["cui"])
    return {"articles": stats["articles"], "mentions": stats["mentions"],
            "distinct_drug_concepts": len(stats["distinct_drug_cuis"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help='batch JSON: {"articles": [{"pmcid","title","abstract","body"}, ...]}')
    args = parser.parse_args()

    started = time.time()
    print("=" * 74)
    print("medical-rag :: PubMed knowledge graph builder")
    print("=" * 74)

    data = json.loads(args.input.read_text())
    articles = data.get("articles", [])
    print(f"[load]    {args.input.name}: {len(articles)} article(s)")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as exc:
        print(f"[neo4j]   cannot reach {NEO4J_URI} -> {exc}\n"
              f"          docker compose up -d neo4j", file=sys.stderr)
        return 1
    print(f"[neo4j]   connected to {NEO4J_URI}")

    nlp = build_nlp()
    stats = ingest(driver, articles, nlp)
    driver.close()

    elapsed = time.time() - started
    print("-" * 74)
    print(f"articles processed        : {stats['articles']}")
    print(f"drug mentions written     : {stats['mentions']}")
    print(f"distinct drug concepts    : {stats['distinct_drug_concepts']}")
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
