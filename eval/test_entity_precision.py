"""Entity exactness + completeness gates over eval/golden_entities.jsonl.

Layer 1 (this file, no LLM): for every mined lookalike case,
fetch_trials_exact on the expanded aliases must achieve
  precision = 1.0  -- ZERO forbidden (lookalike's) NCTs in the result, and
  recall    = 1.0  -- every expected NCT present (all cases are small-N,
                      far under the runtime cap).
Brand/generic cases assert the alias walk bridges brand -> generic.

Needs live Neo4j + Qdrant (same env as the app); skips cleanly when the
golden file is absent or the services are down, so the pure-unit suite
(test_entity_matching.py) still runs anywhere.

Layers 2 (classifier entity extraction) and 3 (end-to-end zero-forbidden
tables) call LLMs; they run only when RUN_LLM_EVALS=1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from research_agent import (  # noqa: E402
    POOL_MAX_TRIALS,
    _expand_entity_aliases,
    fetch_trials_exact,
)

GOLDEN = Path(__file__).resolve().parent / "golden_entities.jsonl"


def _cases(kind: str) -> list[dict]:
    if not GOLDEN.exists():
        return []
    with GOLDEN.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("case_type") == kind]


def _services_up() -> bool:
    try:
        from neo4j import GraphDatabase

        from research_agent import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
        d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        d.verify_connectivity()
        d.close()
        return True
    except Exception:  # noqa: BLE001
        return False


LOOKALIKE = _cases("lookalike")
BRAND = _cases("brand_generic")

pytestmark = pytest.mark.skipif(
    not GOLDEN.exists() or not _services_up(),
    reason="golden_entities.jsonl missing or Neo4j unreachable",
)


@pytest.mark.parametrize(
    "case", LOOKALIKE,
    ids=[c["query_entity"] for c in LOOKALIKE] or None)
def test_lookalike_precision_and_recall(case: dict):
    aliases = _expand_entity_aliases(case["query_entity"], case["kind"])
    trials, total, ncts = fetch_trials_exact(
        aliases, case["kind"], cap=POOL_MAX_TRIALS)

    leaked = ncts & set(case["forbidden_nct_ids"])
    assert not leaked, (
        f"{case['query_entity']!r} leaked lookalike "
        f"{case['lookalike_of']!r} trials: {sorted(leaked)}")

    missing = set(case["expected_nct_ids"]) - ncts
    assert not missing, (
        f"{case['query_entity']!r} missing its own trials: {sorted(missing)}")

    assert total >= len(case["expected_nct_ids"])


@pytest.mark.parametrize(
    "case", BRAND, ids=[c["query_entity"] for c in BRAND] or None)
def test_brand_generic_aliasing(case: dict):
    aliases = [a.lower() for a in
               _expand_entity_aliases(case["query_entity"], case["kind"])]
    assert case["alias_must_include"].lower() in aliases, (
        f"{case['query_entity']!r} did not expand to "
        f"{case['alias_must_include']!r}: {aliases}")


@pytest.mark.skipif(os.getenv("RUN_LLM_EVALS") != "1",
                    reason="LLM evals opt-in via RUN_LLM_EVALS=1")
class TestClassifierEntityExtraction:
    """Layer 2: the intent classifier must emit entities for named products
    and NOTHING for class/target queries (else the whole answer gets
    wrongly hard-filtered)."""

    @pytest.fixture(scope="class")
    def classify(self):
        from langchain_core.messages import HumanMessage, SystemMessage

        from research_agent import (INTENT_MODEL, INTENT_SYSTEM,
                                    IntentClassification, build_intent_llm)
        llm = build_intent_llm(INTENT_MODEL)

        def _run(q: str) -> IntentClassification:
            return llm.invoke([SystemMessage(content=INTENT_SYSTEM),
                               HumanMessage(content=q)])
        return _run

    @pytest.mark.parametrize("case", _cases("class_query"),
                             ids=lambda c: c["query"][:40])
    def test_class_queries_emit_no_entities(self, classify, case):
        verdict = classify(case["query"])
        assert not verdict.exact_entities, (
            f"class query wrongly extracted entities: "
            f"{[e.name for e in verdict.exact_entities]}")

    @pytest.mark.parametrize("case", LOOKALIKE[:5],
                             ids=[c["query_entity"] for c in LOOKALIKE[:5]] or None)
    def test_named_drugs_are_extracted(self, classify, case):
        verdict = classify(
            f"Show me all clinical trials of {case['query_entity']}")
        names = [e.name for e in (verdict.exact_entities or [])]
        assert case["query_entity"] in names, (
            f"named drug not extracted: got {names}")
