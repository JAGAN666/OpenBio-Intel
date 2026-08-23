"""Mechanism coverage / correctness gates over eval/golden_mechanisms.jsonl.

Layer 1 -- the KB tier (no LLM; needs Neo4j with build_drug_kb.py run):
  coverage     = share of approved drugs the KB resolves to a mechanism
  correctness  = share of resolved mechanisms matching the reference regex
  abstention   = unknown / placebo names must NOT get a mechanism
Layer 2 -- end-to-end (RUN_LLM_EVALS=1): for a few investigational agents
  the table's Mechanism cell must be populated with a verified tier and
  every 'trial_text' quote must occur in the trial record.

    uv run pytest eval/test_mechanism.py -q
    RUN_LLM_EVALS=1 uv run pytest eval/test_mechanism.py -q
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GOLDEN = Path(__file__).resolve().parent / "golden_mechanisms.jsonl"
RUN_LLM = os.getenv("RUN_LLM_EVALS") == "1"


def cases() -> list[dict]:
    if not GOLDEN.exists():
        return []
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def kb_facts() -> dict[str, dict]:
    from research_agent import _graph_client, get_drug_mechanisms
    try:
        with _graph_client().session() as s:
            n = s.run("MATCH (s:Substance) RETURN count(s) AS n").single()["n"]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j unavailable: {exc}")
    if not n:
        pytest.skip("drug KB not built (run build_drug_kb.py)")
    return get_drug_mechanisms([c["name"] for c in cases()])


@pytest.mark.skipif(not cases(), reason="golden_mechanisms.jsonl missing")
def test_kb_coverage_and_correctness(kb_facts):
    approved = [c for c in cases() if c["tier"] == "approved"]
    inv = [c for c in cases() if c["tier"] == "investigational"]
    res = {}
    for tier, group in (("approved", approved), ("investigational", inv)):
        covered = [c for c in group if (kb_facts.get(c["name"]) or {}).get("mechanism")]
        correct = [c for c in covered
                   if re.search(c["expect_regex"], kb_facts[c["name"]]["mechanism"], re.I)]
        wrong = [(c["name"], kb_facts[c["name"]]["mechanism"]) for c in covered
                 if c not in correct]
        res[tier] = (len(covered), len(correct), len(group), wrong)
        print(f"{tier:<16} coverage {len(covered)}/{len(group)}  "
              f"correct {len(correct)}/{len(covered)}  wrong={wrong}")
    cov_a, cor_a, n_a, wrong_a = res["approved"]
    assert cov_a / n_a >= 0.85, f"approved coverage {cov_a}/{n_a}"
    assert cor_a / max(1, cov_a) >= 0.9, f"approved correctness: {wrong_a}"
    cov_i, cor_i, n_i, wrong_i = res["investigational"]
    assert cov_i / n_i >= 0.6, f"investigational coverage {cov_i}/{n_i}"
    assert cor_i / max(1, cov_i) >= 0.9, f"investigational correctness: {wrong_i}"


@pytest.mark.skipif(not cases(), reason="golden_mechanisms.jsonl missing")
def test_kb_abstains_on_unknowns(kb_facts):
    for c in cases():
        if c["tier"] == "unknown":
            assert not (kb_facts.get(c["name"]) or {}).get("mechanism"), c["name"]


def test_placebo_names_are_not_studied():
    from research_agent import _studied_intervention_names
    t = {"interventions": [{"type": "DRUG", "name": "Placebo matching BMS-986278"},
                           {"type": "DRUG", "name": "BMS-986278"}]}
    assert _studied_intervention_names(t) == ["BMS-986278"]


@pytest.mark.skipif(not RUN_LLM, reason="set RUN_LLM_EVALS=1")
def test_end_to_end_mechanism_column():
    from langchain_core.messages import HumanMessage
    from research_agent import (DEFAULT_MODEL, _evidence_in_source, _norm_ws,
                                make_graph)
    g = make_graph(DEFAULT_MODEL, verbose=False)
    probes = ["Trials of BMS-986278", "Trials of ABX464", "Trials of NVL-655"]
    for q in probes:
        final = g.invoke(
            {"messages": [HumanMessage(content=q)], "tool_rounds": 0, "result": None,
             "is_in_domain": None, "has_results": None, "synthesis_retries": 0,
             "synthesis_error": None, "retrieved_trials": [], "extracted_rows": [],
             "retrieved_literature": [], "retrieved_fda": [], "retrieved_crls": [],
             "retrieved_safety": [], "retrieved_exclusivity": [], "retrieved_stats": [],
             "asked_entities": [], "asked_constraints": {}, "enforced_constraints": [],
             "prepared_pools": None, "trial_total_matching": None, "coverage_note": None},
            config={"recursion_limit": 30})
        rows = final["result"].table_data
        assert rows, q
        verified = [r for r in rows if r.mechanism_source in ("kb", "trial_text", "literature")]
        print(f"{q}: {len(verified)}/{len(rows)} rows with a verified mechanism")
        assert len(verified) / len(rows) >= 0.8, q
        by_nct = {t["NCTId"]: t for t in final["prepared_pools"]["trials"]}
        for r in rows:
            if r.mechanism_source == "trial_text" and not r.mechanism_evidence.startswith("[NCT"):
                assert _evidence_in_source(r.mechanism_evidence,
                                           _norm_ws(json.dumps(by_nct[r.nct_id]))), r.nct_id
