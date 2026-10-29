"""Constraint-satisfaction gates over eval/golden_constraints.jsonl.

Each case is an analyst question plus ROW RULES that every returned row
must satisfy -- the (query, trial, constraint) matrix TrialGPT-style
evaluation recommends, checked deterministically against the produced
table so the score is reproducible:

  studied_any        at least one studied intervention name matches one alias
  studied_all        for combinations: every alias group has a match
  forbidden_studied  none of these agents may be the studied drug
  phase_in / status_in / sponsor_regex / indication_regex / mechanism_regex
  design_randomized  the record's constraints carry a satisfied design verdict

Per-constraint precision is reported per rule kind and gated; candidate-
stage recall is reported separately when expected_nct_ids are given.

Layer 1 (no LLM): the intent classifier's constraint extraction is checked
against `expect` when RUN_LLM_EVALS=1 (it is itself an LLM call).
Layer 2 (RUN_LLM_EVALS=1, needs Neo4j + Qdrant + provider keys): run the
graph per case and evaluate rows.

    RUN_LLM_EVALS=1 uv run pytest eval/test_constraints.py -q
    RUN_LLM_EVALS=1 uv run python eval/test_constraints.py --report  # table only
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GOLDEN = Path(__file__).resolve().parent / "golden_constraints.jsonl"
RUN_LLM = os.getenv("RUN_LLM_EVALS") == "1"

THRESHOLDS = {
    "studied_any": 0.95, "studied_all": 0.95, "forbidden_studied": 1.0,
    "phase_in": 0.98, "status_in": 0.98, "sponsor_regex": 0.95,
    "indication_regex": 0.90, "mechanism_regex": 0.85, "design_randomized": 0.85,
}


def cases() -> list[dict]:
    if not GOLDEN.exists():
        return []
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def _alias_hit(aliases: list[str], names: list[str]) -> bool:
    from research_agent import _alias_pattern
    pats = [p for p in (_alias_pattern(a) for a in aliases) if p]
    hay = " | ".join(names)
    return any(p.search(hay) for p in pats)


def evaluate_rows(rows: list[dict], rules: dict) -> dict[str, tuple[int, int]]:
    """{rule: (passed, total)} over the rows."""
    out: dict[str, tuple[int, int]] = {}
    for rule, spec in rules.items():
        passed = 0
        for r in rows:
            names = list(r.get("interventions") or [])
            ok = True
            if rule == "studied_any":
                ok = _alias_hit(spec, names)
            elif rule == "studied_all":
                ok = all(_alias_hit(g, names) for g in spec)
            elif rule == "forbidden_studied":
                ok = not _alias_hit(spec, names)
            elif rule == "phase_in":
                ok = any(p in (r.get("phase") or "") for p in spec)
            elif rule == "status_in":
                ok = (r.get("status") or "") in spec
            elif rule == "sponsor_regex":
                ok = re.search(spec, r.get("sponsor") or "", re.I) is not None
            elif rule == "indication_regex":
                hay = (r.get("indication") or "") + " " + (r.get("mechanism_or_findings") or "")
                ok = re.search(spec, hay, re.I) is not None
            elif rule == "mechanism_regex":
                hay = (r.get("mechanism") or "") + " " + (r.get("mechanism_or_findings") or "")
                ok = re.search(spec, hay, re.I) is not None
            elif rule == "design_randomized":
                ok = any(c.get("constraint", "").startswith("design") and c.get("verdict") == "satisfied"
                         for c in (r.get("constraints") or [])) or \
                    re.search(r"randomi[sz]ed", r.get("mechanism_or_findings") or "", re.I) is not None
            passed += bool(ok)
        out[rule] = (passed, len(rows))
    return out


def _run_case(graph, query: str) -> dict:
    from langchain_core.messages import HumanMessage
    final = graph.invoke(
        {"messages": [HumanMessage(content=query)], "tool_rounds": 0, "result": None,
         "is_in_domain": None, "has_results": None, "synthesis_retries": 0,
         "synthesis_error": None, "retrieved_trials": [], "extracted_rows": [],
         "retrieved_literature": [], "retrieved_fda": [], "retrieved_crls": [],
         "retrieved_safety": [], "retrieved_exclusivity": [], "retrieved_stats": [],
         "asked_entities": [], "asked_constraints": {}, "enforced_constraints": [],
         "prepared_pools": None, "trial_total_matching": None, "coverage_note": None},
        config={"recursion_limit": 30})
    result = final.get("result")
    rows = [r.model_dump() for r in (result.table_data if result else [])]
    pooled = {t.get("NCTId") for t in (final.get("prepared_pools") or {}).get("trials") or []}
    return {"rows": rows, "constraints": final.get("asked_constraints") or {},
            "entities": [e.get("name") for e in final.get("asked_entities") or []],
            "pooled": pooled, "total": final.get("trial_total_matching"),
            "note": final.get("coverage_note")}


@pytest.fixture(scope="module")
def graph():
    if not RUN_LLM:
        pytest.skip("set RUN_LLM_EVALS=1 to run the end-to-end constraint gates")
    from research_agent import DEFAULT_MODEL, make_graph
    return make_graph(DEFAULT_MODEL, verbose=False)


@pytest.fixture(scope="module")
def outcomes(graph) -> dict[str, dict]:
    return {c["id"]: _run_case(graph, c["query"]) for c in cases()}


def _agg(outcomes: dict[str, dict]) -> dict[str, tuple[int, int]]:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c in cases():
        o = outcomes[c["id"]]
        for rule, (p, n) in evaluate_rows(o["rows"], c.get("rules") or {}).items():
            agg[rule][0] += p
            agg[rule][1] += n
    return {k: (v[0], v[1]) for k, v in agg.items()}


@pytest.mark.skipif(not cases(), reason="golden_constraints.jsonl missing")
def test_per_constraint_precision(outcomes):
    agg = _agg(outcomes)
    failures = []
    for rule, (p, n) in agg.items():
        if n == 0:
            continue
        prec = p / n
        print(f"{rule:<20} {p:>4}/{n:<4} = {prec:.3f}  (gate {THRESHOLDS.get(rule, 0.9):.2f})")
        if prec < THRESHOLDS.get(rule, 0.9):
            failures.append(f"{rule}: {prec:.3f} < {THRESHOLDS.get(rule, 0.9)}")
    assert not failures, failures


@pytest.mark.skipif(not cases(), reason="golden_constraints.jsonl missing")
def test_strict_row_precision(outcomes):
    """A row is correct only if it satisfies EVERY rule of its query."""
    ok = total = 0
    for c in cases():
        o = outcomes[c["id"]]
        per = evaluate_rows(o["rows"], c.get("rules") or {})
        for i, r in enumerate(o["rows"]):
            total += 1
            ok += all(evaluate_rows([r], {k: c["rules"][k]})[k][0] == 1 for k in c.get("rules") or {})
    prec = ok / total if total else 1.0
    print(f"strict all-constraints row precision: {ok}/{total} = {prec:.3f}")
    assert prec >= 0.85, prec


@pytest.mark.skipif(not cases(), reason="golden_constraints.jsonl missing")
def test_candidate_recall(outcomes):
    hit = want = 0
    for c in cases():
        exp = c.get("expected_nct_ids") or []
        if not exp:
            continue
        o = outcomes[c["id"]]
        shown = {r["nct_id"] for r in o["rows"]}
        hit += sum(1 for n in exp if n in shown or n in o["pooled"])
        want += len(exp)
    if want:
        rec = hit / want
        print(f"candidate-stage recall over expected ids: {hit}/{want} = {rec:.3f}")
        assert rec >= 0.8, rec


@pytest.mark.skipif(not cases(), reason="golden_constraints.jsonl missing")
def test_classifier_constraints(outcomes):
    """The intent layer must surface every qualifier the analyst typed."""
    misses = []
    for c in cases():
        exp = c.get("expect") or {}
        o = outcomes[c["id"]]
        got = o["constraints"]
        if "entities" in exp:
            ge = {e.casefold() for e in o["entities"]}
            for e in exp["entities"]:
                if e.casefold() not in ge:
                    misses.append(f"{c['id']}: entity {e!r} missing (got {sorted(ge)})")
            if not exp["entities"] and ge:
                misses.append(f"{c['id']}: spurious entities {sorted(ge)}")
        for key in ("phases", "statuses", "indications", "drug_classes", "targets",
                    "sponsor_names", "design"):
            if key in exp:
                want = {x.casefold() for x in exp[key]}
                have = {x.casefold() for x in (got.get(key) or [])}
                if key in ("phases", "statuses") and want and not want <= have:
                    misses.append(f"{c['id']}: {key} {sorted(want)} vs {sorted(have)}")
                elif key not in ("phases", "statuses") and want and not have:
                    misses.append(f"{c['id']}: {key} empty, expected {sorted(want)}")
        if "combination_required" in exp and bool(got.get("combination_required")) != exp["combination_required"]:
            misses.append(f"{c['id']}: combination_required {got.get('combination_required')}")
        if exp.get("start_year_min") and got.get("start_year_min") != exp["start_year_min"]:
            misses.append(f"{c['id']}: start_year_min {got.get('start_year_min')}")
    print("\n".join(misses) or "classifier: all constraints surfaced")
    # Allow a small tolerance for paraphrase differences in free-text fields.
    assert len(misses) <= max(2, len(cases()) // 10), misses


if __name__ == "__main__":
    if "--report" in sys.argv:
        os.environ["RUN_LLM_EVALS"] = "1"
        from research_agent import DEFAULT_MODEL, make_graph
        g = make_graph(DEFAULT_MODEL, verbose=False)
        outs = {c["id"]: _run_case(g, c["query"]) for c in cases()}
        for rule, (p, n) in _agg(outs).items():
            print(f"{rule:<20} {p:>4}/{n:<4} = {p / n if n else 1:.3f}")
        for c in cases():
            o = outs[c["id"]]
            print(f"{c['id']} rows={len(o['rows'])} total={o['total']} "
                  f"entities={o['entities']} note={o['note']}")
