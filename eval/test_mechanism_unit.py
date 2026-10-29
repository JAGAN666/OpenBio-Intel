"""Unit tests for the deterministic mechanism contract -- no services.

The provenance tier on a Mechanism cell is a promise to the analyst:
'kb' and 'trial_text' mean verified, 'model_knowledge' means the model
said so. These tests pin the rules that keep that promise honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent import (  # noqa: E402
    TrialRow,
    _evidence_in_source,
    _finalize_mechanism,
    _norm_ws,
    _propagate_mechanisms,
)


def _row(**kw) -> TrialRow:
    base = dict(nct_id="NCT00000001", sponsor="Acme", phase="Phase 2",
                interventions=["XYZ-101"], mechanism="", mechanism_source="unknown",
                mechanism_evidence="", mechanism_or_findings="Setting: x vs y — z.",
                mechanism_described=False)
    base.update(kw)
    return TrialRow(**base)


KB = {"Pembrolizumab 200 mg": {"pref_name": "Pembrolizumab",
                               "mechanism": "PD-1 inhibitor (PDCD1)",
                               "ref_url": "https://platform.opentargets.org/"}}
SOURCE = _norm_ws('TRIAL RECORD: {"description": "XYZ-101 is a humanized '
                  'anti-PD-1 monoclonal antibody that blocks PD-1"}')


class TestFinalize:
    def test_kb_overrides_model(self):
        r = _row(interventions=["Pembrolizumab 200 mg"], mechanism="something else",
                 mechanism_source="model_knowledge")
        _finalize_mechanism(r, ["Pembrolizumab 200 mg"], KB, SOURCE)
        assert r.mechanism == "PD-1 inhibitor (PDCD1)"
        assert r.mechanism_source == "kb" and r.mechanism_described

    def test_trial_text_requires_verbatim_span(self):
        r = _row(mechanism="anti-PD-1 antibody", mechanism_source="trial_text",
                 mechanism_evidence="humanized anti-PD-1 monoclonal antibody")
        _finalize_mechanism(r, ["XYZ-101"], {}, SOURCE)
        assert r.mechanism_source == "trial_text" and r.mechanism_described

    def test_pool_quote_relabelled_to_literature(self):
        pools = _norm_ws('LITERATURE: "XYZ-101, a first-in-class TLR7 agonist, showed..."')
        r = _row(mechanism="TLR7 agonist", mechanism_source="trial_text",
                 mechanism_evidence="a first-in-class TLR7 agonist")
        _finalize_mechanism(r, ["XYZ-101"], {}, SOURCE, pools)
        assert r.mechanism_source == "literature" and r.mechanism_described

    def test_record_quote_relabelled_to_trial_text(self):
        r = _row(mechanism="anti-PD-1 antibody", mechanism_source="literature",
                 mechanism_evidence="humanized anti-PD-1 monoclonal antibody")
        _finalize_mechanism(r, ["XYZ-101"], {}, SOURCE, "")
        assert r.mechanism_source == "trial_text"

    def test_paraphrased_span_downgrades(self):
        r = _row(mechanism="anti-PD-1 antibody", mechanism_source="trial_text",
                 mechanism_evidence="an antibody against PD-1 (humanised)")
        _finalize_mechanism(r, ["XYZ-101"], {}, SOURCE)
        # dev code + unverifiable claim -> refused entirely
        assert r.mechanism == "" and r.mechanism_source == "unknown"

    def test_model_knowledge_refused_for_dev_codes(self):
        r = _row(mechanism="KRAS G12C inhibitor", mechanism_source="model_knowledge")
        _finalize_mechanism(r, ["ABC-1234"], {}, SOURCE)
        assert r.mechanism == "" and r.mechanism_source == "unknown"

    def test_model_knowledge_allowed_for_named_drug(self):
        r = _row(interventions=["cetuximab"], mechanism="EGFR antibody",
                 mechanism_source="model_knowledge")
        _finalize_mechanism(r, ["cetuximab"], {}, SOURCE)
        assert r.mechanism == "EGFR antibody"
        assert r.mechanism_source == "model_knowledge" and not r.mechanism_described

    def test_combination_takes_weakest_tier(self):
        r = _row(interventions=["Pembrolizumab 200 mg", "cetuximab"],
                 mechanism="EGFR antibody", mechanism_source="model_knowledge")
        _finalize_mechanism(r, ["Pembrolizumab 200 mg", "cetuximab"], KB, SOURCE)
        assert r.mechanism.startswith("Pembrolizumab: PD-1 inhibitor (PDCD1); cetuximab: EGFR antibody")
        assert r.mechanism_source == "model_knowledge"

    def test_partial_kb_coverage_flagged(self):
        r = _row(interventions=["Pembrolizumab 200 mg", "ABC-1234"])
        _finalize_mechanism(r, ["Pembrolizumab 200 mg", "ABC-1234"], KB, SOURCE)
        assert r.mechanism_source == "kb"
        assert "no mechanism on record for ABC-1234" in r.mechanism

    def test_two_arms_one_substance(self):
        kb = {"BMS-986278 Dose A": {"pref_name": "BMS-986278", "mechanism": "LPA1 antagonist", "ref_url": "u"},
              "BMS-986278 Dose B": {"pref_name": "BMS-986278", "mechanism": "LPA1 antagonist", "ref_url": "u"}}
        r = _row(interventions=["BMS-986278 Dose A", "BMS-986278 Dose B"])
        _finalize_mechanism(r, list(kb), kb, SOURCE)
        assert r.mechanism == "LPA1 antagonist"

    def test_unknown_source_value_normalised(self):
        r = _row(mechanism="", mechanism_source="banana")
        _finalize_mechanism(r, ["XYZ-101"], {}, SOURCE)
        assert r.mechanism_source == "unknown"


class TestEvidence:
    def test_whitespace_case_insensitive(self):
        assert _evidence_in_source("Humanized  anti-PD-1 Monoclonal", SOURCE)

    def test_too_short_rejected(self):
        assert not _evidence_in_source("PD-1", SOURCE)


class TestPropagation:
    def test_same_drug_inherits_verified_mechanism(self):
        a = _row(nct_id="NCT1", interventions=["XYZ-101"], mechanism="anti-PD-1 antibody",
                 mechanism_source="trial_text", mechanism_evidence="humanized anti-PD-1",
                 mechanism_described=True)
        b = _row(nct_id="NCT2", interventions=["XYZ-101 50 mg"])
        c = _row(nct_id="NCT3", interventions=["OTHER-9"])
        assert _propagate_mechanisms([a, b, c]) == 1
        assert b.mechanism == "anti-PD-1 antibody" and b.mechanism_source == "trial_text"
        assert b.mechanism_evidence.startswith("[NCT1]")
        assert c.mechanism == ""

    def test_combination_rows_not_used_as_source(self):
        a = _row(nct_id="NCT1", interventions=["A", "B"], mechanism="A: x; B: y",
                 mechanism_source="trial_text", mechanism_described=True)
        b = _row(nct_id="NCT2", interventions=["A"])
        assert _propagate_mechanisms([a, b]) == 0
