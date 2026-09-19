"""Unit tests for constraint exactness -- no services, no LLMs.

Encodes the second round of exactness guarantees: a trial that merely
MENTIONS the asked drug (prior therapy, comparator context) is not a
trial OF that drug; combination asks require every drug to be studied;
phase/status/sponsor qualifiers are hard filters, never ranking hints.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent import (  # noqa: E402
    QueryConstraints,
    _alias_pattern,
    _apply_query_constraints,
    _combination_required,
    _entity_match_tier,
    _entity_matches_trial,
    _status_set,
    _studied_intervention_names,
)


def _pats(*names):
    return [_alias_pattern(n) for n in names]


def _trial(**kw):
    base = {
        "NCTId": "NCT00000001", "BriefTitle": "A study of drug X in NSCLC",
        "Phase": ["Phase 3"], "OverallStatus": "RECRUITING",
        "LeadSponsorName": "Acme Pharma",
        "conditions": ["Non-small Cell Lung Cancer"],
        "interventions": [{"type": "DRUG", "name": "Drug X"},
                          {"type": "DRUG", "name": "Placebo"}],
        "BriefSummary": "Patients previously treated with pembrolizumab "
                        "receive Drug X.",
    }
    base.update(kw)
    return base


class TestStudiedVsMentioned:
    def test_mention_in_summary_is_not_studied(self):
        t = _trial()
        assert _entity_match_tier(_pats("pembrolizumab"), t) == "mentioned"
        assert not _entity_matches_trial(_pats("pembrolizumab"), t)
        assert _entity_matches_trial(_pats("pembrolizumab"), t,
                                     studied_only=False)

    def test_studied_intervention_matches(self):
        t = _trial()
        assert _entity_match_tier(_pats("Drug X"), t) == "studied"
        assert _entity_matches_trial(_pats("drug-x"), t)

    def test_title_only_counts(self):
        t = _trial(interventions=[{"type": "DRUG", "name": "MK-3475"}],
                   BriefTitle="Pembrolizumab plus chemo in NSCLC")
        assert _entity_match_tier(_pats("pembrolizumab"), t) == "title"
        assert _entity_matches_trial(_pats("pembrolizumab"), t)

    def test_placebo_arm_is_not_studied(self):
        t = _trial(interventions=[{"type": "DRUG", "name": "Placebo matching AABC"},
                                  {"type": "DRUG", "name": "AABC"}])
        assert _studied_intervention_names(t) == ["AABC"]

    def test_etl_studied_names_preferred(self):
        t = _trial(studiedInterventionNames=["obefazimod", "ABX464"],
                   interventions=[])
        assert _entity_match_tier(_pats("ABX464"), t) == "studied"

    def test_other_names_are_studied(self):
        t = _trial(interventions=[{"type": "DRUG", "name": "MK-3475",
                                   "otherNames": ["pembrolizumab", "Keytruda"]}])
        assert _entity_match_tier(_pats("Keytruda"), t) == "studied"

    def test_company_checks_sponsor_and_collaborators(self):
        t = _trial(collaborators=["Merck Sharp & Dohme LLC"])
        assert _entity_match_tier(_pats("Merck"), t, "company") == "studied"
        assert _entity_match_tier(_pats("Pfizer"), t, "company") is None

    def test_procedure_interventions_ignored(self):
        t = _trial(interventions=[{"type": "PROCEDURE", "name": "CT scan"},
                                  {"type": "DRUG", "name": "Drug X"}])
        assert _studied_intervention_names(t) == ["Drug X"]


class TestCombination:
    def test_flag_wins(self):
        assert _combination_required("anything", {"combination_required": True}, 1)

    def test_wording_with_two_drugs(self):
        assert _combination_required("pembrolizumab plus lenvatinib trials", {}, 2)
        assert _combination_required("X combined with Y", {}, 2)
        assert _combination_required("X + Y in RCC", {}, 2)

    def test_single_drug_or_no_wording(self):
        assert not _combination_required("pembrolizumab plus chemo", {}, 1)
        assert not _combination_required("pembrolizumab or nivolumab", {}, 2)


class TestStructuredFilters:
    def test_phase_filter(self):
        pool = [_trial(Phase=["Phase 3"]), _trial(Phase=["Phase 2"]),
                _trial(Phase=["Phase 2", "Phase 3"])]
        kept, enforced, _ = _apply_query_constraints(pool, {"phases": ["phase3"]})
        assert len(kept) == 2 and enforced == ["phase"]

    def test_status_live_alias(self):
        assert _status_set(["live"]) >= {"RECRUITING", "ACTIVE_NOT_RECRUITING"}
        pool = [_trial(OverallStatus="RECRUITING"),
                _trial(OverallStatus="COMPLETED")]
        kept, enforced, _ = _apply_query_constraints(pool, {"statuses": ["ongoing"]})
        assert [t["OverallStatus"] for t in kept] == ["RECRUITING"]

    def test_status_exact(self):
        pool = [_trial(OverallStatus="RECRUITING"),
                _trial(OverallStatus="COMPLETED")]
        kept, _, _ = _apply_query_constraints(pool, {"statuses": ["Completed"]})
        assert [t["OverallStatus"] for t in kept] == ["COMPLETED"]

    def test_sponsor_filter_boundary(self):
        pool = [_trial(LeadSponsorName="Merck Sharp & Dohme LLC"),
                _trial(LeadSponsorName="Merck KGaA"),
                _trial(LeadSponsorName="Pfizer")]
        kept, enforced, _ = _apply_query_constraints(
            pool, {"sponsor_names": ["Merck"]})
        assert len(kept) == 2 and "sponsor" in enforced

    def test_year_unenforceable_without_dates(self):
        pool = [_trial()]
        kept, enforced, unenf = _apply_query_constraints(
            pool, {"start_year_min": 2023})
        assert kept == pool and "start_year" in unenf and not enforced

    def test_year_enforced_with_dates(self):
        pool = [_trial(StartDate="2024-03"), _trial(StartDate="2019-01-15")]
        kept, enforced, _ = _apply_query_constraints(
            pool, {"start_year_min": 2023})
        assert [t["StartDate"] for t in kept] == ["2024-03"]
        assert enforced == ["start_year"]

    def test_country_filter(self):
        pool = [_trial(countries=["United States", "Canada"]),
                _trial(countries=["Japan"])]
        kept, enforced, _ = _apply_query_constraints(
            pool, {"countries": ["united states"]})
        assert len(kept) == 1 and enforced == ["country"]

    def test_empty_constraints_noop(self):
        pool = [_trial(), _trial()]
        kept, enforced, unenf = _apply_query_constraints(
            pool, QueryConstraints().model_dump())
        assert kept == pool and not enforced and not unenf


class TestSchemaDefaults:
    def test_constraints_default_empty(self):
        c = QueryConstraints()
        assert c.phases == [] and c.combination_required is False
        assert c.setting is None
