"""Unit tests for deterministic helpers used by the data and query paths."""

from fetch_exclusivity import _parse_date
from research_agent import _aact_guard, _normalise_phase


def test_parse_date_accepts_supported_formats_and_rejects_invalid_values():
    assert _parse_date("Aug 24, 2026") == "2026-08-24"
    assert _parse_date("08/24/2026") == "2026-08-24"
    assert _parse_date("2026-08-24") == "2026-08-24"
    assert _parse_date("  ") is None
    assert _parse_date("24/08/2026") is None


def test_normalise_phase_accepts_human_numeric_and_separator_variants():
    assert _normalise_phase("Phase 3") == "Phase 3"
    assert _normalise_phase("phase-3") == "Phase 3"
    assert _normalise_phase("PHASE_3") == "Phase 3"
    assert _normalise_phase("3") == "Phase 3"
    assert _normalise_phase(None) is None
    assert _normalise_phase("Phase 0") == "Phase 0"


def test_aact_guard_allows_one_select_and_rejects_non_read_queries():
    assert _aact_guard("SELECT nct_id FROM studies") is None
    assert _aact_guard("SELECT 1; SELECT 2") is not None
    assert _aact_guard("UPDATE studies SET status = 'x'") is not None
    assert _aact_guard("SELECT 1; DROP TABLE studies") is not None
