"""ClinicalTrials.gov v2 record schema shared by EVERY ingest path.

Stdlib-only on purpose: build_kg.py runs in its own isolated venv
(scispacy) and cannot import the main project's modules, yet the three
Qdrant writers (fetch_and_embed_trials.py, seed_bulk_data.py,
fetch_daily_updates.py) and the Neo4j writer must agree byte-for-byte on
which API fields are pulled and how a trial record is shaped. Before this
module each script carried its own copy of a nine-field list, and the
fields that actually carry mechanism text (intervention descriptions,
detailed description) were requested by none of them.

Two derived fields are the load-bearing ones for exactness:

- `studiedInterventionNames`: names (+ otherNames) of the agents a trial
  actually investigates -- drug-like interventions whose arm role is not
  placebo/sham. The entity filter matches against THIS, so a drug that is
  merely mentioned in the summary ("previously treated with X") cannot
  pass as a trial of X.
- `interventionRoles`: {name: studied|comparator|placebo|other}, derived
  from the intervention type, a placebo-name pattern and the arm-group
  types the intervention appears in. Written to the INVESTIGATES edge in
  Neo4j so the graph can answer "trials that STUDY X" exactly.
"""
from __future__ import annotations

import json
import re

API_FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.identificationModule.officialTitle",
    "protocolSection.identificationModule.acronym",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.designInfo",
    "protocolSection.designModule.enrollmentInfo",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.startDateStruct",
    "protocolSection.statusModule.primaryCompletionDateStruct",
    "protocolSection.statusModule.completionDateStruct",
    "protocolSection.statusModule.lastUpdatePostDateStruct",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "protocolSection.sponsorCollaboratorsModule.collaborators",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.descriptionModule.detailedDescription",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.conditionsModule.keywords",
    "protocolSection.armsInterventionsModule.armGroups",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.eligibilityModule.sex",
    "protocolSection.eligibilityModule.minimumAge",
    "protocolSection.eligibilityModule.maximumAge",
    "protocolSection.contactsLocationsModule.locations",
]

PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not Applicable",
}

# Intervention types that can be "the studied agent" of a drug trial.
STUDIED_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT", "GENETIC",
                 "DIETARY_SUPPLEMENT", "UNKNOWN", ""}

PLACEBO_NAME_RE = re.compile(
    r"^(?:placebos?|matching placebo|placebo matching|sham|vehicle|dummy|"
    r"saline|normal saline)(?:[\s,:()-]+.*)?$", re.IGNORECASE)
PLACEBO_ARM_TYPES = {"PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION"}
COMPARATOR_ARM_TYPES = {"ACTIVE_COMPARATOR"}

# Text caps: the Qdrant payload is the display/extraction record, so
# descriptions are kept generously; eligibility is the longest field on
# CT.gov and is clipped to keep points under Qdrant's comfortable size.
DETAILED_DESCRIPTION_MAX = 6000
ELIGIBILITY_MAX = 4000
INTERVENTION_DESCRIPTION_MAX = 1500


def _s(v) -> str:
    return (v or "").strip() if isinstance(v, str) else ""


def extract_arm_groups(proto: dict) -> list[dict]:
    raw = proto.get("armsInterventionsModule", {}).get("armGroups", []) or []
    out = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        out.append({
            "label": _s(a.get("label")),
            "type": _s(a.get("type")).upper(),
            "description": _s(a.get("description"))[:INTERVENTION_DESCRIPTION_MAX],
            "interventionNames": [n for n in (a.get("interventionNames") or [])
                                  if isinstance(n, str)],
        })
    return out


def extract_interventions(proto: dict) -> list[dict]:
    """protocolSection.armsInterventionsModule.interventions ->
    [{type, name, description, otherNames, armGroupLabels}].

    Every level uses .get() with a default: armsInterventionsModule is
    absent on observational studies, and an individual intervention can
    lack `type`. `description` is where sponsors write "XYZ-101 is a
    humanized anti-PD-1 monoclonal antibody" -- the single richest
    mechanism source in the registry, previously discarded.
    """
    raw = proto.get("armsInterventionsModule", {}).get("interventions", []) or []
    out = []
    for iv in raw:
        if not isinstance(iv, dict):
            continue
        name = _s(iv.get("name"))
        if not name:
            continue  # an unnamed intervention is not filterable or citable
        out.append({
            "type": (_s(iv.get("type")) or "UNKNOWN").upper(),
            "name": name,
            "description": _s(iv.get("description"))[:INTERVENTION_DESCRIPTION_MAX],
            "otherNames": [n.strip() for n in (iv.get("otherNames") or [])
                           if isinstance(n, str) and n.strip()],
            "armGroupLabels": [l for l in (iv.get("armGroupLabels") or [])
                               if isinstance(l, str)],
        })
    return out


def intervention_roles(interventions: list[dict],
                       arm_groups: list[dict]) -> dict[str, str]:
    """{intervention name: 'studied' | 'comparator' | 'placebo' | 'other'}.

    An intervention is `placebo` when its name looks like a placebo/sham
    or it appears ONLY in placebo/sham/no-intervention arms; `comparator`
    when it appears only in ACTIVE_COMPARATOR arms; `other` for
    non-drug-like types; otherwise `studied`. Trials without arm-group
    data (common on older records) fall back to type + name only.
    """
    arm_type_by_label = {a["label"]: a["type"] for a in arm_groups if a.get("label")}
    roles: dict[str, str] = {}
    for iv in interventions:
        name, itype = iv["name"], iv.get("type", "UNKNOWN")
        if itype not in STUDIED_TYPES:
            roles[name] = "other"
            continue
        if PLACEBO_NAME_RE.match(name):
            roles[name] = "placebo"
            continue
        arm_types = {arm_type_by_label.get(l, "") for l in (iv.get("armGroupLabels") or [])}
        arm_types.discard("")
        if arm_types and arm_types <= PLACEBO_ARM_TYPES:
            roles[name] = "placebo"
        elif arm_types and arm_types <= COMPARATOR_ARM_TYPES:
            roles[name] = "comparator"
        else:
            roles[name] = "studied"
    return roles


def studied_intervention_names(interventions: list[dict],
                               roles: dict[str, str]) -> list[str]:
    """Studied agents plus their registered synonyms (otherNames), deduped,
    order-preserving. Comparators are deliberately EXCLUDED: an analyst
    asking for 'trials of X' does not mean trials where X is the control."""
    seen: set[str] = set()
    out: list[str] = []
    for iv in interventions:
        if roles.get(iv["name"]) != "studied":
            continue
        for n in [iv["name"]] + list(iv.get("otherNames") or []):
            key = n.casefold()
            if key not in seen:
                seen.add(key)
                out.append(n)
    return out


def _date(struct) -> str | None:
    if isinstance(struct, dict):
        d = _s(struct.get("date"))
        return d or None
    return None


def build_trial_payload(study: dict, s3_key: str | None) -> dict | None:
    """The canonical Qdrant payload for one study, or None when the study
    has no id or no brief summary (nothing to vectorize)."""
    proto = study.get("protocolSection", {}) or {}
    ident = proto.get("identificationModule", {}) or {}
    nct_id = ident.get("nctId")
    if not nct_id:
        return None
    desc = proto.get("descriptionModule", {}) or {}
    summary = _s(desc.get("briefSummary"))
    if not summary:
        return None

    design = proto.get("designModule", {}) or {}
    status = proto.get("statusModule", {}) or {}
    sponsors = proto.get("sponsorCollaboratorsModule", {}) or {}
    elig = proto.get("eligibilityModule", {}) or {}
    conditions = [c for c in (proto.get("conditionsModule", {}) or {})
                  .get("conditions", []) or [] if isinstance(c, str)]
    keywords = [k for k in (proto.get("conditionsModule", {}) or {})
                .get("keywords", []) or [] if isinstance(k, str)]
    interventions = extract_interventions(proto)
    arm_groups = extract_arm_groups(proto)
    roles = intervention_roles(interventions, arm_groups)
    studied = studied_intervention_names(interventions, roles)
    countries: list[str] = []
    for loc in (proto.get("contactsLocationsModule", {}) or {}).get("locations", []) or []:
        c = _s(loc.get("country")) if isinstance(loc, dict) else ""
        if c and c not in countries:
            countries.append(c)
    primary_outcomes = [
        _s(o.get("measure")) for o in
        (proto.get("outcomesModule", {}) or {}).get("primaryOutcomes", []) or []
        if isinstance(o, dict) and _s(o.get("measure"))]
    design_info = design.get("designInfo") or {}
    start = _date(status.get("startDateStruct"))

    return {
        "NCTId": nct_id,
        "BriefTitle": ident.get("briefTitle") or "(no title)",
        "OfficialTitle": _s(ident.get("officialTitle")) or None,
        "Acronym": _s(ident.get("acronym")) or None,
        "Phase": [PHASE_LABELS.get(p, p) for p in (design.get("phases") or [])],
        "OverallStatus": status.get("overallStatus"),
        "LeadSponsorName": (sponsors.get("leadSponsor") or {}).get("name"),
        "collaborators": [_s(c.get("name")) for c in (sponsors.get("collaborators") or [])
                          if isinstance(c, dict) and _s(c.get("name"))],
        "conditions": conditions,
        "keywords": keywords,
        "interventions": interventions,
        "armGroups": arm_groups,
        "interventionRoles": roles,
        "studyType": design.get("studyType"),
        # Flattened names: a list of dicts cannot back a keyword/full-text
        # index; these parallel lists are what exact-match filtering uses.
        "interventionNames": [iv["name"] for iv in interventions],
        "studiedInterventionNames": studied,
        "StartDate": start,
        "StartYear": int(start[:4]) if start and start[:4].isdigit() else None,
        "PrimaryCompletionDate": _date(status.get("primaryCompletionDateStruct")),
        "CompletionDate": _date(status.get("completionDateStruct")),
        "LastUpdateDate": _date(status.get("lastUpdatePostDateStruct")),
        "Enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "designInfo": {
            "allocation": _s(design_info.get("allocation")) or None,
            "interventionModel": _s(design_info.get("interventionModel")) or None,
            "masking": _s((design_info.get("maskingInfo") or {}).get("masking")) or None,
            "primaryPurpose": _s(design_info.get("primaryPurpose")) or None,
        },
        "PrimaryOutcomes": primary_outcomes[:8],
        "countries": countries,
        "EligibilityCriteria": _s(elig.get("eligibilityCriteria"))[:ELIGIBILITY_MAX] or None,
        "Sex": _s(elig.get("sex")) or None,
        "MinimumAge": _s(elig.get("minimumAge")) or None,
        "MaximumAge": _s(elig.get("maximumAge")) or None,
        "BriefSummary": summary,
        "DetailedDescription": _s(desc.get("detailedDescription"))[:DETAILED_DESCRIPTION_MAX] or None,
        "SourceURL": f"https://clinicaltrials.gov/study/{nct_id}",
        "SourceS3Key": s3_key,
    }


def build_embedding_text(payload: dict) -> str:
    """The enriched string handed to the embedding model. Structured lines
    lead, then intervention descriptions (mechanism-bearing), then the
    narrative summary. nomic/OpenAI embedders have 8K-token windows, so
    nothing here is truncated by the tokenizer."""
    iv_lines = []
    for iv in payload.get("interventions") or []:
        line = f"{iv['type']}: {iv['name']}"
        if iv.get("otherNames"):
            line += f" ({', '.join(iv['otherNames'][:4])})"
        if iv.get("description"):
            line += f" -- {iv['description'][:400]}"
        iv_lines.append(line)
    conds = payload.get("conditions") or []
    outcomes = payload.get("PrimaryOutcomes") or []
    return (
        f"Title: {payload.get('BriefTitle') or ''}\n"
        f"Conditions: {', '.join(conds) if conds else 'Not specified'}\n"
        f"Interventions: {'; '.join(iv_lines) if iv_lines else 'Not specified'}\n"
        f"Study Type: {payload.get('studyType') or 'Not specified'}\n"
        f"Primary Outcomes: {'; '.join(outcomes) if outcomes else 'Not specified'}\n"
        f"Summary: {payload.get('BriefSummary') or ''}"
    )


def trial_kg_props(payload: dict) -> dict:
    """Neo4j Trial-node properties derived from the canonical payload.
    Neo4j properties cannot hold nested maps, so structured sub-objects
    are JSON-serialised (and deserialised back in research_agent)."""
    return {
        "nct_id": payload["NCTId"],
        "title": payload.get("BriefTitle") or "(no title)",
        "phase": payload.get("Phase") or [],
        "status": payload.get("OverallStatus"),
        "sponsor": payload.get("LeadSponsorName"),
        "collaborators": payload.get("collaborators") or [],
        "study_type": payload.get("studyType"),
        "conditions": payload.get("conditions") or [],
        "summary": (payload.get("BriefSummary") or "")[:2000],
        "intervention_names": payload.get("interventionNames") or [],
        "studied_intervention_names": payload.get("studiedInterventionNames") or [],
        "interventions_json": json.dumps(payload.get("interventions") or []),
        "arm_groups_json": json.dumps(payload.get("armGroups") or []),
        "start_date": payload.get("StartDate"),
        "start_year": payload.get("StartYear"),
        "primary_completion_date": payload.get("PrimaryCompletionDate"),
        "enrollment": payload.get("Enrollment"),
        "countries": payload.get("countries") or [],
        "design_json": json.dumps(payload.get("designInfo") or {}),
    }
