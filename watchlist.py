"""Watchlists + change detection -- the daily-monitoring layer.

Converts the platform from one-shot reports into a monitoring tool: users
watch entities (drug / company / NCT id / topic), and the check job diffs
two live sources for changes since the last check:

  1. ClinicalTrials.gov v2 API -- trials matching the entity whose
     lastUpdatePostDate falls after the last check (new registrations AND
     material updates both move this date).
  2. openFDA transparency/crl -- new Complete Response Letters naming a
     watched company (the highest-value negative regulatory event).

STORAGE: a single JSON document in the existing S3 data-lake bucket
(watchlist/watchlist.json), digests under watchlist/digests/. Deliberately
NOT a database: the Phase-3 reliability work introduces Postgres for
LangGraph checkpointing and this moves there with it -- standing up RDS
for a JSON document tonight would be infrastructure ahead of need. The
concurrency story at this scale (single-user watchlist edits) is
last-writer-wins, which S3 PUT gives for free.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import boto3
import requests

S3_BUCKET = os.getenv("S3_BUCKET", "medical-rag-raw-data-lake-jn-9043")
WATCHLIST_KEY = "watchlist/watchlist.json"
LATEST_DIGEST_KEY = "watchlist/latest_digest.json"

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"
CRL_API = "https://api.fda.gov/transparency/crl.json"

VALID_TYPES = ("drug", "company", "nct", "topic")


def _s3():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def _load(key: str, default):
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except _s3().exceptions.NoSuchKey:
        return default
    except Exception:
        # Bucket unreachable (no creds in some local setups) surfaces to the
        # API layer as an explicit error rather than a silent empty list.
        raise


def _save(key: str, doc) -> None:
    _s3().put_object(Bucket=S3_BUCKET, Key=key,
                     Body=json.dumps(doc, indent=2).encode(),
                     ContentType="application/json")


# --- CRUD --------------------------------------------------------------------
def list_entries() -> dict:
    return _load(WATCHLIST_KEY, {"entries": [], "last_checked": None})


def add_entry(entry_type: str, value: str) -> dict:
    if entry_type not in VALID_TYPES:
        raise ValueError(f"type must be one of {VALID_TYPES}")
    doc = list_entries()
    value = value.strip()
    if any(e["type"] == entry_type and e["value"].lower() == value.lower()
           for e in doc["entries"]):
        return doc  # idempotent
    doc["entries"].append({
        "id": str(uuid.uuid4())[:8],
        "type": entry_type,
        "value": value,
        "added_at": datetime.now(timezone.utc).isoformat(),
    })
    _save(WATCHLIST_KEY, doc)
    return doc


def remove_entry(entry_id: str) -> dict:
    doc = list_entries()
    doc["entries"] = [e for e in doc["entries"] if e["id"] != entry_id]
    _save(WATCHLIST_KEY, doc)
    return doc


# --- change detection ----------------------------------------------------------
def _ctgov_changes(term: str, since_iso: str, is_nct: bool) -> list[dict]:
    params = {
        "format": "json",
        "pageSize": 20,
        "sort": "LastUpdatePostDate:desc",
        "fields": ("NCTId|BriefTitle|OverallStatus|Phase|LeadSponsorName|"
                   "LastUpdatePostDate"),
    }
    if is_nct:
        params["filter.ids"] = term
    else:
        params["query.term"] = term
    r = requests.get(CTGOV_API, params=params, timeout=30)
    r.raise_for_status()
    out = []
    since = since_iso[:10]
    for st in r.json().get("studies", []):
        proto = st.get("protocolSection", {})
        status = proto.get("statusModule", {})
        upd = (status.get("lastUpdatePostDateStruct") or {}).get("date") or ""
        if upd <= since:
            continue
        ident = proto.get("identificationModule", {})
        out.append({
            "nct_id": ident.get("nctId"),
            "title": (ident.get("briefTitle") or "")[:140],
            "status": status.get("overallStatus"),
            "last_update": upd,
            "url": f"https://clinicaltrials.gov/study/{ident.get('nctId')}",
        })
    return out


def _crl_changes(company: str, since_iso: str) -> list[dict]:
    r = requests.get(CRL_API, params={
        "search": f'company_name:"{company}"', "limit": 20}, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    out = []
    since = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    for letter in r.json().get("results", []):
        try:
            ld = datetime.strptime(letter.get("letter_date", ""), "%m/%d/%Y") \
                .replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ld <= since:
            continue
        out.append({
            "application_numbers": letter.get("application_number"),
            "company": letter.get("company_name"),
            "letter_date": letter.get("letter_date"),
            "letter_type": letter.get("letter_type"),
        })
    return out


def run_check() -> dict:
    """Diff every watched entity against both live sources; persist and
    return the digest. `since` = last check, or 7 days back on first run."""
    doc = list_entries()
    since = doc.get("last_checked") or (
        datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    items = []
    for e in doc["entries"]:
        finding: dict = {"entry": e, "trial_changes": [], "new_crls": [],
                         "errors": []}
        try:
            finding["trial_changes"] = _ctgov_changes(
                e["value"], since, is_nct=(e["type"] == "nct"))
        except Exception as exc:  # noqa: BLE001 -- one entity must not sink the digest
            finding["errors"].append(f"ctgov: {exc}")
        if e["type"] == "company":
            try:
                finding["new_crls"] = _crl_changes(e["value"], since)
            except Exception as exc:  # noqa: BLE001
                finding["errors"].append(f"crl: {exc}")
        if finding["trial_changes"] or finding["new_crls"] or finding["errors"]:
            items.append(finding)

    now = datetime.now(timezone.utc)
    digest = {
        "checked_at": now.isoformat(),
        "since": since,
        "entries_checked": len(doc["entries"]),
        "items": items,
    }
    _save(LATEST_DIGEST_KEY, digest)
    _save(f"watchlist/digests/{now.strftime('%Y-%m-%dT%H%M%SZ')}.json", digest)
    doc["last_checked"] = now.isoformat()
    _save(WATCHLIST_KEY, doc)
    return digest


def latest_digest() -> dict:
    return _load(LATEST_DIGEST_KEY, {"checked_at": None, "items": []})


if __name__ == "__main__":
    print(json.dumps(run_check(), indent=2)[:4000])
