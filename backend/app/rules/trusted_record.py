"""Mock trusted-record comparison (Phase 2, PROTOTYPE ONLY).

This compares the extracted document fields against a small, hard-coded set of
**synthetic** records. It is a stand-in to demonstrate the "does this document
match a record of truth?" check.

DO NOT connect this to any government database or external confidential system.
The data below is invented for the prototype/demo only.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.api.schemas.document import ScreeningResponse
from app.rules.dates import parse_date, parse_mrz_date
from app.rules.schemas import RuleFinding, RuleSeverity, RuleStatus

CATEGORY = "trusted_record"

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


# --- SYNTHETIC prototype data (NOT real identities) ------------------------
# Keyed by normalised document number. ``date_of_birth`` is ISO format.
_MOCK_RECORDS: Dict[str, dict] = {
    "L898902C3": {
        "name": "ANNA MARIA ERIKSSON",
        "date_of_birth": "1974-08-12",
        "status": "VALID",
    },
    "X1234567": {
        "name": "JOHN DOE",
        "date_of_birth": "1990-01-15",
        "status": "REVOKED",
    },
}


def _norm_id(text: str) -> str:
    return _NON_ALNUM_RE.sub("", (text or "").upper())


def _norm_name(text: str) -> str:
    return " ".join(sorted(t for t in re.split(r"\s+", (text or "").upper()) if t))


def _visual_or_any(result: ScreeningResponse, name: str) -> Optional[str]:
    fv = result.fields.get(name)
    return fv.value if fv and fv.value else None


def _lookup(result: ScreeningResponse) -> tuple[Optional[str], Optional[dict]]:
    """Find a mock record by visual or MRZ document number."""
    candidates = []
    doc = _visual_or_any(result, "document_number")
    if doc:
        candidates.append(doc)
    if result.mrz and result.mrz.detected and result.mrz.fields.document_number:
        candidates.append(result.mrz.fields.document_number)
    for c in candidates:
        rec = _MOCK_RECORDS.get(_norm_id(c))
        if rec:
            return c, rec
    return (candidates[0] if candidates else None), None


def check_trusted_record(result: ScreeningResponse) -> List[RuleFinding]:
    findings: List[RuleFinding] = []
    doc_no, record = _lookup(result)

    if doc_no is None:
        findings.append(
            RuleFinding.make(
                "TRUSTED_RECORD_LOOKUP", CATEGORY, RuleStatus.NOT_APPLICABLE,
                RuleSeverity.INFO, "No document number available for trusted-record lookup.",
                field="document_number",
            )
        )
        return findings

    if record is None:
        findings.append(
            RuleFinding.make(
                "TRUSTED_RECORD_EXISTS", CATEGORY, RuleStatus.WARNING, RuleSeverity.LOW,
                "Document number not found in the (synthetic prototype) trusted-record set.",
                field="document_number",
                evidence={"document_number": doc_no, "source": "synthetic-prototype"},
            )
        )
        return findings

    findings.append(
        RuleFinding.make(
            "TRUSTED_RECORD_EXISTS", CATEGORY, RuleStatus.PASS, RuleSeverity.INFO,
            "Document number found in the (synthetic prototype) trusted-record set.",
            field="document_number",
            evidence={"document_number": doc_no, "source": "synthetic-prototype"},
        )
    )

    # Name comparison (order-independent). Prefer the fuller MRZ name when
    # present, since the visual read may be a partial (surname-only) capture.
    doc_name = None
    if result.mrz and result.mrz.detected:
        mf = result.mrz.fields
        doc_name = " ".join(p for p in (mf.given_names, mf.surname) if p).strip() or None
    doc_name = doc_name or _visual_or_any(result, "name")
    if doc_name:
        match = _norm_name(doc_name) == _norm_name(record["name"])
        findings.append(
            RuleFinding.make(
                "TRUSTED_RECORD_NAME", CATEGORY,
                RuleStatus.PASS if match else RuleStatus.FAIL,
                RuleSeverity.INFO if match else RuleSeverity.HIGH,
                "Name matches the trusted record." if match else "Name does not match the trusted record.",
                field="name",
                evidence={"document_value": doc_name, "record_value": record["name"],
                          "source": "synthetic-prototype"},
            )
        )

    # DOB comparison (parse both to date objects; visual text or MRZ YYMMDD).
    doc_dob = None
    dob_field = result.fields.get("date_of_birth")
    if dob_field and dob_field.value:
        doc_dob = (
            parse_date(dob_field.value)
            if dob_field.source == "visual"
            else parse_mrz_date(dob_field.value, is_expiry=False)
        )
    record_dob = parse_date(record["date_of_birth"])
    if doc_dob and record_dob:
        match = doc_dob == record_dob
        findings.append(
            RuleFinding.make(
                "TRUSTED_RECORD_DOB", CATEGORY,
                RuleStatus.PASS if match else RuleStatus.FAIL,
                RuleSeverity.INFO if match else RuleSeverity.HIGH,
                "Date of birth matches the trusted record."
                if match else "Date of birth does not match the trusted record.",
                field="date_of_birth",
                evidence={"document_value": doc_dob.isoformat(),
                          "record_value": record_dob.isoformat(),
                          "source": "synthetic-prototype"},
            )
        )

    # Record status (VALID / REVOKED / ...). Non-VALID is a WARNING, not a verdict.
    status_valid = record["status"] == "VALID"
    findings.append(
        RuleFinding.make(
            "TRUSTED_RECORD_STATUS", CATEGORY,
            RuleStatus.PASS if status_valid else RuleStatus.WARNING,
            RuleSeverity.INFO if status_valid else RuleSeverity.MEDIUM,
            f"Trusted-record status is {record['status']}.",
            field="document_number",
            evidence={"status": record["status"], "source": "synthetic-prototype"},
        )
    )

    return findings
