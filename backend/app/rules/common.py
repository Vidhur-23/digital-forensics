"""Generic, document-agnostic deterministic checks (Phase 2).

Two families of checks that apply to any document type:

* **Required-field presence** — is each field the document type is expected to
  carry actually populated? The *set* of required fields is per-type (a passport
  requires an MRZ; other types may not), so it is looked up by document type.
* **Generic field-format** — non-empty values, plausible generic structure of
  the nationality code and document number.

Deliberately NOT here (belongs in ``rules/passport.py``): strict passport
document-number formats. This module keeps the *generic* structural floor only,
so document/template-specific formats can be layered on per type without a
single universal regex.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.api.schemas.document import ScreeningResponse
from app.rules.schemas import RuleFinding, RuleSeverity, RuleStatus

REQUIRED_CATEGORY = "required_field"
FORMAT_CATEGORY = "format"

# Required fields per document type. Extend this map as new types are supported;
# do not assume every type shares the passport's field set.
#   (visual/textual fields, mrz_required)
_REQUIRED_FIELDS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "passport": (
        ("name", "date_of_birth", "document_number", "nationality", "issue_date", "expiry_date"),
        True,  # a passport is expected to carry an MRZ
    ),
}

# Generic (non-strict) structural expectations.
_NATIONALITY_RE = re.compile(r"^[A-Z]{3}$")          # ICAO 3-letter code
_DOC_NO_GENERIC_RE = re.compile(r"^[A-Za-z0-9\- ]{4,20}$")  # loose, not passport-specific


def _value(result: ScreeningResponse, name: str) -> Optional[str]:
    fv = result.fields.get(name)
    if not fv:
        return None
    v = (fv.value or "").strip()
    return v or None


def required_fields_for(document_type: str) -> Tuple[Tuple[str, ...], bool]:
    return _REQUIRED_FIELDS.get(document_type, _REQUIRED_FIELDS["passport"])


def check_required_fields(result: ScreeningResponse) -> List[RuleFinding]:
    """One finding per required field: present (PASS) or missing (FAIL/WARNING)."""
    findings: List[RuleFinding] = []
    fields, mrz_required = required_fields_for(result.document_type)

    for name in fields:
        present = _value(result, name) is not None
        findings.append(
            RuleFinding.make(
                rule_id=f"REQUIRED_{name.upper()}",
                category=REQUIRED_CATEGORY,
                status=RuleStatus.PASS if present else RuleStatus.FAIL,
                severity=RuleSeverity.INFO if present else RuleSeverity.MEDIUM,
                field=name,
                message=(
                    f"Required field '{name}' is present."
                    if present
                    else f"Required field '{name}' is missing."
                ),
                evidence={} if present else {"value": None},
            )
        )

    if mrz_required:
        mrz_present = bool(result.mrz and result.mrz.detected)
        findings.append(
            RuleFinding.make(
                rule_id="REQUIRED_MRZ",
                category=REQUIRED_CATEGORY,
                status=RuleStatus.PASS if mrz_present else RuleStatus.WARNING,
                severity=RuleSeverity.INFO if mrz_present else RuleSeverity.MEDIUM,
                field="mrz",
                message=(
                    "MRZ is present." if mrz_present else "MRZ is missing for a document type that expects one."
                ),
            )
        )

    return findings


def check_field_formats(result: ScreeningResponse) -> List[RuleFinding]:
    """Generic (non-restrictive) structural validation of present fields.

    Only emits findings for fields that are present — absence is handled by
    :func:`check_required_fields`. Name is intentionally NOT format-restricted
    (names vary enormously); we only check it is non-empty.
    """
    findings: List[RuleFinding] = []

    # Non-empty checks for every present field (catches whitespace-only OCR).
    for name, fv in result.fields.items():
        raw = (fv.value or "").strip()
        if raw:
            continue
        findings.append(
            RuleFinding.make(
                rule_id=f"FORMAT_NONEMPTY_{name.upper()}",
                category=FORMAT_CATEGORY,
                status=RuleStatus.FAIL,
                severity=RuleSeverity.LOW,
                field=name,
                message=f"Field '{name}' is present but empty after trimming.",
                evidence={"value": fv.value},
            )
        )

    # Nationality: expect an ICAO 3-letter alpha code.
    nat = _value(result, "nationality")
    if nat is not None:
        ok = bool(_NATIONALITY_RE.match(nat.upper()))
        findings.append(
            RuleFinding.make(
                "FORMAT_NATIONALITY", FORMAT_CATEGORY,
                RuleStatus.PASS if ok else RuleStatus.WARNING,
                RuleSeverity.INFO if ok else RuleSeverity.LOW,
                field="nationality",
                message=(
                    "Nationality is a 3-letter country code."
                    if ok
                    else "Nationality is not a standard 3-letter country code."
                ),
                evidence={"value": nat},
            )
        )

    # Document number: generic structural floor only (strict passport format is
    # checked in rules/passport.py).
    doc_no = _value(result, "document_number")
    if doc_no is not None:
        ok = bool(_DOC_NO_GENERIC_RE.match(doc_no))
        findings.append(
            RuleFinding.make(
                "FORMAT_DOCUMENT_NUMBER_GENERIC", FORMAT_CATEGORY,
                RuleStatus.PASS if ok else RuleStatus.WARNING,
                RuleSeverity.INFO if ok else RuleSeverity.LOW,
                field="document_number",
                message=(
                    "Document number has a plausible generic structure."
                    if ok
                    else "Document number contains unexpected characters or length."
                ),
                evidence={"value": doc_no},
            )
        )

    return findings
