"""Cross-field consistency checks: visual fields vs parsed MRZ (Phase 2).

The visual (printed) fields and the MRZ are two independent encodings of the
same identity. When they disagree, that is a concrete, explainable signal worth
review — but still evidence, never a verdict.

Comparisons are done on *normalised* values so pure formatting differences never
raise a false mismatch:
* names -> case-folded, whitespace-collapsed token multisets (order-independent,
  MRZ ``<`` separators removed);
* dates -> real ``date`` objects (visual text vs raw MRZ ``YYMMDD``);
* document number / nationality -> case-folded, separator-stripped strings.

A field whose visual value was itself filled from the MRZ (``source == "mrz"``)
cannot be independently cross-checked, so it is reported ``NOT_APPLICABLE``.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from app.api.schemas.document import FieldValue, ScreeningResponse
from app.rules.dates import parse_date, parse_mrz_date
from app.rules.schemas import RuleFinding, RuleSeverity, RuleStatus

CATEGORY = "consistency"

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


def _norm_tokens(text: str) -> Counter:
    return Counter(t for t in re.split(r"\s+", text.upper().strip()) if t)


def _norm_id(text: str) -> str:
    return _NON_ALNUM_RE.sub("", text.upper())


def _visual(result: ScreeningResponse, name: str) -> Optional[FieldValue]:
    """Return the field only if it is an *independent* visual read."""
    fv = result.fields.get(name)
    if fv and fv.value and fv.value.strip() and fv.source == "visual":
        return fv
    return None


def _na(rule_id: str, field: str, reason: str) -> RuleFinding:
    return RuleFinding.make(
        rule_id, CATEGORY, RuleStatus.NOT_APPLICABLE, RuleSeverity.INFO,
        reason, field=field,
    )


def check_consistency(result: ScreeningResponse) -> List[RuleFinding]:
    findings: List[RuleFinding] = []

    if not result.mrz or not result.mrz.detected:
        findings.append(
            _na("CONSISTENCY_MRZ", "mrz", "No MRZ detected; visual/MRZ consistency not applicable.")
        )
        return findings

    m = result.mrz.fields

    findings.append(_check_name(result, m))
    findings.append(_check_dob(result, m))
    findings.append(_check_document_number(result, m))
    findings.append(_check_nationality(result, m))
    findings.append(_check_expiry(result, m))
    return findings


def _check_name(result: ScreeningResponse, m) -> RuleFinding:
    fv = _visual(result, "name")
    mrz_name = " ".join(p for p in (m.given_names, m.surname) if p).strip()
    if fv is None or not mrz_name:
        return _na("CONSISTENCY_NAME", "name", "No independent visual name to compare against MRZ.")
    v_tokens = _norm_tokens(fv.value)
    m_tokens = _norm_tokens(mrz_name)
    if v_tokens == m_tokens:
        return RuleFinding.make(
            "CONSISTENCY_NAME", CATEGORY, RuleStatus.PASS, RuleSeverity.INFO,
            "Visual name matches MRZ name (order/formatting ignored).",
            field="name",
            evidence={"visual_value": fv.value, "mrz_value": mrz_name},
        )
    # A partial visual read (every visual token also appears in the MRZ name,
    # e.g. surname-only) is not a value conflict — it is a subset, so PASS
    # rather than raise a false mismatch on incomplete OCR.
    if v_tokens and all(m_tokens[t] >= c for t, c in v_tokens.items()):
        return RuleFinding.make(
            "CONSISTENCY_NAME", CATEGORY, RuleStatus.PASS, RuleSeverity.INFO,
            "Visual name is a subset of the MRZ name (partial read, no conflict).",
            field="name",
            evidence={"visual_value": fv.value, "mrz_value": mrz_name},
        )
    return RuleFinding.make(
        "CONSISTENCY_NAME", CATEGORY, RuleStatus.FAIL, RuleSeverity.HIGH,
        "Visual name differs from MRZ name.",
        field="name",
        evidence={"visual_value": fv.value, "mrz_value": mrz_name},
    )


def _check_dob(result: ScreeningResponse, m) -> RuleFinding:
    fv = _visual(result, "date_of_birth")
    v_date = parse_date(fv.value) if fv else None
    m_date = parse_mrz_date(m.date_of_birth, is_expiry=False)
    if fv is None or m_date is None:
        return _na("CONSISTENCY_DOB", "date_of_birth", "No comparable visual/MRZ date of birth.")
    if v_date is None:
        return RuleFinding.make(
            "CONSISTENCY_DOB", CATEGORY, RuleStatus.FAIL, RuleSeverity.MEDIUM,
            "Visual date of birth could not be parsed for comparison.",
            field="date_of_birth",
            evidence={"visual_value": fv.value, "mrz_value": m_date.isoformat()},
        )
    if v_date == m_date:
        return RuleFinding.make(
            "CONSISTENCY_DOB", CATEGORY, RuleStatus.PASS, RuleSeverity.INFO,
            "Visual date of birth matches MRZ date of birth.",
            field="date_of_birth",
            evidence={"visual_value": v_date.isoformat(), "mrz_value": m_date.isoformat()},
        )
    return RuleFinding.make(
        "CONSISTENCY_DOB", CATEGORY, RuleStatus.FAIL, RuleSeverity.HIGH,
        "Visual date of birth differs from MRZ date of birth.",
        field="date_of_birth",
        evidence={"visual_value": v_date.isoformat(), "mrz_value": m_date.isoformat()},
    )


def _check_document_number(result: ScreeningResponse, m) -> RuleFinding:
    fv = _visual(result, "document_number")
    if fv is None or not m.document_number:
        return _na("CONSISTENCY_DOCUMENT_NUMBER", "document_number",
                   "No independent visual document number to compare against MRZ.")
    v = _norm_id(fv.value)
    mm = _norm_id(m.document_number)
    if v == mm:
        return RuleFinding.make(
            "CONSISTENCY_DOCUMENT_NUMBER", CATEGORY, RuleStatus.PASS, RuleSeverity.INFO,
            "Visual document number matches MRZ document number.",
            field="document_number",
            evidence={"visual_value": fv.value, "mrz_value": m.document_number},
        )
    # One being a prefix of the other is a common OCR truncation, not a true
    # mismatch — flag for review rather than asserting a discrepancy.
    if v and mm and (v.startswith(mm) or mm.startswith(v)):
        return RuleFinding.make(
            "CONSISTENCY_DOCUMENT_NUMBER", CATEGORY, RuleStatus.WARNING, RuleSeverity.LOW,
            "Visual and MRZ document numbers differ only by truncation (possible OCR read error).",
            field="document_number",
            evidence={"visual_value": fv.value, "mrz_value": m.document_number},
        )
    return RuleFinding.make(
        "CONSISTENCY_DOCUMENT_NUMBER", CATEGORY, RuleStatus.FAIL, RuleSeverity.HIGH,
        "Visual document number differs from MRZ document number.",
        field="document_number",
        evidence={"visual_value": fv.value, "mrz_value": m.document_number},
    )


def _check_nationality(result: ScreeningResponse, m) -> RuleFinding:
    fv = _visual(result, "nationality")
    if fv is None or not m.nationality:
        return _na("CONSISTENCY_NATIONALITY", "nationality",
                   "No independent visual nationality to compare against MRZ.")
    if _norm_id(fv.value) == _norm_id(m.nationality):
        return RuleFinding.make(
            "CONSISTENCY_NATIONALITY", CATEGORY, RuleStatus.PASS, RuleSeverity.INFO,
            "Visual nationality matches MRZ nationality.",
            field="nationality",
            evidence={"visual_value": fv.value, "mrz_value": m.nationality},
        )
    return RuleFinding.make(
        "CONSISTENCY_NATIONALITY", CATEGORY, RuleStatus.FAIL, RuleSeverity.MEDIUM,
        "Visual nationality differs from MRZ nationality.",
        field="nationality",
        evidence={"visual_value": fv.value, "mrz_value": m.nationality},
    )


def _check_expiry(result: ScreeningResponse, m) -> RuleFinding:
    fv = _visual(result, "expiry_date")
    v_date = parse_date(fv.value) if fv else None
    m_date = parse_mrz_date(m.expiry_date, is_expiry=True)
    if fv is None or m_date is None:
        return _na("CONSISTENCY_EXPIRY", "expiry_date", "No comparable visual/MRZ expiry date.")
    if v_date is None:
        return RuleFinding.make(
            "CONSISTENCY_EXPIRY", CATEGORY, RuleStatus.FAIL, RuleSeverity.MEDIUM,
            "Visual expiry date could not be parsed for comparison.",
            field="expiry_date",
            evidence={"visual_value": fv.value, "mrz_value": m_date.isoformat()},
        )
    if v_date == m_date:
        return RuleFinding.make(
            "CONSISTENCY_EXPIRY", CATEGORY, RuleStatus.PASS, RuleSeverity.INFO,
            "Visual expiry date matches MRZ expiry date.",
            field="expiry_date",
            evidence={"visual_value": v_date.isoformat(), "mrz_value": m_date.isoformat()},
        )
    return RuleFinding.make(
        "CONSISTENCY_EXPIRY", CATEGORY, RuleStatus.FAIL, RuleSeverity.HIGH,
        "Visual expiry date differs from MRZ expiry date.",
        field="expiry_date",
        evidence={"visual_value": v_date.isoformat(), "mrz_value": m_date.isoformat()},
    )
