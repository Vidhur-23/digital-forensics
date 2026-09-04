"""Passport/TD3-template-specific deterministic rules (Phase 2).

Rules here are tied to the passport template specifically and must NOT be
promoted into ``rules/common.py`` (which stays document-agnostic). Keeping them
separate is what lets the engine select a per-type rule set:

    document type -> applicable rule set -> rules

Checks:
* MRZ layout must be TD3 for a passport;
* MRZ document-type indicator should begin with ``P``;
* sex indicator must be one of ``M`` / ``F`` / ``X`` / ``<`` (unspecified);
* passport document number must match the TD3 template (up to 9 alphanumerics).
"""
from __future__ import annotations

import re
from typing import List, Optional

from app.api.schemas.document import ScreeningResponse
from app.rules.schemas import RuleFinding, RuleSeverity, RuleStatus

CATEGORY = "passport"

# TD3 passport document number: alphanumeric, up to 9 characters.
_PASSPORT_DOC_NO_RE = re.compile(r"^[A-Z0-9]{1,9}$")
_VALID_SEX = {"M", "F", "X", "<", ""}


def _value(result: ScreeningResponse, name: str) -> Optional[str]:
    fv = result.fields.get(name)
    if not fv:
        return None
    v = (fv.value or "").strip()
    return v or None


def check_passport(result: ScreeningResponse) -> List[RuleFinding]:
    findings: List[RuleFinding] = []

    # Passport document-number template (strict, unlike the generic check).
    doc_no = _value(result, "document_number")
    if doc_no is not None:
        ok = bool(_PASSPORT_DOC_NO_RE.match(_norm(doc_no)))
        findings.append(
            RuleFinding.make(
                "PASSPORT_DOC_NUMBER_FORMAT", CATEGORY,
                RuleStatus.PASS if ok else RuleStatus.WARNING,
                RuleSeverity.INFO if ok else RuleSeverity.MEDIUM,
                field="document_number",
                message=(
                    "Document number matches the passport (TD3) template."
                    if ok
                    else "Document number does not match the passport (TD3) template (<=9 alphanumerics)."
                ),
                evidence={"value": doc_no},
            )
        )

    if result.mrz and result.mrz.detected:
        # TD3 layout expected for passports.
        fmt = result.mrz.format
        fmt_ok = fmt == "TD3"
        findings.append(
            RuleFinding.make(
                "PASSPORT_MRZ_FORMAT", CATEGORY,
                RuleStatus.PASS if fmt_ok else RuleStatus.WARNING,
                RuleSeverity.INFO if fmt_ok else RuleSeverity.MEDIUM,
                field="mrz",
                message=(
                    "MRZ layout is TD3 as expected for a passport."
                    if fmt_ok
                    else f"MRZ layout '{fmt}' is not the expected TD3 passport layout."
                ),
                evidence={"format": fmt},
            )
        )

        # Document-type indicator should be a 'P' family code.
        dtype = (result.mrz.fields.document_type or "").upper()
        dtype_ok = dtype.startswith("P")
        findings.append(
            RuleFinding.make(
                "PASSPORT_MRZ_TYPE", CATEGORY,
                RuleStatus.PASS if dtype_ok else RuleStatus.WARNING,
                RuleSeverity.INFO if dtype_ok else RuleSeverity.MEDIUM,
                field="mrz",
                message=(
                    "MRZ document-type indicator is a passport ('P') code."
                    if dtype_ok
                    else "MRZ document-type indicator is not a passport ('P') code."
                ),
                evidence={"document_type": dtype},
            )
        )

        # Sex indicator must be a valid code.
        sex = (result.mrz.fields.sex or "").upper()
        sex_ok = sex in _VALID_SEX
        findings.append(
            RuleFinding.make(
                "PASSPORT_MRZ_SEX", CATEGORY,
                RuleStatus.PASS if sex_ok else RuleStatus.WARNING,
                RuleSeverity.INFO if sex_ok else RuleSeverity.LOW,
                field="sex",
                message=(
                    "MRZ sex indicator is a valid code."
                    if sex_ok
                    else "MRZ sex indicator is not one of M/F/X/<."
                ),
                evidence={"value": sex},
            )
        )

    return findings


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())
