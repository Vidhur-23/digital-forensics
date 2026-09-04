"""MRZ parsing and deterministic check-digit validation (Phase 2).

Phase 1 (``app.ocr.mrz``) *detects* the MRZ and best-effort reads the visible
values. Phase 2 re-parses the raw MRZ lines into a structured, positional
representation and validates the ICAO 9303 check digits. This is fully
deterministic — no ML, no heuristics beyond the published standard.

Supported layout: TD3 (passport) — two lines of 44 characters. The parser is
positional and extensible: a new layout (TD1/TD2/visa) can be added by writing
another ``parse_*`` + field-position table without touching the check-digit code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.api.schemas.document import ScreeningResponse
from app.rules.schemas import RuleFinding, RuleSeverity, RuleStatus

CATEGORY = "mrz"

TD3_LINE_LENGTH = 44

# Character values for the ICAO 9303 check-digit algorithm.
# Digits map to their value, A-Z map to 10..35, filler '<' maps to 0.
_WEIGHTS = (7, 3, 1)


def char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 10
    if c == "<":
        return 0
    # Any unexpected character (OCR noise) contributes 0; the mismatch surfaces
    # via the check-digit comparison rather than an exception.
    return 0


def compute_check_digit(data: str) -> str:
    """ICAO 9303 check digit: weighted sum of char values mod 10."""
    total = 0
    for i, ch in enumerate(data):
        total += char_value(ch) * _WEIGHTS[i % 3]
    return str(total % 10)


@dataclass
class MRZCheckDigit:
    name: str
    data: str          # the field substring the digit protects
    expected: str      # digit encoded in the MRZ
    computed: str      # digit we calculated

    @property
    def ok(self) -> bool:
        return self.expected == self.computed


@dataclass
class ParsedMRZ:
    format: str
    document_type: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    document_number: str = ""
    nationality: str = ""
    date_of_birth: str = ""       # raw YYMMDD
    sex: str = ""
    expiry_date: str = ""         # raw YYMMDD
    optional_data: str = ""       # personal number / optional field
    check_digits: List[MRZCheckDigit] = field(default_factory=list)


def _clean(token: str) -> str:
    return token.replace("<", " ").strip()


def parse_td3(line1: str, line2: str) -> ParsedMRZ:
    """Parse the two TD3 lines positionally and build the check-digit table."""
    p = ParsedMRZ(format="TD3")

    # Line 1: type(1) filler(1) issuing(3) name(39)
    p.document_type = line1[0:1].replace("<", "")
    p.issuing_country = _clean(line1[2:5])
    names = line1[5:]
    if "<<" in names:
        surname, _, given = names.partition("<<")
        p.surname = _clean(surname)
        p.given_names = _clean(given)
    else:
        p.surname = _clean(names)

    # Line 2 positional layout:
    #  0-8  document number      9    doc-number check digit
    # 10-12 nationality         13-18 date of birth       19 dob check digit
    #    20 sex                 21-26 expiry date          27 expiry check digit
    # 28-41 optional data        42   optional check digit 43 composite check
    doc_no = line2[0:9]
    doc_no_chk = line2[9:10]
    p.document_number = _clean(doc_no)
    p.nationality = _clean(line2[10:13])
    dob = line2[13:19]
    dob_chk = line2[19:20]
    p.date_of_birth = dob
    p.sex = _clean(line2[20:21])
    exp = line2[21:27]
    exp_chk = line2[27:28]
    p.expiry_date = exp
    optional = line2[28:42]
    optional_chk = line2[42:43]
    p.optional_data = _clean(optional)
    composite_chk = line2[43:44]

    # Composite spans doc-no + its check, dob + its check, expiry + its check,
    # and the optional field + its check digit (ICAO 9303 upper+lower composite).
    composite_data = line2[0:10] + line2[13:20] + line2[21:43]

    checks: List[MRZCheckDigit] = []
    if doc_no_chk:
        checks.append(MRZCheckDigit("document_number", doc_no, doc_no_chk, compute_check_digit(doc_no)))
    if dob_chk:
        checks.append(MRZCheckDigit("date_of_birth", dob, dob_chk, compute_check_digit(dob)))
    if exp_chk:
        checks.append(MRZCheckDigit("expiry_date", exp, exp_chk, compute_check_digit(exp)))
    # Optional-data check digit is only meaningful when the field is used.
    if optional_chk and optional_chk != "<":
        checks.append(
            MRZCheckDigit("optional_data", optional, optional_chk, compute_check_digit(optional))
        )
    if composite_chk:
        checks.append(
            MRZCheckDigit("composite", composite_data, composite_chk, compute_check_digit(composite_data))
        )
    p.check_digits = checks
    return p


def parse_mrz_lines(lines: List[str]) -> Optional[ParsedMRZ]:
    """Parse raw MRZ lines. Currently supports TD3 (2 lines)."""
    lines = [ln.strip().upper() for ln in lines if ln.strip()]
    if len(lines) == 2 and all(len(ln) == TD3_LINE_LENGTH for ln in lines):
        return parse_td3(lines[0], lines[1])
    # Length-tolerant fallback: pad/truncate to 44 so OCR that dropped/added a
    # filler still yields a best-effort structured parse (check digits may fail,
    # which is itself the useful signal).
    if len(lines) == 2:
        norm = [(ln + "<" * TD3_LINE_LENGTH)[:TD3_LINE_LENGTH] for ln in lines]
        return parse_td3(norm[0], norm[1])
    return None


def _mrz_lines_from_result(result: ScreeningResponse) -> List[str]:
    if not result.mrz or not result.mrz.detected:
        return []
    if result.mrz.text:
        return [ln for ln in result.mrz.text.split("\n") if ln.strip()]
    return []


def check_mrz(result: ScreeningResponse) -> List[RuleFinding]:
    """Deterministic MRZ structure + check-digit rules over a Phase 1 result."""
    findings: List[RuleFinding] = []

    if not result.mrz or not result.mrz.detected:
        findings.append(
            RuleFinding.make(
                "MRZ_PRESENT", CATEGORY, RuleStatus.WARNING, RuleSeverity.MEDIUM,
                "No MRZ was detected; MRZ-based checks were skipped.",
                field="mrz",
            )
        )
        return findings

    lines = _mrz_lines_from_result(result)
    parsed = parse_mrz_lines(lines)

    if parsed is None:
        findings.append(
            RuleFinding.make(
                "MRZ_STRUCTURE", CATEGORY, RuleStatus.FAIL, RuleSeverity.MEDIUM,
                "MRZ was detected but could not be parsed into a known layout.",
                field="mrz",
                evidence={"lines": lines},
            )
        )
        return findings

    findings.append(
        RuleFinding.make(
            "MRZ_STRUCTURE", CATEGORY, RuleStatus.PASS, RuleSeverity.INFO,
            f"MRZ parsed as {parsed.format}.",
            field="mrz",
            evidence={"format": parsed.format},
        )
    )

    # One finding per check digit.
    for chk in parsed.check_digits:
        findings.append(
            RuleFinding.make(
                rule_id=f"MRZ_CHECK_{chk.name.upper()}",
                category=CATEGORY,
                status=RuleStatus.PASS if chk.ok else RuleStatus.FAIL,
                severity=RuleSeverity.INFO if chk.ok else RuleSeverity.HIGH,
                message=(
                    f"MRZ check digit for {chk.name} is valid."
                    if chk.ok
                    else f"MRZ check digit for {chk.name} does not match the encoded value."
                ),
                field=chk.name,
                evidence={
                    "data": chk.data,
                    "expected_check_digit": chk.expected,
                    "computed_check_digit": chk.computed,
                },
            )
        )

    return findings
