"""Date parsing, normalisation and date-relationship rules (Phase 2).

Two layers live here:

1. **Reusable date utilities** — parse the human-readable date formats the
   Phase 1 extractor emits (``12 APR 1998``, ``1998-04-12`` ...) and the raw
   ``YYMMDD`` values the MRZ carries, into real :class:`datetime.date` objects.
   Everything downstream compares ``date`` objects, never raw strings.

2. **Date rule checks** — deterministic relationship rules over those dates:
   issue < expiry, expiry status (VALID/EXPIRED), and DOB < issue.

Nothing here decides authenticity. An expired document is reported as EXPIRED,
never as forged.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import List, Optional

from app.api.schemas.document import ScreeningResponse
from app.rules.schemas import RuleFinding, RuleSeverity, RuleStatus

CATEGORY = "date"

# Human-readable formats the Phase 1 visual extractor can produce.
#
# The extractor (``app.ocr.extractor``) accepts any of space, ``/``, ``.`` or
# ``-`` as a separator and both 2- and 4-digit years, so a genuine passport can
# yield ``12-APR-1998``, ``12/04/98`` or ``12 APR 98`` just as legitimately as
# ``12 APR 1998``. We therefore collapse every separator run to a single space
# in :func:`parse_date` before matching, and the format list below is written
# against that normalised form — one space between each component. Keeping the
# format separator identical to the input separator (the old approach) rejected
# valid dates purely on punctuation, which is the bug this list fixes.
_TEXT_FORMATS = (
    "%d %b %Y",   # 12 APR 1998   (also 12-APR-1998, 12/APR/1998, ...)
    "%d %B %Y",   # 12 APRIL 1998
    "%d %b %y",   # 12 APR 98
    "%d %B %y",   # 12 APRIL 98
    "%Y %m %d",   # 1998-04-12    (year-first is always 4-digit)
    "%d %m %Y",   # 12/04/1998
    "%d %m %y",   # 12/04/98
)

# One or more separator characters (space, slash, dot, hyphen) between the
# day/month/year components. ``%Y`` will not match a 2-digit run, so 4- and
# 2-digit years stay unambiguous and never collide across the formats above.
_SEP_RE = re.compile(r"[ /.\-]+")


def parse_date(raw: Optional[str]) -> Optional[date]:
    """Parse a human-readable date string into a ``date``.

    Accepts the day/month/year separators (space, ``/``, ``.``, ``-``) and the
    2- or 4-digit years the Phase 1 extractor can emit, so a valid date is not
    rejected merely for its punctuation or year width.

    Returns ``None`` for empty, malformed or impossible dates (e.g.
    ``32 APR 1998``). ``strptime`` rejects impossible day/month values, so we get
    calendar validation for free.
    """
    if not raw:
        return None
    text = _SEP_RE.sub(" ", raw.strip().upper()).strip()
    for fmt in _TEXT_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_mrz_date(yymmdd: Optional[str], *, is_expiry: bool = False) -> Optional[date]:
    """Parse a raw MRZ ``YYMMDD`` value into a ``date``.

    The MRZ has no century. We resolve the two-digit year conservatively:
    * dates of birth cannot be in the future, so a year that would land ahead of
      today is pushed back a century (``74`` -> 1974, not 2074);
    * expiry dates are taken at face value in the current century (``12`` ->
      2012). We do NOT push a past expiry forward, because an expired document is
      legitimate (it is simply EXPIRED, reported elsewhere) and must keep its
      real past date rather than being rewritten into the future.
    """
    if not yymmdd:
        return None
    s = yymmdd.strip()
    if len(s) != 6 or not s.isdigit():
        return None
    yy, mm, dd = int(s[0:2]), int(s[2:4]), int(s[4:6])
    today = date.today()
    century = today.year - (today.year % 100)  # e.g. 2000
    year = century + yy
    if not is_expiry:
        # DOB in the future belongs to the previous century.
        if year > today.year:
            year -= 100
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def is_valid_date(raw: Optional[str]) -> bool:
    """True if ``raw`` parses to a real calendar date."""
    return parse_date(raw) is not None


# --- rule checks -----------------------------------------------------------


def _field(result: ScreeningResponse, name: str) -> Optional[str]:
    fv = result.fields.get(name)
    return fv.value if fv else None


def _field_source(result: ScreeningResponse, name: str) -> str:
    fv = result.fields.get(name)
    return getattr(fv, "source", "visual") if fv else "visual"


def _parse_field(result: ScreeningResponse, name: str, *, is_expiry: bool = False) -> Optional[date]:
    """Parse a date field with the parser its ``source`` requires.

    MRZ-sourced fields carry the raw ``YYMMDD`` the machine-readable zone prints
    (e.g. ``740812``), which :func:`parse_date` cannot read — they need
    :func:`parse_mrz_date` and its century windowing. Visual fields carry
    human-readable text. Picking the parser by ``source`` stops a perfectly
    valid MRZ date being reported as an invalid calendar date.
    """
    fv = result.fields.get(name)
    if fv is None or not fv.value:
        return None
    if getattr(fv, "source", "visual") == "mrz":
        return parse_mrz_date(fv.value, is_expiry=is_expiry)
    return parse_date(fv.value)


def check_dates(result: ScreeningResponse, *, today: Optional[date] = None) -> List[RuleFinding]:
    """Run all date-related deterministic checks over a Phase 1 result."""
    today = today or date.today()
    findings: List[RuleFinding] = []

    # A date field filled from the MRZ carries a raw YYMMDD read out of the
    # machine-readable zone. If the MRZ was misread (its check digits fail
    # wholesale) that value is garbage — an unreadable MRZ, not a genuinely
    # invalid printed date. Reporting it as an impossible calendar date, or
    # feeding it into the issue/expiry ordering checks, would raise concerns off
    # data we know we couldn't read. So MRZ-sourced date fields are treated as
    # absent while the MRZ is unreliable; the unreadable MRZ is reported once by
    # MRZ_UNRELIABLE instead. Import locally to avoid a rules import cycle.
    from app.rules.mrz import mrz_is_reliable

    # Only suppress when an MRZ is actually present and reads unreliably; a field
    # merely tagged source="mrz" with no detected MRZ is still validated as-is.
    mrz_detected = bool(result.mrz and result.mrz.detected)
    mrz_unreadable = mrz_detected and not mrz_is_reliable(result)

    def _usable(name: str) -> bool:
        """False for an MRZ-sourced field whose detected MRZ is unreadable."""
        return not (mrz_unreadable and _field_source(result, name) == "mrz")

    dob_raw = _field(result, "date_of_birth") if _usable("date_of_birth") else None
    issue_raw = _field(result, "issue_date") if _usable("issue_date") else None
    expiry_raw = _field(result, "expiry_date") if _usable("expiry_date") else None

    dob = _parse_field(result, "date_of_birth") if _usable("date_of_birth") else None
    issue = _parse_field(result, "issue_date") if _usable("issue_date") else None
    expiry = _parse_field(result, "expiry_date", is_expiry=True) if _usable("expiry_date") else None

    # 1. Malformed / impossible date detection (per present field).
    for name, raw, parsed in (
        ("date_of_birth", dob_raw, dob),
        ("issue_date", issue_raw, issue),
        ("expiry_date", expiry_raw, expiry),
    ):
        if raw is None:
            continue  # absence is a required-field concern, not a date one
        if parsed is None:
            # A FAIL here is "present but unparseable", NOT "missing" — the field
            # was read (the value is quoted below) but does not resolve to a real
            # calendar date. Name the value and its source (MRZ vs printed text)
            # so the finding can't be mistaken for an absent field: an MRZ-sourced
            # failure is usually a garbled machine-read, not a bad printed date.
            label = name.replace("_", " ")
            source = _field_source(result, name)
            source_label = "the MRZ" if source == "mrz" else "the printed text"
            findings.append(
                RuleFinding.make(
                    rule_id=f"DATE_VALID_{name.upper()}",
                    category=CATEGORY,
                    status=RuleStatus.FAIL,
                    severity=RuleSeverity.HIGH,
                    field=name,
                    message=(
                        f"The {label} value '{raw}' (read from {source_label}) is "
                        f"present but is not a recognisable calendar date."
                    ),
                    evidence={"value": raw, "source": source},
                )
            )
        else:
            findings.append(
                RuleFinding.make(
                    rule_id=f"DATE_VALID_{name.upper()}",
                    category=CATEGORY,
                    status=RuleStatus.PASS,
                    severity=RuleSeverity.INFO,
                    field=name,
                    message=f"{name} is a valid calendar date.",
                    evidence={"value": raw, "normalized": parsed.isoformat()},
                )
            )

    # 2. Issue date < expiry date.
    if issue and expiry:
        if issue < expiry:
            findings.append(
                RuleFinding.make(
                    "DATE_ISSUE_BEFORE_EXPIRY", CATEGORY, RuleStatus.PASS,
                    RuleSeverity.INFO, "Issue date precedes expiry date.",
                    field="issue_date",
                    evidence={"issue_date": issue.isoformat(), "expiry_date": expiry.isoformat()},
                )
            )
        else:
            findings.append(
                RuleFinding.make(
                    "DATE_ISSUE_BEFORE_EXPIRY", CATEGORY, RuleStatus.FAIL,
                    RuleSeverity.HIGH,
                    "Issue date is on or after the expiry date, which is impossible.",
                    field="issue_date",
                    evidence={"issue_date": issue.isoformat(), "expiry_date": expiry.isoformat()},
                )
            )

    # 3. Expiry status (VALID / EXPIRED) — expired is NOT fraud.
    if expiry:
        expired = expiry < today
        findings.append(
            RuleFinding.make(
                "DATE_EXPIRY_STATUS", CATEGORY,
                RuleStatus.WARNING if expired else RuleStatus.PASS,
                RuleSeverity.LOW if expired else RuleSeverity.INFO,
                (
                    "Document is EXPIRED (this alone is not evidence of fraud)."
                    if expired
                    else "Document is within its validity period."
                ),
                field="expiry_date",
                evidence={
                    "expiry_date": expiry.isoformat(),
                    "as_of": today.isoformat(),
                    "status": "EXPIRED" if expired else "VALID",
                },
            )
        )

    # 4. DOB must precede issue date (a document cannot be issued before birth).
    if dob and issue:
        if dob < issue:
            findings.append(
                RuleFinding.make(
                    "DATE_DOB_BEFORE_ISSUE", CATEGORY, RuleStatus.PASS,
                    RuleSeverity.INFO, "Date of birth precedes issue date.",
                    field="date_of_birth",
                    evidence={"date_of_birth": dob.isoformat(), "issue_date": issue.isoformat()},
                )
            )
        else:
            findings.append(
                RuleFinding.make(
                    "DATE_DOB_BEFORE_ISSUE", CATEGORY, RuleStatus.FAIL,
                    RuleSeverity.HIGH,
                    "Date of birth is on or after the issue date, which is impossible.",
                    field="date_of_birth",
                    evidence={"date_of_birth": dob.isoformat(), "issue_date": issue.isoformat()},
                )
            )

    return findings
