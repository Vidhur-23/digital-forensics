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
_TEXT_FORMATS = (
    "%d %b %Y",   # 12 APR 1998
    "%d %B %Y",   # 12 APRIL 1998
    "%Y-%m-%d",   # 1998-04-12
    "%d/%m/%Y",   # 12/04/1998
    "%d-%m-%Y",   # 12-04-1998
    "%d.%m.%Y",   # 12.04.1998
    "%Y/%m/%d",   # 1998/04/12
)

_MONTH_SEP_RE = re.compile(r"[ /.\-]")


def parse_date(raw: Optional[str]) -> Optional[date]:
    """Parse a human-readable date string into a ``date``.

    Returns ``None`` for empty, malformed or impossible dates (e.g.
    ``32 APR 1998``). ``strptime`` rejects impossible day/month values, so we get
    calendar validation for free.
    """
    if not raw:
        return None
    text = raw.strip().upper()
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


def check_dates(result: ScreeningResponse, *, today: Optional[date] = None) -> List[RuleFinding]:
    """Run all date-related deterministic checks over a Phase 1 result."""
    today = today or date.today()
    findings: List[RuleFinding] = []

    dob_raw = _field(result, "date_of_birth")
    issue_raw = _field(result, "issue_date")
    expiry_raw = _field(result, "expiry_date")

    dob = parse_date(dob_raw)
    issue = parse_date(issue_raw)
    expiry = parse_date(expiry_raw)

    # 1. Malformed / impossible date detection (per present field).
    for name, raw, parsed in (
        ("date_of_birth", dob_raw, dob),
        ("issue_date", issue_raw, issue),
        ("expiry_date", expiry_raw, expiry),
    ):
        if raw is None:
            continue  # absence is a required-field concern, not a date one
        if parsed is None:
            findings.append(
                RuleFinding.make(
                    rule_id=f"DATE_VALID_{name.upper()}",
                    category=CATEGORY,
                    status=RuleStatus.FAIL,
                    severity=RuleSeverity.HIGH,
                    field=name,
                    message=f"Value in {name} is not a valid calendar date.",
                    evidence={"value": raw},
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
