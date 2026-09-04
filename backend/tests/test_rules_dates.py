"""Phase 2: date parsing, normalisation and relationship rules."""
from __future__ import annotations

from datetime import date

from app.rules.dates import check_dates, is_valid_date, parse_date, parse_mrz_date
from app.rules.schemas import RuleStatus
from tests.rules_helpers import field, genuine_response, make_response


def _status(findings, rule_id):
    return next(f.status for f in findings if f.rule_id == rule_id)


# --- parsing ---------------------------------------------------------------


def test_parse_supported_formats():
    assert parse_date("12 APR 1998") == date(1998, 4, 12)
    assert parse_date("1998-04-12") == date(1998, 4, 12)
    assert parse_date("12/04/1998") == date(1998, 4, 12)
    assert parse_date("12-04-1998") == date(1998, 4, 12)


def test_parse_rejects_impossible_and_malformed():
    assert parse_date("32 APR 1998") is None
    assert parse_date("1998-13-01") is None
    assert parse_date("not a date") is None
    assert parse_date("") is None
    assert is_valid_date("12 APR 1998") is True
    assert is_valid_date("32 APR 1998") is False


def test_parse_mrz_date_century_windowing():
    # DOB in the future is pushed back a century.
    assert parse_mrz_date("740812", is_expiry=False) == date(1974, 8, 12)
    # Expiry taken at face value in the current century (not pushed forward),
    # so an expired document keeps its real past date.
    assert parse_mrz_date("120415", is_expiry=True) == date(2012, 4, 15)
    assert parse_mrz_date("300101", is_expiry=True) == date(2030, 1, 1)
    assert parse_mrz_date("99AB99") is None


# --- relationship rules ----------------------------------------------------


def test_invalid_date_flagged():
    r = make_response({"date_of_birth": field("32 APR 1998")})
    findings = check_dates(r)
    assert _status(findings, "DATE_VALID_DATE_OF_BIRTH") == RuleStatus.FAIL


def test_issue_after_expiry_fails():
    r = make_response(
        {"issue_date": field("01 JAN 2030"), "expiry_date": field("01 JAN 2025")}
    )
    findings = check_dates(r)
    assert _status(findings, "DATE_ISSUE_BEFORE_EXPIRY") == RuleStatus.FAIL


def test_issue_before_expiry_passes():
    r = make_response(
        {"issue_date": field("01 JAN 2020"), "expiry_date": field("01 JAN 2030")}
    )
    findings = check_dates(r)
    assert _status(findings, "DATE_ISSUE_BEFORE_EXPIRY") == RuleStatus.PASS


def test_expired_is_warning_not_fail():
    r = make_response({"expiry_date": field("01 JAN 2010")})
    findings = check_dates(r, today=date(2026, 9, 5))
    f = next(f for f in findings if f.rule_id == "DATE_EXPIRY_STATUS")
    assert f.status == RuleStatus.WARNING  # expired != forged
    assert f.evidence["status"] == "EXPIRED"


def test_valid_expiry_status():
    r = make_response({"expiry_date": field("01 JAN 2030")})
    findings = check_dates(r, today=date(2026, 9, 5))
    f = next(f for f in findings if f.rule_id == "DATE_EXPIRY_STATUS")
    assert f.status == RuleStatus.PASS
    assert f.evidence["status"] == "VALID"


def test_dob_after_issue_fails():
    # DOB 2020, issue 2010 -> impossible.
    r = make_response(
        {"date_of_birth": field("01 JAN 2020"), "issue_date": field("01 JAN 2010")}
    )
    findings = check_dates(r)
    assert _status(findings, "DATE_DOB_BEFORE_ISSUE") == RuleStatus.FAIL


def test_genuine_dates_all_pass():
    findings = check_dates(genuine_response(), today=date(2026, 9, 5))
    assert all(f.status == RuleStatus.PASS for f in findings)
