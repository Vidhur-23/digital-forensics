"""Phase 2: Rules Engine orchestration + common/passport/trusted-record rules."""
from __future__ import annotations

from app.rules.engine import RulesEngine
from app.rules.schemas import RuleStatus
from tests.rules_helpers import field, genuine_response, make_mrz, make_response


def _by_id(results, rule_id):
    return next(f for f in results.findings if f.rule_id == rule_id)


def _has(results, rule_id):
    return any(f.rule_id == rule_id for f in results.findings)


def test_genuine_passport_all_pass():
    results = RulesEngine().evaluate(genuine_response())
    assert results.findings
    assert results.summary.has_failures is False
    assert all(f.status == RuleStatus.PASS for f in results.findings)


def test_missing_required_field():
    r = genuine_response()
    del r.fields["date_of_birth"]
    results = RulesEngine().evaluate(r)
    assert _by_id(results, "REQUIRED_DATE_OF_BIRTH").status == RuleStatus.FAIL


def test_invalid_date_relationship():
    r = genuine_response()
    r.fields["issue_date"] = field("01 JAN 2030")
    r.fields["expiry_date"] = field("01 JAN 2025")
    results = RulesEngine().evaluate(r)
    assert _by_id(results, "DATE_ISSUE_BEFORE_EXPIRY").status == RuleStatus.FAIL


def test_expired_passport_not_forged():
    mrz = make_mrz("L898902C3", "UTO", "740812", "F", "100101", "ERIKSSON", "ANNA MARIA")
    r = make_response(
        {
            "name": field("ANNA MARIA ERIKSSON"),
            "date_of_birth": field("12 AUG 1974"),
            "document_number": field("L898902C3"),
            "nationality": field("UTO"),
            "issue_date": field("01 JAN 2005"),
            "expiry_date": field("01 JAN 2010"),
        },
        mrz,
    )
    results = RulesEngine().evaluate(r)
    exp = _by_id(results, "DATE_EXPIRY_STATUS")
    assert exp.status == RuleStatus.WARNING  # not FAIL / not "forged"
    assert exp.evidence["status"] == "EXPIRED"


def test_dob_mismatch_via_engine():
    r = genuine_response()
    r.fields["date_of_birth"] = field("12 AUG 1999")  # MRZ still 1974
    results = RulesEngine().evaluate(r)
    assert _by_id(results, "CONSISTENCY_DOB").status == RuleStatus.FAIL


def test_document_number_mismatch_via_engine():
    r = genuine_response()
    r.fields["document_number"] = field("Z0000000")  # MRZ still L898902C3
    results = RulesEngine().evaluate(r)
    assert _by_id(results, "CONSISTENCY_DOCUMENT_NUMBER").status == RuleStatus.FAIL


def test_invalid_mrz_check_digit_via_engine():
    r = genuine_response()
    line1, line2 = r.mrz.text.split("\n")
    line2 = line2[:9] + ("0" if line2[9] != "0" else "1") + line2[10:]  # break doc-no check
    r.mrz.text = f"{line1}\n{line2}"
    results = RulesEngine().evaluate(r)
    assert _by_id(results, "MRZ_CHECK_DOCUMENT_NUMBER").status == RuleStatus.FAIL


# --- passport-specific rules ----------------------------------------------


def test_passport_specific_rules_present_and_pass():
    results = RulesEngine().evaluate(genuine_response())
    assert _by_id(results, "PASSPORT_MRZ_FORMAT").status == RuleStatus.PASS
    assert _by_id(results, "PASSPORT_MRZ_TYPE").status == RuleStatus.PASS
    assert _by_id(results, "PASSPORT_DOC_NUMBER_FORMAT").status == RuleStatus.PASS


# --- mock trusted record ---------------------------------------------------


def test_trusted_record_match():
    results = RulesEngine().evaluate(genuine_response())
    assert _by_id(results, "TRUSTED_RECORD_EXISTS").status == RuleStatus.PASS
    assert _by_id(results, "TRUSTED_RECORD_DOB").status == RuleStatus.PASS
    assert _by_id(results, "TRUSTED_RECORD_STATUS").status == RuleStatus.PASS


def test_trusted_record_dob_mismatch():
    # Document number exists in the mock DB, but DOB differs -> DOB mismatch,
    # record still exists and its status is still reported.
    r = genuine_response()
    r.fields["date_of_birth"] = field("01 JAN 1990")
    r.mrz.fields.date_of_birth = "900101"  # keep MRZ consistent with visual
    results = RulesEngine().evaluate(r)
    assert _by_id(results, "TRUSTED_RECORD_EXISTS").status == RuleStatus.PASS
    assert _by_id(results, "TRUSTED_RECORD_DOB").status == RuleStatus.FAIL


def test_engine_summary_counts():
    results = RulesEngine().evaluate(genuine_response())
    assert results.summary.total == len(results.findings)
    assert results.summary.by_status.get("PASS", 0) == len(results.findings)


def test_rule_set_selection_extensible():
    engine = RulesEngine()
    passport_groups = engine.select_rule_groups("passport")
    unknown_groups = engine.select_rule_groups("something_unknown")
    # Unknown types fall back to the passport rule set (Phase 1 default).
    assert passport_groups == unknown_groups
