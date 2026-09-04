"""Phase 2: visual vs MRZ cross-field consistency."""
from __future__ import annotations

from app.rules.consistency import check_consistency
from app.rules.schemas import RuleStatus
from tests.rules_helpers import field, make_mrz, make_response


def _find(findings, rule_id):
    return next(f for f in findings if f.rule_id == rule_id)


def _mrz():
    return make_mrz("L898902C3", "UTO", "740812", "F", "300101", "DOE", "JOHN")


def test_name_normalization_passes():
    # Visual "JOHN DOE" vs MRZ "DOE<<JOHN" -> PASS (order/format ignored).
    r = make_response({"name": field("JOHN DOE")}, _mrz())
    assert _find(check_consistency(r), "CONSISTENCY_NAME").status == RuleStatus.PASS


def test_name_conflict_fails():
    r = make_response({"name": field("JANE SMITH")}, _mrz())
    assert _find(check_consistency(r), "CONSISTENCY_NAME").status == RuleStatus.FAIL


def test_dob_match_passes():
    r = make_response({"date_of_birth": field("12 AUG 1974")}, _mrz())
    assert _find(check_consistency(r), "CONSISTENCY_DOB").status == RuleStatus.PASS


def test_dob_mismatch_fails():
    # Visual 1998-04-12 vs MRZ dob 740812 (1974-08-12).
    r = make_response({"date_of_birth": field("1998-04-12")}, _mrz())
    f = _find(check_consistency(r), "CONSISTENCY_DOB")
    assert f.status == RuleStatus.FAIL
    assert f.evidence["visual_value"] == "1998-04-12"
    assert f.evidence["mrz_value"] == "1974-08-12"


def test_document_number_mismatch_fails():
    r = make_response({"document_number": field("Z9999999")}, _mrz())
    assert _find(check_consistency(r), "CONSISTENCY_DOCUMENT_NUMBER").status == RuleStatus.FAIL


def test_document_number_match_passes():
    r = make_response({"document_number": field("L898902C3")}, _mrz())
    assert _find(check_consistency(r), "CONSISTENCY_DOCUMENT_NUMBER").status == RuleStatus.PASS


def test_nationality_mismatch_fails():
    r = make_response({"nationality": field("XXX")}, _mrz())
    assert _find(check_consistency(r), "CONSISTENCY_NATIONALITY").status == RuleStatus.FAIL


def test_expiry_mismatch_fails():
    r = make_response({"expiry_date": field("01 JAN 2029")}, _mrz())  # MRZ says 2030-01-01
    assert _find(check_consistency(r), "CONSISTENCY_EXPIRY").status == RuleStatus.FAIL


def test_mrz_sourced_field_not_applicable():
    # A field filled from the MRZ can't be independently cross-checked.
    r = make_response({"document_number": field("L898902C3", source="mrz")}, _mrz())
    assert _find(check_consistency(r), "CONSISTENCY_DOCUMENT_NUMBER").status == RuleStatus.NOT_APPLICABLE


def test_no_mrz_makes_consistency_not_applicable():
    r = make_response({"name": field("JOHN DOE")})  # no MRZ
    findings = check_consistency(r)
    assert findings and findings[0].status == RuleStatus.NOT_APPLICABLE
