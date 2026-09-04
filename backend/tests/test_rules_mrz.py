"""Phase 2: MRZ parsing and ICAO 9303 check-digit validation."""
from __future__ import annotations

from app.rules.mrz import check_mrz, compute_check_digit, parse_td3
from app.rules.schemas import RuleStatus
from tests.rules_helpers import make_mrz, make_response

# Known-valid ICAO TD3 specimen (Utopia / ERIKSSON). Independently documented as
# a fully valid MRZ, so it validates the check-digit algorithm itself.
SPEC_L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
SPEC_L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_compute_check_digit_known_values():
    # Documented ICAO worked examples.
    assert compute_check_digit("D23145890734") == "9"
    assert compute_check_digit("340712") == "7"


def test_parse_td3_specimen_fields():
    p = parse_td3(SPEC_L1, SPEC_L2)
    assert p.format == "TD3"
    assert p.document_number == "L898902C3"
    assert p.nationality == "UTO"
    assert p.date_of_birth == "740812"
    assert p.expiry_date == "120415"
    assert p.sex == "F"
    assert p.surname == "ERIKSSON"
    assert "ANNA" in p.given_names


def test_specimen_all_check_digits_valid():
    p = parse_td3(SPEC_L1, SPEC_L2)
    assert p.check_digits, "expected check digits to be computed"
    for chk in p.check_digits:
        assert chk.ok, f"{chk.name}: expected {chk.expected} computed {chk.computed}"


def test_invalid_check_digit_detected():
    # Corrupt the document-number check digit (position 9: '6' -> '5').
    bad_l2 = SPEC_L2[:9] + "5" + SPEC_L2[10:]
    p = parse_td3(SPEC_L1, bad_l2)
    doc_chk = next(c for c in p.check_digits if c.name == "document_number")
    assert doc_chk.ok is False


def test_check_mrz_all_pass_on_generated_valid_mrz():
    mrz = make_mrz("L898902C3", "UTO", "740812", "F", "300101", "ERIKSSON", "ANNA MARIA")
    findings = check_mrz(make_response({}, mrz))
    checks = [f for f in findings if f.rule_id.startswith("MRZ_CHECK_")]
    assert checks and all(f.status == RuleStatus.PASS for f in checks)


def test_check_mrz_reports_failed_check_digit():
    mrz = make_mrz("L898902C3", "UTO", "740812", "F", "300101", "ERIKSSON", "ANNA MARIA")
    # Break the DOB check digit inside the raw MRZ text (position 19 of line 2).
    line1, line2 = mrz.text.split("\n")
    line2 = line2[:19] + ("0" if line2[19] != "0" else "1") + line2[20:]
    mrz.text = f"{line1}\n{line2}"
    findings = check_mrz(make_response({}, mrz))
    dob_chk = next(f for f in findings if f.rule_id == "MRZ_CHECK_DATE_OF_BIRTH")
    assert dob_chk.status == RuleStatus.FAIL


def test_no_mrz_yields_warning_not_applicable():
    findings = check_mrz(make_response({}))
    assert any(f.rule_id == "MRZ_PRESENT" and f.status == RuleStatus.WARNING for f in findings)
