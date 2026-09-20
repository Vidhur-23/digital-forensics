"""MRZ detection + raw field extraction (no validation)."""
from __future__ import annotations

from app.ocr.mrz import detect_mrz
from app.ocr.schemas import OCRResult, OCRWord
from tests.conftest import build_passport_ocr_result


def test_mrz_detected_from_passport_ocr():
    mrz = detect_mrz(build_passport_ocr_result())
    assert mrz.detected is True
    assert mrz.format == "TD3"
    assert mrz.bbox is not None and len(mrz.bbox) == 4
    assert len(mrz.lines) == 2


def test_mrz_fields_parsed():
    mrz = detect_mrz(build_passport_ocr_result())
    f = mrz.fields
    assert f.document_type == "P"
    assert f.issuing_country == "UTO"
    assert f.surname == "ERIKSSON"
    assert "ANNA" in f.given_names
    assert f.document_number == "L898902C3"
    assert f.nationality == "UTO"
    assert f.date_of_birth == "740812"
    assert f.expiry_date == "120415"
    assert f.sex == "F"


def test_no_mrz_when_absent():
    words = [OCRWord(text="Just a caption", confidence=0.9, bbox=[0, 0, 100, 20])]
    mrz = detect_mrz(OCRResult(words=words))
    assert mrz.detected is False


# --- robustness against noisy OCR line selection --------------------------

_MRZ_1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
_MRZ_2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def _word(text: str, line_id: int, y: int) -> OCRWord:
    return OCRWord(
        text=text, confidence=0.95, bbox=[100, y, 100 + 12 * len(text), y + 30],
        line_id=line_id,
    )


def test_mrz_ignores_serial_line_below_zone():
    """A long alphanumeric line below the MRZ must not hijack detection."""
    words = [
        _word(_MRZ_1, 0, 900),
        _word(_MRZ_2, 1, 940),
        _word("MRZBARCODE0000000000000000000000000", 2, 980),
    ]
    f = detect_mrz(OCRResult(words=words)).fields
    assert f.document_number == "L898902C3"
    assert f.date_of_birth == "740812"
    assert f.sex == "F"


def test_mrz_handles_two_rows_merged_into_one_line():
    """When OCR merges both TD3 rows into a single line it is split correctly."""
    words = [_word(_MRZ_1, 0, 900), _word(_MRZ_2, 0, 900)]
    mrz = detect_mrz(OCRResult(words=words))
    assert mrz.format == "TD3"
    assert len(mrz.lines) == 2
    assert mrz.fields.document_number == "L898902C3"
    assert mrz.fields.sex == "F"
