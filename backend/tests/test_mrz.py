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
