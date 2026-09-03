"""Passport field extraction."""
from __future__ import annotations

from app.ocr.extractor import PASSPORT_FIELDS, get_extractor
from app.ocr.mrz import detect_mrz
from tests.conftest import build_passport_ocr_result


def test_passport_fields_extracted_with_bboxes():
    ocr = build_passport_ocr_result()
    mrz = detect_mrz(ocr)
    fields = get_extractor("passport").extract(ocr, mrz)

    # All target Phase-1 fields present.
    for name in PASSPORT_FIELDS:
        assert name in fields, f"missing field: {name}"

    # Every field carries value, confidence and a bounding box.
    for f in fields.values():
        assert f.value
        assert 0.0 <= f.confidence <= 1.0
        assert f.bbox is not None and len(f.bbox) == 4

    assert fields["nationality"].value == "UTO"
    assert "1974" in fields["date_of_birth"].value
    assert "ERIKSSON" in fields["name"].value


def test_mrz_fallback_marks_source():
    """With no visual nationality line, the value comes from the MRZ."""
    ocr = build_passport_ocr_result()
    # Drop the visual "Nationality" line to force MRZ fallback.
    ocr.words = [w for w in ocr.words if "Nationality" not in w.text]
    mrz = detect_mrz(ocr)
    fields = get_extractor("passport").extract(ocr, mrz)

    assert fields["nationality"].value == "UTO"
    assert fields["nationality"].source == "mrz"
