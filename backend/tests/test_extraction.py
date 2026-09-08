"""Passport field extraction."""
from __future__ import annotations

from app.ocr.extractor import PASSPORT_FIELDS, get_extractor
from app.ocr.mrz import detect_mrz
from app.ocr.schemas import OCRResult, OCRWord
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


def _word(text: str, line_id: int, x: int, y: int) -> OCRWord:
    return OCRWord(text=text, confidence=0.95, bbox=[x, y, x + 12 * len(text), y + 30], line_id=line_id)


def test_issue_date_label_and_value_on_separate_lines():
    """Value printed on the line *below* its label is still captured (pass 2)."""
    ocr = OCRResult(
        words=[
            _word("Date of Issue", 0, 100, 100),
            _word("15 APR 2007", 1, 100, 140),   # value directly below the label
            _word("Date of Expiry", 2, 100, 200),
            _word("15 APR 2012", 3, 100, 240),
        ]
    )
    fields = get_extractor("passport").extract(ocr, detect_mrz(ocr))
    assert fields["issue_date"].value == "15 APR 2007"
    assert fields["expiry_date"].value == "15 APR 2012"


def test_date_value_to_the_right_of_label():
    """Value printed to the right of its label on the same row is captured."""
    ocr = OCRResult(
        words=[
            _word("Date of Birth", 0, 100, 100),
            _word("12 AUG 1974", 1, 500, 100),   # same row, to the right
        ]
    )
    fields = get_extractor("passport").extract(ocr, detect_mrz(ocr))
    assert fields["date_of_birth"].value == "12 AUG 1974"


def test_mrz_fallback_marks_source():
    """With no visual nationality line, the value comes from the MRZ."""
    ocr = build_passport_ocr_result()
    # Drop the visual "Nationality" line to force MRZ fallback.
    ocr.words = [w for w in ocr.words if "Nationality" not in w.text]
    mrz = detect_mrz(ocr)
    fields = get_extractor("passport").extract(ocr, mrz)

    assert fields["nationality"].value == "UTO"
    assert fields["nationality"].source == "mrz"
