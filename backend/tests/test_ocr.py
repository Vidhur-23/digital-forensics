"""OCR result structure + real Tesseract smoke test."""
from __future__ import annotations

import numpy as np
import pytest

from app.ocr.schemas import OCRResult, OCRWord


def test_ocr_result_has_text_confidence_bbox(passport_ocr_result: OCRResult):
    assert passport_ocr_result.words
    for w in passport_ocr_result.words:
        assert isinstance(w.text, str) and w.text
        assert 0.0 <= w.confidence <= 1.0
        assert len(w.bbox) == 4
    assert 0.0 <= passport_ocr_result.mean_confidence <= 1.0


def test_lines_grouped_and_sorted_top_to_bottom():
    words = [
        OCRWord(text="B", confidence=0.9, bbox=[0, 200, 10, 220], line_id=1),
        OCRWord(text="A", confidence=0.9, bbox=[0, 10, 10, 30], line_id=0),
        OCRWord(text="A2", confidence=0.9, bbox=[20, 10, 30, 30], line_id=0),
    ]
    lines = OCRResult(words=words).lines()
    assert [ln.text for ln in lines] == ["A A2", "B"]
    # merged bbox for the first line spans both words
    assert lines[0].bbox == [0, 10, 30, 30]


def _tesseract_available() -> bool:
    try:
        from app.ocr.engine import TesseractOCREngine

        return TesseractOCREngine().is_available()
    except Exception:
        return False


@pytest.mark.skipif(
    not _tesseract_available(), reason="tesseract binary not installed"
)
def test_real_tesseract_reads_text_with_boxes():
    """End-to-end OCR on a synthetic image (runs only if tesseract present)."""
    from PIL import Image, ImageDraw

    from app.ocr.engine import TesseractOCREngine

    img = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(img).text((30, 80), "HELLO WORLD", fill="black")
    arr = np.array(img)

    result = TesseractOCREngine().extract(arr)
    joined = " ".join(w.text.upper() for w in result.words)
    assert "HELLO" in joined
    assert all(len(w.bbox) == 4 for w in result.words)
    assert all(0.0 <= w.confidence <= 1.0 for w in result.words)
