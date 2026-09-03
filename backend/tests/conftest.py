"""Shared test fixtures.

Provides a deterministic stub OCR engine (so field-extraction / MRZ / pipeline
logic is tested without needing the tesseract binary) and helpers to synthesise
image bytes and a passport-shaped OCR result. No real identity data is used.
"""
from __future__ import annotations

import io
from typing import List

import numpy as np
import pytest
from PIL import Image

from app.ocr.engine import OCREngine
from app.ocr.schemas import OCRResult, OCRWord

# --- synthetic passport OCR content (ICAO specimen values) -----------------

# Two TD3 MRZ lines (44 chars each) for the classic "UTOPIA / ERIKSSON" sample.
MRZ_LINE_1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
MRZ_LINE_2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"

# Visual lines (label + value) as they'd appear on the passport bio page.
_VISUAL_LINES = [
    "PASSPORT",
    "Surname: ERIKSSON",
    "Name: ANNA MARIA ERIKSSON",
    "Nationality: UTO",
    "Passport No: L898902C3",
    "Date of Birth: 12 AUG 1974",
    "Date of Issue: 15 APR 2007",
    "Date of Expiry: 15 APR 2012",
]


def _line_word(text: str, line_id: int, y: int) -> OCRWord:
    """Represent a whole visual line as one OCR word with a plausible bbox."""
    width = 12 * len(text)
    return OCRWord(
        text=text,
        confidence=0.95,
        bbox=[100, y, 100 + width, y + 30],
        line_id=line_id,
    )


def build_passport_ocr_result() -> OCRResult:
    words: List[OCRWord] = []
    y = 100
    line_id = 0
    for text in _VISUAL_LINES:
        words.append(_line_word(text, line_id, y))
        line_id += 1
        y += 50

    # MRZ region near the bottom of the page.
    words.append(_line_word(MRZ_LINE_1, line_id, 900))
    words.append(_line_word(MRZ_LINE_2, line_id + 1, 940))
    return OCRResult(words=words)


class StubOCREngine(OCREngine):
    """Returns a fixed passport OCR result regardless of the input image."""

    def __init__(self, result: OCRResult | None = None):
        self._result = result or build_passport_ocr_result()

    def extract(self, image: np.ndarray) -> OCRResult:  # noqa: D401
        return self._result

    def is_available(self) -> bool:
        return True


def make_image_bytes(width: int = 1000, height: int = 1000, fmt: str = "PNG") -> bytes:
    """A valid, decodable blank image (content is irrelevant to the stub)."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def stub_engine() -> StubOCREngine:
    return StubOCREngine()


@pytest.fixture
def passport_ocr_result() -> OCRResult:
    return build_passport_ocr_result()


@pytest.fixture
def image_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture
def client(stub_engine):
    """FastAPI TestClient with the pipeline overridden to use the stub engine."""
    from fastapi.testclient import TestClient

    from app.dependencies import get_pipeline
    from app.main import app
    from app.pipeline.pipeline import ScreeningPipeline

    app.dependency_overrides[get_pipeline] = lambda: ScreeningPipeline(stub_engine)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
