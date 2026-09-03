"""Shared FastAPI dependencies (Phase 1).

Provides a lazily-constructed singleton screening pipeline backed by Tesseract.
Tests override :func:`get_pipeline` to inject a deterministic OCR stub.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.ocr.engine import TesseractOCREngine
from app.pipeline.pipeline import ScreeningPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> ScreeningPipeline:
    engine = TesseractOCREngine(tesseract_cmd=settings.tesseract_cmd)
    return ScreeningPipeline(ocr_engine=engine)
