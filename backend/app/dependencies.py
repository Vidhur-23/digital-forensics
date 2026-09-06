"""Shared FastAPI dependencies (Phase 1).

Provides a lazily-constructed singleton screening pipeline backed by Tesseract.
Tests override :func:`get_pipeline` to inject a deterministic OCR stub.
"""
from __future__ import annotations

from functools import lru_cache

from app.biometrics.service import InsightFaceBiometricService
from app.config import settings
from app.forensics.engine import ForensicLayerService
from app.intelligence.engine import IntelligenceEngine
from app.intelligence.llm.providers import build_provider
from app.intelligence.llm.service import LLMEvaluator
from app.ocr.engine import TesseractOCREngine
from app.pipeline.pipeline import ScreeningPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> ScreeningPipeline:
    engine = TesseractOCREngine(tesseract_cmd=settings.tesseract_cmd)
    # One forensic service per process: the adapter imports the (heavy) CV
    # signal modules once on first use and reuses them across requests.
    forensics = ForensicLayerService()
    # One biometric service per process: the InsightFace model is loaded lazily
    # on first use (when a reference image is supplied) and reused thereafter.
    biometrics = InsightFaceBiometricService()
    # Phase 5: the intelligence engine's deterministic core always runs; the LLM
    # provider is selected from configuration (Null/offline unless APP_LLM_* is
    # set), so no network or API key is required for the pipeline to work.
    intelligence = IntelligenceEngine(LLMEvaluator(build_provider(settings)))
    return ScreeningPipeline(
        ocr_engine=engine,
        forensic_service=forensics,
        biometric_service=biometrics,
        intelligence_engine=intelligence,
    )
