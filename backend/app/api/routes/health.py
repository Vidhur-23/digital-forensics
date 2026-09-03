"""Health / readiness endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Basic liveness check plus whether the OCR backend is ready."""
    tesseract_ready = False
    try:
        from app.ocr.engine import TesseractOCREngine

        tesseract_ready = TesseractOCREngine(
            tesseract_cmd=settings.tesseract_cmd
        ).is_available()
    except Exception:
        tesseract_ready = False

    return {
        "status": "ok",
        "app": settings.app_name,
        "ocr_backend": "tesseract",
        "ocr_ready": tesseract_ready,
    }
