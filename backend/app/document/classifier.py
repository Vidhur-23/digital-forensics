"""Document type identification.

Phase 1 targets the passport. The classifier is deliberately simple and signal
based (MRZ presence + keywords) but returns a structured result so more document
types (driving licence, visa, national ID) can be added later without changing
callers.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ocr.mrz import MRZResult
from app.ocr.schemas import OCRResult


@dataclass
class ClassificationResult:
    document_type: str
    confidence: float


# Keyword hints per document type (extend this map for new types).
_KEYWORDS = {
    "passport": ("passport", "passeport", "pasaporte"),
}


def classify_document(ocr: OCRResult, mrz: MRZResult) -> ClassificationResult:
    text = ocr.full_text.lower()

    has_passport_kw = any(k in text for k in _KEYWORDS["passport"])
    # A TD3 MRZ starting with 'P' is a strong passport signal.
    mrz_is_passport = mrz.detected and mrz.fields.document_type.upper().startswith("P")

    if mrz_is_passport and has_passport_kw:
        return ClassificationResult("passport", 0.98)
    if mrz_is_passport or has_passport_kw:
        return ClassificationResult("passport", 0.85)

    # Phase 1 demo default: assume passport but flag low confidence so Phase 2
    # can decide whether to trust it.
    return ClassificationResult("passport", 0.40)
