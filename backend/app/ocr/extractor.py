"""Field extraction: turn OCR text into labelled document fields.

The OCR engine answers *"what text is visible?"*. The extractor answers
*"what does that text represent?"* — mapping tokens to fields like name,
date_of_birth, document_number, etc., each carrying its source bounding box and
confidence so Phase 2 (rules) and Phase 3 (forensics) can consume them.

Extensibility: :class:`FieldExtractor` is the interface; register a new
subclass (driving licence, visa, national ID, ...) and select it by document
type in the pipeline. Only the passport extractor is implemented in Phase 1.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel

from app.ocr.mrz import MRZResult
from app.ocr.schemas import BBox, OCRLine, OCRResult

# Passport fields we attempt to populate in Phase 1.
PASSPORT_FIELDS = (
    "name",
    "date_of_birth",
    "document_number",
    "nationality",
    "issue_date",
    "expiry_date",
)


class ExtractedField(BaseModel):
    value: str
    confidence: float
    bbox: Optional[BBox] = None
    source: str = "visual"  # "visual" (OCR text) or "mrz"


# --- helpers ---------------------------------------------------------------

# Dates like "12 APR 1998", "12/04/1998", "12-04-1998", "1998-04-12".
_DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}[ /.\-][A-Z]{3}[ /.\-]\d{2,4}"  # 12 APR 1998
    r"|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"    # 12/04/1998
    r"|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}"      # 1998-04-12
    r")\b"
)
_DOC_NO_RE = re.compile(r"\b([A-Z]{1,2}[0-9]{6,8}|[0-9]{7,9})\b")
_NATIONALITY_RE = re.compile(r"\b([A-Z]{3})\b")


def _find_date(text: str) -> Optional[str]:
    m = _DATE_RE.search(text.upper())
    return m.group(1) if m else None


class FieldExtractor(ABC):
    """Interface for document-type-specific field extraction."""

    document_type: str = "generic"

    @abstractmethod
    def extract(
        self, ocr: OCRResult, mrz: MRZResult
    ) -> Dict[str, ExtractedField]:  # pragma: no cover - interface
        ...


class PassportFieldExtractor(FieldExtractor):
    """Heuristic passport extractor.

    Strategy (in priority order per field):
      1. Label-anchored visual read — find a keyword line and read the value,
         keeping the OCR bounding box.
      2. MRZ fallback — reliable machine-readable values, tagged source="mrz"
         and located at the MRZ bounding box.
    """

    document_type = "passport"

    # keyword -> field for label-anchored dates / values
    _DATE_LABELS = {
        "date_of_birth": ("birth", "dob"),
        "issue_date": ("issue", "issued"),
        "expiry_date": ("expiry", "expiration", "expire", "valid until"),
    }

    def extract(self, ocr: OCRResult, mrz: MRZResult) -> Dict[str, ExtractedField]:
        lines = ocr.lines()
        fields: Dict[str, ExtractedField] = {}

        self._extract_dates(lines, fields)
        self._extract_document_number(lines, fields)
        self._extract_nationality(lines, fields)
        self._extract_name(lines, fields)

        self._fill_from_mrz(mrz, fields)
        return fields

    # --- visual passes -----------------------------------------------------

    def _extract_dates(self, lines: List[OCRLine], out: Dict[str, ExtractedField]):
        for line in lines:
            low = line.text.lower()
            date = _find_date(line.text)
            if not date:
                continue
            for field, keywords in self._DATE_LABELS.items():
                if field in out:
                    continue
                if any(k in low for k in keywords):
                    out[field] = ExtractedField(
                        value=date, confidence=line.confidence, bbox=line.bbox
                    )

    def _extract_document_number(self, lines, out):
        for line in lines:
            low = line.text.lower()
            if "passport" in low or re.search(r"\bno\b|\bnumber\b|\bno\.", low):
                m = _DOC_NO_RE.search(line.text.upper())
                if m:
                    out["document_number"] = ExtractedField(
                        value=m.group(1), confidence=line.confidence, bbox=line.bbox
                    )
                    return

    def _extract_nationality(self, lines, out):
        for line in lines:
            if "national" in line.text.lower():
                m = _NATIONALITY_RE.search(line.text.upper())
                if m:
                    out["nationality"] = ExtractedField(
                        value=m.group(1), confidence=line.confidence, bbox=line.bbox
                    )
                    return

    def _extract_name(self, lines, out):
        for line in lines:
            low = line.text.lower()
            if "surname" in low or low.strip().startswith("name"):
                # Value is the uppercase run after the label on the same line.
                m = re.search(r"[:\-]?\s*([A-Z][A-Z ]{2,})$", line.text.strip())
                if m:
                    out["name"] = ExtractedField(
                        value=m.group(1).strip(),
                        confidence=line.confidence,
                        bbox=line.bbox,
                    )
                    return

    # --- MRZ fallback ------------------------------------------------------

    def _fill_from_mrz(self, mrz: MRZResult, out: Dict[str, ExtractedField]):
        if not mrz.detected:
            return
        f = mrz.fields
        conf = 0.90  # MRZ is machine-readable; treat as high-confidence
        bbox = mrz.bbox

        def put(field: str, value: str):
            if value and field not in out:
                out[field] = ExtractedField(
                    value=value, confidence=conf, bbox=bbox, source="mrz"
                )

        name = " ".join(p for p in (f.given_names, f.surname) if p).strip()
        put("name", name)
        put("document_number", f.document_number)
        put("nationality", f.nationality)
        put("date_of_birth", f.date_of_birth)
        put("expiry_date", f.expiry_date)


# Registry so the pipeline can select an extractor by document type and new
# document templates can be added without touching the pipeline.
_EXTRACTORS: Dict[str, FieldExtractor] = {
    "passport": PassportFieldExtractor(),
}


def get_extractor(document_type: str) -> FieldExtractor:
    """Return the extractor for a document type (passport default in Phase 1)."""
    return _EXTRACTORS.get(document_type, _EXTRACTORS["passport"])
