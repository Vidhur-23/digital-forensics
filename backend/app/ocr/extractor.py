"""Field extraction: turn OCR text into labelled document fields.

The OCR engine answers *"what text is visible?"*. The extractor answers
*"what does that text represent?"* — mapping tokens to fields like name,
date_of_birth, document_number, etc., each carrying its source bounding box and
confidence so Phase 2 (rules) and Phase 3 (forensics) can consume them.

:class:`FieldExtractor` is the interface; the passport extractor is the only
implementation and is selected by document type in the pipeline.
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
        """Assign a date to each labelled date field.

        Passports do not reliably print a label and its date on one OCR line: the
        value often sits to the *right* of the label, or on the line *below* it.
        We try two strategies, most reliable first, and never reuse a date line
        that has already been assigned to another field:

          1. label keyword and a date on the **same** line;
          2. a label line (keyword present, date maybe not) paired with the
             nearest date on a **neighbouring** line — to its right on the same
             row, or directly beneath it.
        """
        dated = [(ln, d) for ln in lines if (d := _find_date(ln.text))]
        claimed: set[int] = set()

        # Pass 1: label + date share one line (most reliable).
        for field, keywords in self._DATE_LABELS.items():
            if field in out:
                continue
            for ln, date in dated:
                if id(ln) in claimed:
                    continue
                if any(k in ln.text.lower() for k in keywords):
                    out[field] = ExtractedField(
                        value=date, confidence=ln.confidence, bbox=ln.bbox
                    )
                    claimed.add(id(ln))
                    break

        # Pass 2: label on one line, its date on a neighbouring line.
        for field, keywords in self._DATE_LABELS.items():
            if field in out:
                continue
            label = next(
                (ln for ln in lines if any(k in ln.text.lower() for k in keywords)),
                None,
            )
            if label is None:
                continue
            pick = self._nearest_date_line(label, dated, claimed)
            if pick is not None:
                ln, date = pick
                out[field] = ExtractedField(
                    value=date, confidence=ln.confidence, bbox=ln.bbox
                )
                claimed.add(id(ln))

    @staticmethod
    def _nearest_date_line(
        label: OCRLine,
        dated: List[tuple],
        claimed: set,
    ) -> Optional[tuple]:
        """Nearest unclaimed date line that reads as ``label``'s value.

        Candidates are the date on the same row to the label's right, or on a
        line below it (within ~3x the label height so we don't reach across the
        page). Same-row-right wins over below; ties break on proximity. Dates
        above the label are never its value.
        """
        lx1, ly1, lx2, ly2 = label.bbox
        label_height = max(ly2 - ly1, 1)
        label_cy = (ly1 + ly2) / 2

        best: Optional[tuple] = None
        best_cost: Optional[tuple] = None
        for ln, date in dated:
            if id(ln) in claimed or ln is label:
                continue
            cx1, cy1, cx2, cy2 = ln.bbox
            cand_cy = (cy1 + cy2) / 2
            same_row = cy1 <= label_cy <= cy2 or ly1 <= cand_cy <= ly2
            if same_row and cx1 >= lx1:
                cost = (0, max(cx1 - lx2, 0))                      # value to the right
            elif cy1 >= label_cy:
                gap = cy1 - ly2
                if gap > 3 * label_height:
                    continue                                        # too far below
                cost = (1, gap + abs(cx1 - lx1) * 0.25)             # value on a line below
            else:
                continue                                            # above -> not its value
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, (ln, date)
        return best

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
