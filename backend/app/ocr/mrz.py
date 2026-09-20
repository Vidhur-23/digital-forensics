"""MRZ (Machine Readable Zone) detection and extraction.

Phase 1 scope: **detect** the MRZ region, return its raw text + bounding box,
and best-effort parse the visible field values. This performs NO validation:
no check-digit verification, no cross-field consistency, no authenticity call.
Those belong to the Phase 2 rules engine, which will consume the raw values
produced here.

Supported layout: TD3 (passport) — two lines of 44 characters. The parser is
tolerant of OCR noise and simply extracts substrings; it never rejects input.
"""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from app.ocr.schemas import BBox, OCRResult

# A candidate MRZ line: mostly A-Z, 0-9 and the filler '<', reasonably long.
_MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{25,}$")

# TD3 lines are exactly 44 characters. OCR may drop or add a few, so we accept a
# tolerance band when deciding which candidate lines are the *real* MRZ.
_TD3_LEN = 44
_TD3_MIN = 38
_TD3_MAX = 48


def _select_mrz_lines(candidates: List) -> List:
    """Pick the two lines that actually make up a TD3 MRZ.

    ``candidates`` is a list of ``(OCRLine, normalised_text)`` in top-to-bottom
    reading order. We must not simply grab the last two charset-matching lines:
    a long serial/barcode/control-number below the MRZ, or the MRZ arriving as a
    single merged line, both break that assumption.
    """
    # Case 1: the two rows were merged into one ~88-char line. Split in half.
    if len(candidates) == 1:
        ln, text = candidates[0]
        if 2 * _TD3_MIN <= len(text) <= 2 * _TD3_MAX:
            mid = len(text) // 2
            return [(ln, text[:mid]), (ln, text[mid:])]

    # Case 2: prefer candidates whose length matches a TD3 line, which excludes
    # stray serials/barcodes that happen to use the MRZ character set.
    strong = [c for c in candidates if _TD3_MIN <= len(c[1]) <= _TD3_MAX]
    if len(strong) >= 2:
        return strong[-2:]

    # Fallback: keep prior behaviour — the last two candidate lines.
    return candidates[-2:]


class MRZFields(BaseModel):
    """Raw values read out of the MRZ. Empty string == not present/unreadable."""

    document_type: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    document_number: str = ""
    nationality: str = ""
    date_of_birth: str = ""  # raw YYMMDD as printed in the MRZ
    sex: str = ""
    expiry_date: str = ""  # raw YYMMDD as printed in the MRZ


class MRZResult(BaseModel):
    detected: bool = False
    format: Optional[str] = None  # e.g. "TD3"
    text: str = ""  # raw MRZ lines joined by newline
    lines: List[str] = Field(default_factory=list)
    bbox: Optional[BBox] = None  # region covering the MRZ in original pixels
    fields: MRZFields = Field(default_factory=MRZFields)


def _clean(token: str) -> str:
    """Trim MRZ filler characters and surrounding whitespace."""
    return token.replace("<", " ").strip()


def _parse_td3(line1: str, line2: str) -> MRZFields:
    """Parse the two TD3 lines. Positional, tolerant, no validation."""
    f = MRZFields()

    # Line 1: P<ISS SURNAME<<GIVEN<NAMES<<<...
    if line1:
        f.document_type = line1[0:1].replace("<", "")
        f.issuing_country = _clean(line1[2:5])
        names = line1[5:]
        if "<<" in names:
            surname, _, given = names.partition("<<")
            f.surname = _clean(surname)
            f.given_names = _clean(given)
        else:
            f.surname = _clean(names)

    # Line 2: doc_no(9) chk(1) nat(3) dob(6) chk(1) sex(1) exp(6) chk(1) ...
    if line2:
        f.document_number = _clean(line2[0:9])
        f.nationality = _clean(line2[10:13])
        f.date_of_birth = _clean(line2[13:19])
        f.sex = _clean(line2[20:21])
        f.expiry_date = _clean(line2[21:27])

    return f


def detect_mrz(ocr_result: OCRResult) -> MRZResult:
    """Locate MRZ lines within OCR output and extract raw fields.

    We scan OCR lines (bottom of the document first, where the MRZ lives) for
    tokens matching the MRZ character set and pick the last consecutive block of
    such lines.
    """
    candidates = []  # (OCRLine, normalised_text)
    for line in ocr_result.lines():
        # Collapse spaces — OCR often splits the MRZ into words.
        normalised = re.sub(r"\s+", "", line.text.upper())
        if _MRZ_LINE_RE.match(normalised):
            candidates.append((line, normalised))

    if not candidates:
        return MRZResult(detected=False)

    # Select the two lines that actually form the TD3 MRZ, tolerating a stray
    # serial line below the zone or the two rows arriving merged as one line.
    mrz_lines = _select_mrz_lines(candidates)
    texts = [t for _, t in mrz_lines]

    # Merge bounding boxes into one region.
    boxes = [ln.bbox for ln, _ in mrz_lines]
    bbox: BBox = [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]

    fmt = None
    fields = MRZFields()
    if len(texts) == 2:
        fmt = "TD3"
        fields = _parse_td3(texts[0], texts[1])

    return MRZResult(
        detected=True,
        format=fmt,
        text="\n".join(texts),
        lines=texts,
        bbox=bbox,
        fields=fields,
    )
