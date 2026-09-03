"""OCR result representation.

These models are the internal contract between the OCR engine and everything
downstream (field extraction, MRZ detection, API response). Nothing here is
tied to a specific OCR library, so the engine can be swapped later.

A bounding box is ``[x1, y1, x2, y2]`` in **original image pixel coordinates**
(top-left origin). Confidence is normalised to ``0.0 .. 1.0``.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

BBox = List[int]  # [x1, y1, x2, y2]


class OCRWord(BaseModel):
    """A single recognised token with its location and confidence."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    line_id: int = 0  # words sharing a line_id belong to the same text line

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]


class OCRLine(BaseModel):
    """Words grouped into a text line, with a merged bounding box."""

    text: str
    confidence: float
    bbox: BBox
    words: List[OCRWord]


class OCRResult(BaseModel):
    """Full OCR output for one image."""

    words: List[OCRWord] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """All text joined line-by-line, preserving reading order."""
        return "\n".join(line.text for line in self.lines())

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    def lines(self) -> List[OCRLine]:
        """Group words into lines using their ``line_id``."""
        buckets: dict[int, List[OCRWord]] = {}
        for w in self.words:
            buckets.setdefault(w.line_id, []).append(w)

        lines: List[OCRLine] = []
        for line_id in sorted(buckets):
            words = sorted(buckets[line_id], key=lambda w: w.x1)
            text = " ".join(w.text for w in words)
            conf = sum(w.confidence for w in words) / len(words)
            x1 = min(w.x1 for w in words)
            y1 = min(w.y1 for w in words)
            x2 = max(w.x2 for w in words)
            y2 = max(w.y2 for w in words)
            lines.append(
                OCRLine(text=text, confidence=conf, bbox=[x1, y1, x2, y2], words=words)
            )
        # Sort lines top-to-bottom for stable reading order.
        lines.sort(key=lambda ln: ln.bbox[1])
        return lines
