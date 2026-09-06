"""Pydantic response models for Phase 3 forensic evidence.

These models are a thin, typed *view* over the Evidence Object produced by the
teammate's forensic engine (``forensic_layer/pipeline/forensics_engine.py``).
They deliberately mirror that engine's output rather than re-interpreting it, so
the forensic layer stays the source of truth.

Design contract (consistent with the Phase 2 rule schema):
* A forensic finding is **manipulation evidence, not a verdict**. ``passed =
  False`` means a signal layer flagged an anomaly worth review — it never means
  "the document is fake". Combined assessment is reserved for the later
  intelligence / evidence-fusion phase.
* ``status`` distinguishes a forensic *result* from a forensic *outage*. An
  engine that could not run at all is ``UNAVAILABLE`` — which is NOT the same as
  a clean "no manipulation detected" result.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.ocr.schemas import BBox


class ForensicStatus(str, Enum):
    """Whether the forensic engine actually ran for this document."""

    # The engine executed and produced findings (individual layers may still
    # have failed — those are represented as ``processing_error`` findings).
    COMPLETED = "COMPLETED"
    # The engine could not run at all (not importable, crashed before producing
    # an Evidence Object, calibration artifacts missing, etc.). This is an
    # outage, explicitly NOT a clean result.
    UNAVAILABLE = "UNAVAILABLE"


class ForensicFinding(BaseModel):
    """A single forensic signal-layer result.

    Mirrors one entry of the engine's ``findings`` list. ``region`` is the
    suspicious bounding box in ORIGINAL image pixel coordinates (``[x1, y1, x2,
    y2]``, the same convention as OCR/field bboxes) when a layer localises one;
    ``None`` otherwise.
    """

    layer_name: str
    finding_type: str
    passed: bool
    score: float
    severity: str
    confidence: float
    # Suspicious region in original-image pixels, when the layer localises one.
    region: Optional[BBox] = None
    # Supporting signal tags (derived from the layer's finding type) so a
    # reviewer/frontend can group evidence without parsing free text.
    signals: List[str] = Field(default_factory=list)
    # The layer's raw measurements (distances, thresholds, counts, ...).
    raw_metric: Dict[str, Any] = Field(default_factory=dict)
    # Human-readable explanation (the engine's ``explanation``).
    message: str = ""
    # Per-layer engine status: "ok" or "error".
    status: str = "ok"


class ForensicSummary(BaseModel):
    """Aggregate counts over the forensic layers — a tally, not a verdict."""

    layers_requested: int = 0
    layers_completed: int = 0
    layers_failed: int = 0
    flagged_count: int = 0
    # Did every layer pass? (Still evidence, not a "genuine" verdict.)
    all_passed: bool = False


class ForensicResults(BaseModel):
    """Phase 3 forensic output: layer findings plus a non-verdict summary.

    Attached to the screening response as ``forensics``. When the engine could
    not run, ``status`` is ``UNAVAILABLE`` and ``error`` explains why — callers
    must treat that as "unknown", never as "clean".
    """

    status: ForensicStatus
    engine_version: Optional[str] = None
    findings: List[ForensicFinding] = Field(default_factory=list)
    summary: ForensicSummary = Field(default_factory=ForensicSummary)
    elapsed_seconds: Optional[float] = None
    # Populated only when ``status == UNAVAILABLE``.
    error: Optional[str] = None

    @classmethod
    def unavailable(cls, error: str) -> "ForensicResults":
        """Build an explicit forensic-outage result (never a clean result)."""
        return cls(status=ForensicStatus.UNAVAILABLE, error=error)
