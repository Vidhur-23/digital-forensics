"""Pydantic response models for Phase 4 biometric face verification.

Why a dedicated model (and not a reuse of ``RuleFinding`` / ``ForensicFinding``):
this follows the project's established per-phase pattern — Phase 2 has
``RuleResults``, Phase 3 has ``ForensicResults``, and each is attached as an
optional field on the shared :class:`ScreeningResponse`. Biometrics is the
fourth peer. It is **not** a duplicate: it carries information no existing model
represents (two faces, a similarity score, and the biometric decision
vocabulary MATCH / MISMATCH / INSUFFICIENT_QUALITY / NO_FACE / AMBIGUOUS /
UNAVAILABLE). The generic status enums that already exist — ``RuleStatus``
(PASS/WARNING/FAIL) and ``ForensicStatus`` (COMPLETED/UNAVAILABLE) — cannot
express that vocabulary, which Phase 4 explicitly requires.

Shared primitives ARE reused: face regions use the project's ``BBox`` type
(``[x1, y1, x2, y2]`` in original image pixels), exactly like OCR/field/forensic
regions.

Design contract (same spirit as Phases 2/3):
* Biometrics reports **identity correspondence evidence, not a fraud verdict**.
  It never emits PASSPORT_GENUINE / PASSPORT_FAKE — combined assessment is a
  later phase.
* Poor input is reported as INSUFFICIENT_QUALITY / NO_FACE / UNAVAILABLE, never
  silently downgraded to MISMATCH.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.ocr.schemas import BBox


class BiometricStatus(str, Enum):
    """Outcome of the identity-correspondence comparison."""

    MATCH = "MATCH"                              # faces correspond (>= threshold)
    MISMATCH = "MISMATCH"                        # faces do not correspond
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"  # a face too poor to compare
    NO_FACE = "NO_FACE"                          # no face in one/both images
    AMBIGUOUS = "AMBIGUOUS"                      # >1 candidate face, none dominant
    UNAVAILABLE = "UNAVAILABLE"                  # engine/model could not run


class FaceQuality(str, Enum):
    """Fitness of a detected face for reliable comparison."""

    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNAVAILABLE = "UNAVAILABLE"


class FaceEvidence(BaseModel):
    """Per-image face detection + quality evidence.

    ``bbox`` is in the ORIGINAL image's pixel coordinates (``[x1, y1, x2, y2]``),
    the same convention as every other region in the project.
    """

    detected: bool = False
    face_count: int = 0
    bbox: Optional[BBox] = None
    detection_confidence: Optional[float] = None
    quality: FaceQuality = FaceQuality.UNAVAILABLE
    quality_score: Optional[float] = None
    # Raw quality measurements (blur variance, min side px, det score, ...).
    quality_signals: Dict[str, Any] = Field(default_factory=dict)


class BiometricResults(BaseModel):
    """Phase 4 output: face-verification evidence, attached as
    ``ScreeningResponse.biometrics``.

    When verification cannot be performed, ``status`` is one of NO_FACE /
    INSUFFICIENT_QUALITY / AMBIGUOUS / UNAVAILABLE and ``similarity`` is null —
    callers must treat those as "not compared", never as MISMATCH.
    """

    status: BiometricStatus
    # Cosine similarity of the two face embeddings (higher = more similar).
    similarity: Optional[float] = None
    # Prototype operating threshold used for the MATCH/MISMATCH decision.
    threshold: Optional[float] = None
    # Qualitative strength of the decision by margin to threshold: HIGH/MEDIUM/LOW.
    decision_strength: Optional[str] = None
    # Identifier of the face model that produced the embeddings.
    model: Optional[str] = None
    document_face: FaceEvidence = Field(default_factory=FaceEvidence)
    reference_face: FaceEvidence = Field(default_factory=FaceEvidence)
    message: str = ""
    # Populated when status == UNAVAILABLE.
    error: Optional[str] = None

    @classmethod
    def unavailable(cls, error: str, message: str = "") -> "BiometricResults":
        """Build an explicit biometric-outage result (never a MISMATCH)."""
        return cls(
            status=BiometricStatus.UNAVAILABLE,
            error=error,
            message=message or error,
        )
