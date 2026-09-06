"""Face detection for Phase 4 biometrics.

Wraps the InsightFace ``buffalo_l`` model pack, which performs both face
*detection* (SCRFD) and *recognition* (ArcFace ``w600k_r50``) in a single
forward pass — each detected face already carries its 512-D normalised
embedding. This module is responsible only for detection and for selecting the
intended face; embedding extraction and comparison live in their own modules
(``embeddings.py``, ``verifier.py``) to keep the stages isolated.

The model is loaded lazily and once (see :class:`FaceModel`); the application
wires a single instance so a heavy model is never reloaded per request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

MODEL_NAME = "insightface:buffalo_l"

# If the largest detected face is at least this many times the area of the
# second-largest, it is treated as the dominant/intended portrait (e.g. a
# passport main photo vs. a small secondary "ghost" image). Otherwise multiple
# comparably-sized faces are reported as ambiguous rather than picking one.
DOMINANCE_RATIO = 2.0


class FaceModelUnavailable(RuntimeError):
    """Raised when the face model cannot be loaded (missing weights, etc.)."""


@dataclass
class RawFace:
    """One detected face in original-image pixel coordinates."""

    bbox: List[int]  # [x1, y1, x2, y2]
    det_score: float
    embedding: np.ndarray  # L2-normalised 512-D ArcFace embedding
    area: float


class FaceModel:
    """Lazy singleton wrapper around InsightFace ``FaceAnalysis`` (CPU)."""

    def __init__(self, name: str = "buffalo_l", det_size: Tuple[int, int] = (640, 640)):
        self._name = name
        self._det_size = det_size
        self._app = None
        self._load_error: Optional[str] = None

    def _ensure_loaded(self):
        if self._app is not None or self._load_error is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=self._name, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=self._det_size)
            self._app = app
        except Exception as exc:  # missing weights / onnxruntime / etc.
            self._load_error = f"{type(exc).__name__}: {exc}"
        return self._app

    def detect(self, image_bgr: np.ndarray) -> List[RawFace]:
        """Detect all faces in a BGR image. Raises FaceModelUnavailable if the
        model cannot be loaded."""
        app = self._ensure_loaded()
        if app is None:
            raise FaceModelUnavailable(self._load_error or "Face model unavailable.")
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return []

        faces = app.get(image_bgr)
        raw: List[RawFace] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            raw.append(
                RawFace(
                    bbox=[x1, y1, x2, y2],
                    det_score=float(f.det_score),
                    embedding=np.asarray(f.normed_embedding, dtype=np.float64),
                    area=float(max(0, x2 - x1) * max(0, y2 - y1)),
                )
            )
        return raw


def select_primary_face(faces: List[RawFace]) -> Tuple[Optional[RawFace], bool]:
    """Pick the intended face from a detection list.

    Returns ``(face, ambiguous)``:
    * ``(None, False)`` — no face detected.
    * ``(face, False)`` — one face, or a clearly dominant face among several.
    * ``(None, True)`` — several comparably-sized faces; intended face cannot be
      safely identified, so the caller should report AMBIGUOUS (never guess).
    """
    if not faces:
        return None, False
    if len(faces) == 1:
        return faces[0], False

    ordered = sorted(faces, key=lambda f: f.area, reverse=True)
    largest, second = ordered[0], ordered[1]
    if second.area <= 0 or largest.area >= DOMINANCE_RATIO * second.area:
        return largest, False
    return None, True
