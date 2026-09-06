"""Face quality assessment for Phase 4 biometrics.

Prevents unreliable comparisons: a face that is too small, too blurry, or
weakly detected is flagged so the verifier can return INSUFFICIENT_QUALITY
rather than a misleading MISMATCH.

The thresholds below are **prototype operating values** chosen from the
project's sample passports (a genuine portrait scores det≈0.82, min-side≈300px,
Laplacian-variance≈500). They are not calibrated against a labelled quality
benchmark and should be revisited with real data.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from app.api.schemas.biometric import FaceQuality
from app.biometrics.face_detector import RawFace

# Prototype thresholds: (good_min, acceptable_min). Below acceptable_min => POOR.
_DET_SCORE = (0.65, 0.50)
_MIN_SIDE_PX = (80, 45)
_BLUR_VAR = (120.0, 40.0)

# Ordered worst -> best so the overall grade can be the minimum tier.
_ORDER = [FaceQuality.POOR, FaceQuality.ACCEPTABLE, FaceQuality.GOOD]


def _grade(value: float, thresholds: Tuple[float, float]) -> FaceQuality:
    good_min, acc_min = thresholds
    if value >= good_min:
        return FaceQuality.GOOD
    if value >= acc_min:
        return FaceQuality.ACCEPTABLE
    return FaceQuality.POOR


def _blur_variance(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_face_quality(
    image_bgr: np.ndarray, face: RawFace
) -> Tuple[FaceQuality, float, Dict[str, float]]:
    """Grade a detected face for comparison fitness.

    Returns ``(quality, quality_score, signals)`` where ``quality_score`` is a
    coarse 0..1 confidence-in-usability and ``signals`` holds the raw metrics.
    The overall grade is the **worst** of the individual metric grades so a
    single disqualifying factor is not hidden by good ones.
    """
    x1, y1, x2, y2 = face.bbox
    h, w = image_bgr.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    crop = image_bgr[y1c:y2c, x1c:x2c]

    face_w, face_h = x2 - x1, y2 - y1
    min_side = int(min(max(0, face_w), max(0, face_h)))
    blur = _blur_variance(crop)

    grades = [
        _grade(face.det_score, _DET_SCORE),
        _grade(float(min_side), _MIN_SIDE_PX),
        _grade(blur, _BLUR_VAR),
    ]
    overall = min(grades, key=_ORDER.index)

    signals = {
        "detection_confidence": round(float(face.det_score), 4),
        "min_side_px": min_side,
        "blur_variance": round(blur, 2),
    }
    quality_score = round(_ORDER.index(overall) / (len(_ORDER) - 1), 4)
    return overall, quality_score, signals
