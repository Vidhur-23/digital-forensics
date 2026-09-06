"""Face verification for Phase 4 biometrics — comparison logic only.

Compares two face embeddings with cosine similarity (the correct convention for
the InsightFace ArcFace model: embeddings are L2-normalised, higher similarity =
more alike) and turns the score into a MATCH / MISMATCH decision plus a coarse
strength.

Threshold rationale
-------------------
``MATCH_THRESHOLD`` is a **prototype operating threshold, not a calibrated
biometric truth**. It was chosen from the project's sample data, where genuine
same-identity pairs scored ~0.98 cosine and different-identity passport pairs
scored ~0.13-0.36. 0.50 separates those populations with margin. It has NOT
been tuned on a labelled verification benchmark (e.g. an ROC/FMR-FNMR sweep),
and the available same-identity samples are near-duplicate captures, so the
threshold may be conservative for genuine cross-session variation. Treat it as a
demo operating point and recalibrate before any real use.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from app.api.schemas.biometric import BiometricStatus

# Prototype cosine-similarity operating threshold (see module docstring).
MATCH_THRESHOLD = 0.50

# Margin (|similarity - threshold|) bands for a qualitative decision strength.
_STRENGTH_HIGH = 0.20
_STRENGTH_MEDIUM = 0.10


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (dot product of their unit forms)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def decision_strength(similarity: float, threshold: float = MATCH_THRESHOLD) -> str:
    """HIGH/MEDIUM/LOW by distance from the decision boundary."""
    margin = abs(similarity - threshold)
    if margin >= _STRENGTH_HIGH:
        return "HIGH"
    if margin >= _STRENGTH_MEDIUM:
        return "MEDIUM"
    return "LOW"


def verify_embeddings(
    emb_a: np.ndarray, emb_b: np.ndarray, threshold: float = MATCH_THRESHOLD
) -> Tuple[BiometricStatus, float, str]:
    """Compare two embeddings.

    Returns ``(status, similarity, strength)`` where status is MATCH or MISMATCH.
    Quality/no-face/ambiguous states are decided upstream, not here.
    """
    similarity = cosine_similarity(emb_a, emb_b)
    status = BiometricStatus.MATCH if similarity >= threshold else BiometricStatus.MISMATCH
    return status, similarity, decision_strength(similarity, threshold)
