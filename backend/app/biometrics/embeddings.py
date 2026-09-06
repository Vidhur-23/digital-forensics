"""Face embedding extraction for Phase 4 biometrics.

Isolated from verification logic: this module only turns a detected face into a
comparable vector. The embedding itself is produced by the InsightFace ArcFace
recognition model (``w600k_r50``) during detection — a 512-D vector that is
already L2-normalised, so cosine similarity reduces to a dot product.

No model is loaded here; embeddings come from the faces returned by
``face_detector.FaceModel``. Keeping this as its own module makes the
embedding contract explicit and swappable independently of the verifier.
"""
from __future__ import annotations

import numpy as np

from app.biometrics.face_detector import MODEL_NAME, RawFace

EMBEDDING_DIM = 512
EMBEDDING_MODEL = MODEL_NAME  # "insightface:buffalo_l" (ArcFace w600k_r50)


def get_embedding(face: RawFace) -> np.ndarray:
    """Return the L2-normalised embedding for a detected face.

    Re-normalises defensively so downstream cosine similarity is exact even if
    an upstream vector is not perfectly unit-length.
    """
    vec = np.asarray(face.embedding, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm
