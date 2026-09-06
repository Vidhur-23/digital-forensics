"""Phase 4 biometric orchestration service.

Composes the biometric components — detection (``face_detector``), quality
(``quality``), embeddings (``embeddings``) and comparison (``verifier``) — into
a single evidence result. Mirrors the Phase 3 ``ForensicService`` contract:

* injectable interface so the pipeline and tests can swap implementations,
* never raises — any failure becomes an explicit ``UNAVAILABLE`` /
  ``NO_FACE`` / ``INSUFFICIENT_QUALITY`` / ``AMBIGUOUS`` result so the rest of
  the analysis (Phases 1-3) is always returned,
* evidence only — it produces correspondence evidence, never a fraud verdict.

Inputs are BGR ``np.ndarray`` images (OpenCV convention), the same pixels used
by OCR and forensics — no extra colour conversions are introduced.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from app.api.schemas.biometric import (
    BiometricResults,
    BiometricStatus,
    FaceEvidence,
    FaceQuality,
)
from app.biometrics import verifier
from app.biometrics.embeddings import EMBEDDING_MODEL, get_embedding
from app.biometrics.face_detector import (
    FaceModel,
    FaceModelUnavailable,
    RawFace,
    select_primary_face,
)
from app.biometrics.quality import assess_face_quality

_NO_REFERENCE_MSG = (
    "No reference/live face image provided; biometric verification requires a "
    "separate reference face image."
)


class BiometricService(ABC):
    """Interface the pipeline depends on (injectable for tests)."""

    @abstractmethod
    def verify(
        self,
        document_image_bgr: np.ndarray,
        reference_image_bgr: Optional[np.ndarray],
    ) -> BiometricResults:
        """Compare the document face against a reference face image.

        Must not raise: failures are reported via the returned result's status.
        """
        raise NotImplementedError


class InsightFaceBiometricService(BiometricService):
    """Biometric service backed by the InsightFace ``buffalo_l`` model."""

    def __init__(self, model: Optional[FaceModel] = None):
        # One model instance, loaded lazily on first real use and reused.
        self._model = model or FaceModel()

    def verify(
        self,
        document_image_bgr: np.ndarray,
        reference_image_bgr: Optional[np.ndarray],
    ) -> BiometricResults:
        # A reference image is mandatory — without it there is nothing to
        # compare against (never treat this as a mismatch).
        if reference_image_bgr is None or getattr(reference_image_bgr, "size", 0) == 0:
            return BiometricResults(
                status=BiometricStatus.UNAVAILABLE,
                model=EMBEDDING_MODEL,
                message=_NO_REFERENCE_MSG,
                error=_NO_REFERENCE_MSG,
            )

        # Detect on both images (model load failure => whole layer unavailable).
        try:
            doc_faces = self._model.detect(document_image_bgr)
            ref_faces = self._model.detect(reference_image_bgr)
        except FaceModelUnavailable as exc:
            return BiometricResults.unavailable(
                f"Face model unavailable: {exc}",
                message="Biometric face model could not be loaded.",
            )
        except Exception as exc:  # defensive: any inference error
            return BiometricResults.unavailable(
                f"Face detection failed: {exc!r}",
                message="Biometric analysis could not be completed.",
            )

        doc_primary, doc_ambiguous = select_primary_face(doc_faces)
        ref_primary, ref_ambiguous = select_primary_face(ref_faces)

        doc_ev = self._face_evidence(document_image_bgr, doc_faces, doc_primary)
        ref_ev = self._face_evidence(reference_image_bgr, ref_faces, ref_primary)

        base = dict(model=EMBEDDING_MODEL, document_face=doc_ev, reference_face=ref_ev)

        # Ambiguity first: >1 comparable face means the intended face is unclear.
        if doc_ambiguous or ref_ambiguous:
            which = " and ".join(
                w for w, a in (("document", doc_ambiguous), ("reference", ref_ambiguous)) if a
            )
            return BiometricResults(
                status=BiometricStatus.AMBIGUOUS,
                message=f"Multiple comparably-sized faces in the {which} image; "
                "intended face cannot be identified unambiguously.",
                **base,
            )

        # No usable face in one/both images.
        if doc_primary is None or ref_primary is None:
            missing = " and ".join(
                w for w, p in (("document", doc_primary), ("reference", ref_primary)) if p is None
            )
            return BiometricResults(
                status=BiometricStatus.NO_FACE,
                message=f"No face detected in the {missing} image.",
                **base,
            )

        # Quality gate — poor input must not become a mismatch.
        poor = [
            w
            for w, ev in (("document", doc_ev), ("reference", ref_ev))
            if ev.quality in (FaceQuality.POOR, FaceQuality.UNAVAILABLE)
        ]
        if poor:
            return BiometricResults(
                status=BiometricStatus.INSUFFICIENT_QUALITY,
                message=f"Face quality too low for reliable comparison ({', '.join(poor)}).",
                **base,
            )

        # Compare embeddings.
        status, similarity, strength = verifier.verify_embeddings(
            get_embedding(doc_primary), get_embedding(ref_primary)
        )
        verb = "correspond" if status == BiometricStatus.MATCH else "do not correspond"
        return BiometricResults(
            status=status,
            similarity=round(similarity, 4),
            threshold=verifier.MATCH_THRESHOLD,
            decision_strength=strength,
            message=(
                f"Document and reference faces {verb} "
                f"(cosine similarity {similarity:.3f} vs threshold "
                f"{verifier.MATCH_THRESHOLD:.2f})."
            ),
            **base,
        )

    @staticmethod
    def _face_evidence(
        image_bgr: np.ndarray,
        faces: List[RawFace],
        primary: Optional[RawFace],
    ) -> FaceEvidence:
        if primary is not None:
            quality, q_score, signals = assess_face_quality(image_bgr, primary)
            return FaceEvidence(
                detected=True,
                face_count=len(faces),
                bbox=list(primary.bbox),
                detection_confidence=round(float(primary.det_score), 4),
                quality=quality,
                quality_score=q_score,
                quality_signals=signals,
            )
        # No single intended face (none detected, or ambiguous set).
        return FaceEvidence(
            detected=bool(faces),
            face_count=len(faces),
            quality=FaceQuality.UNAVAILABLE,
        )
