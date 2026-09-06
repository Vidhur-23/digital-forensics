"""Phase 4 biometric face-verification tests.

Detection/embedding cases exercise the real InsightFace model on the project's
sample passports; the whole module skips if the model cannot be loaded (e.g.
weights not downloaded / no network on first run). Pipeline-level isolation
cases use lightweight stubs and always run.

Empirically measured on this dataset: same-identity pairs score ~0.98 cosine,
different-identity passport pairs ~0.13-0.36; the prototype threshold is 0.50.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.api.schemas.biometric import BiometricResults, BiometricStatus, FaceQuality
from app.biometrics.face_detector import FaceModel, FaceModelUnavailable
from app.biometrics.service import BiometricService, InsightFaceBiometricService
from app.forensics.engine import FORENSICS_DIR
from app.pipeline.pipeline import ScreeningPipeline
from tests.conftest import StubOCREngine, make_image_bytes

_DATASET = FORENSICS_DIR / "dataset"
_DOC = _DATASET / "reals" / "aze_passport_00.jpg"
_SAME_PERSON = _DATASET / "fakes" / "aze_passport_00_fake_3_104.jpg"  # cos~0.98
_DIFF_PERSON = _DATASET / "reals" / "aze_passport_02.jpg"            # cos~0.17


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bio_service() -> InsightFaceBiometricService:
    """Real service with the InsightFace model loaded once for the module."""
    if not _DOC.exists():
        pytest.skip(f"sample passport dataset not available at {_DATASET}")
    model = FaceModel()
    try:
        model.detect(cv2.imread(str(_DOC)))  # force load
    except FaceModelUnavailable as exc:
        pytest.skip(f"face model unavailable: {exc}")
    return InsightFaceBiometricService(model=model)


def _img(path) -> np.ndarray:
    return cv2.imread(str(path))


# ---------------------------------------------------------------------------
# Test 1 — passport face detected
# ---------------------------------------------------------------------------
def test_passport_face_detected(bio_service):
    res = bio_service.verify(_img(_DOC), _img(_SAME_PERSON))
    assert res.document_face.detected is True
    assert res.document_face.face_count == 1
    assert res.document_face.bbox is not None and len(res.document_face.bbox) == 4
    assert res.document_face.detection_confidence > 0


# ---------------------------------------------------------------------------
# Test 2 — reference face detected
# ---------------------------------------------------------------------------
def test_reference_face_detected(bio_service):
    res = bio_service.verify(_img(_DOC), _img(_SAME_PERSON))
    assert res.reference_face.detected is True
    assert res.reference_face.bbox is not None
    assert res.reference_face.quality != FaceQuality.UNAVAILABLE


# ---------------------------------------------------------------------------
# Test 3 — same-person comparison -> MATCH
# ---------------------------------------------------------------------------
def test_same_person_match(bio_service):
    res = bio_service.verify(_img(_DOC), _img(_SAME_PERSON))
    assert res.status == BiometricStatus.MATCH
    assert res.similarity is not None and res.similarity >= res.threshold
    assert res.decision_strength in {"HIGH", "MEDIUM", "LOW"}


# ---------------------------------------------------------------------------
# Test 4 — different-person comparison -> MISMATCH
# ---------------------------------------------------------------------------
def test_different_person_mismatch(bio_service):
    res = bio_service.verify(_img(_DOC), _img(_DIFF_PERSON))
    assert res.status == BiometricStatus.MISMATCH
    assert res.similarity is not None and res.similarity < res.threshold


# ---------------------------------------------------------------------------
# Test 5 — poor quality reference must NOT become a mismatch
# ---------------------------------------------------------------------------
def test_poor_quality_reference_not_mismatch(bio_service):
    doc = _img(_DOC)
    poor_ref = cv2.GaussianBlur(doc, (0, 0), sigmaX=25)  # heavy blur, face intact-ish
    res = bio_service.verify(doc, poor_ref)
    assert res.status in {
        BiometricStatus.INSUFFICIENT_QUALITY,
        BiometricStatus.NO_FACE,
    }
    assert res.status != BiometricStatus.MISMATCH


# ---------------------------------------------------------------------------
# Test 6 — no face in reference
# ---------------------------------------------------------------------------
def test_no_face_reference(bio_service):
    blank = np.full((600, 800, 3), 255, dtype=np.uint8)
    res = bio_service.verify(_img(_DOC), blank)
    assert res.status == BiometricStatus.NO_FACE
    assert res.document_face.detected is True
    assert res.reference_face.detected is False
    assert res.similarity is None  # never compared


# ---------------------------------------------------------------------------
# Test 7 — multiple faces -> ambiguous
# ---------------------------------------------------------------------------
def test_multiple_faces_ambiguous(bio_service):
    doc = _img(_DOC)
    from app.biometrics.face_detector import select_primary_face

    faces = bio_service._model.detect(doc)
    face = select_primary_face(faces)[0]
    x1, y1, x2, y2 = face.bbox
    crop = doc[y1:y2, x1:x2]
    two_faces = np.hstack([crop, crop.copy()])  # two comparably-sized faces
    res = bio_service.verify(_img(_DOC), two_faces)
    assert res.status == BiometricStatus.AMBIGUOUS
    assert res.reference_face.face_count >= 2
    assert res.similarity is None


# ---------------------------------------------------------------------------
# No-reference -> UNAVAILABLE (not a mismatch)
# ---------------------------------------------------------------------------
def test_no_reference_is_unavailable(bio_service):
    res = bio_service.verify(_img(_DOC), None)
    assert res.status == BiometricStatus.UNAVAILABLE
    assert res.error


# ---------------------------------------------------------------------------
# Test 8 — pipeline integration: all phases coexist
# ---------------------------------------------------------------------------
def test_pipeline_all_phases_coexist(bio_service):
    pipeline = ScreeningPipeline(StubOCREngine(), biometric_service=bio_service)
    resp = pipeline.screen(_DOC.read_bytes(), reference_data=_SAME_PERSON.read_bytes())

    # Phases 1-3 present.
    assert resp.document_type == "passport"
    assert resp.rules is not None and resp.rules.findings
    assert resp.forensics is not None
    # Phase 4 present and computed.
    assert resp.biometrics is not None
    assert resp.biometrics.status == BiometricStatus.MATCH
    assert resp.biometrics.document_face.bbox is not None


# ---------------------------------------------------------------------------
# Isolation — a biometric crash must not destroy Phases 1-3
# ---------------------------------------------------------------------------
class _RaisingBiometricService(BiometricService):
    def verify(self, document_image_bgr, reference_image_bgr) -> BiometricResults:
        raise RuntimeError("simulated biometric crash")


def test_biometric_failure_isolated():
    pipeline = ScreeningPipeline(
        StubOCREngine(), biometric_service=_RaisingBiometricService()
    )
    resp = pipeline.screen(make_image_bytes(), reference_data=make_image_bytes())
    assert resp.rules is not None and resp.rules.findings
    assert resp.forensics is not None
    assert resp.biometrics.status == BiometricStatus.UNAVAILABLE
    assert resp.biometrics.status != BiometricStatus.MISMATCH


# ---------------------------------------------------------------------------
# API — biometrics appears in the analysis response (no second endpoint)
# ---------------------------------------------------------------------------
def test_api_screen_includes_biometrics(bio_service):
    from fastapi.testclient import TestClient

    from app.dependencies import get_pipeline
    from app.main import app

    app.dependency_overrides[get_pipeline] = lambda: ScreeningPipeline(
        StubOCREngine(), biometric_service=bio_service
    )
    try:
        with TestClient(app) as c:
            r = c.post(
                "/api/screen",
                files={
                    "document": ("doc.jpg", _DOC.read_bytes(), "image/jpeg"),
                    "reference_face": ("ref.jpg", _SAME_PERSON.read_bytes(), "image/jpeg"),
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["biometrics"]["status"] == "MATCH"
    assert body["biometrics"]["document_face"]["bbox"] is not None
    # Phases 1-3 still present in the same response object.
    assert body["rules"] is not None and body["forensics"] is not None
