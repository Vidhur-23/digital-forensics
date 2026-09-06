"""Phase 3 integration tests: the forensic engine is wired into the existing
analysis pipeline and its evidence is exposed through POST /api/screen.

Covers:
  1. existing Phase 1 + Phase 2 behaviour is preserved (additive integration),
  2. forensics executes as part of a normal analysis,
  3. forensics appears in the API response,
  4. a flagged finding is well-structured (type/score/severity/region/message),
  5. a genuine image yields a clean COMPLETED result (not required to be zero),
  6. a forensic failure is isolated (Phase 1/2 still returned; status UNAVAILABLE),
  7. returned bounding boxes are valid within the image and use [x1,y1,x2,y2].

Fast, deterministic cases use a stub forensic service. Cases that must exercise
the *real* forensic engine use sample images from ``forensic_layer/dataset/``
and are skipped if that dataset is not present.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.api.schemas.evidence import (
    ForensicFinding,
    ForensicResults,
    ForensicStatus,
    ForensicSummary,
)
from app.config import settings
from app.forensics.engine import ForensicLayerService, ForensicService
from app.pipeline.pipeline import ScreeningPipeline
from tests.conftest import StubOCREngine, make_image_bytes

# ---------------------------------------------------------------------------
# Stub forensic services (deterministic, no CV work)
# ---------------------------------------------------------------------------


class _StubForensicService(ForensicService):
    """Records invocation and returns a canned flagged finding with a region."""

    def __init__(self) -> None:
        self.called_with_shape = None

    def analyze(self, image_bgr, document_id=None) -> ForensicResults:
        self.called_with_shape = image_bgr.shape
        finding = ForensicFinding(
            layer_name="Layer 2: Photo Boundary Analysis",
            finding_type="photo_boundary_anomaly",
            passed=False,
            score=0.78,
            severity="medium",
            confidence=0.6,
            region=[100, 200, 300, 240],
            signals=["photo_boundary_anomaly"],
            raw_metric={"long_straight_line_count": 5, "threshold": 1.0},
            message="Possible pasted or spliced photo region.",
            status="ok",
        )
        return ForensicResults(
            status=ForensicStatus.COMPLETED,
            engine_version="test",
            findings=[finding],
            summary=ForensicSummary(
                layers_requested=1,
                layers_completed=1,
                layers_failed=0,
                flagged_count=1,
                all_passed=False,
            ),
            elapsed_seconds=0.01,
        )


class _RaisingForensicService(ForensicService):
    """Simulates a forensic detector that raises during processing."""

    def analyze(self, image_bgr, document_id=None) -> ForensicResults:
        raise RuntimeError("simulated detector crash")


# ---------------------------------------------------------------------------
# Dataset helpers for real-engine cases
# ---------------------------------------------------------------------------


def _sample_image_bytes(kind: str) -> bytes:
    directory = settings.forensic_layer_path / "dataset" / kind
    if not directory.exists():
        pytest.skip(f"forensic dataset '{kind}' not available at {directory}")
    images = sorted(directory.glob("*.jpg"))
    if not images:
        pytest.skip(f"no sample images in {directory}")
    return images[0].read_bytes()


# ---------------------------------------------------------------------------
# Test 1 — existing pipeline still works (integration is additive)
# ---------------------------------------------------------------------------


def test_phase1_and_phase2_preserved_alongside_forensics():
    pipeline = ScreeningPipeline(StubOCREngine(), forensic_service=_StubForensicService())
    resp = pipeline.screen(make_image_bytes())

    # Phase 1 untouched.
    assert resp.document_type == "passport"
    assert resp.fields
    assert resp.mrz.detected is True
    # Phase 2 untouched.
    assert resp.rules is not None and resp.rules.findings
    # Phase 3 added.
    assert resp.forensics is not None


# ---------------------------------------------------------------------------
# Test 2 — forensics executes as part of a normal analysis
# ---------------------------------------------------------------------------


def test_forensics_executes_in_pipeline():
    stub = _StubForensicService()
    pipeline = ScreeningPipeline(StubOCREngine(), forensic_service=stub)
    resp = pipeline.screen(make_image_bytes())

    # The forensic service was invoked with the original BGR image (3 channels).
    assert stub.called_with_shape is not None
    assert len(stub.called_with_shape) == 3 and stub.called_with_shape[2] == 3
    assert resp.forensics.status == ForensicStatus.COMPLETED


# ---------------------------------------------------------------------------
# Test 3 — forensics appears in the API response
# ---------------------------------------------------------------------------


def test_api_response_contains_forensics_section(image_bytes):
    from fastapi.testclient import TestClient

    from app.dependencies import get_pipeline
    from app.main import app

    app.dependency_overrides[get_pipeline] = lambda: ScreeningPipeline(
        StubOCREngine(), forensic_service=_StubForensicService()
    )
    try:
        with TestClient(app) as c:
            r = c.post(
                "/api/screen",
                files={"document": ("passport.png", image_bytes, "image/png")},
            )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    body = r.json()
    assert "forensics" in body and body["forensics"] is not None
    assert body["forensics"]["status"] == "COMPLETED"
    assert "summary" in body["forensics"]
    # Phase 1 + Phase 2 still present in the same response.
    assert body["document_type"] == "passport"
    assert body["rules"] is not None


# ---------------------------------------------------------------------------
# Test 4 — finding structure (real engine on a fake sample -> anomalies)
# ---------------------------------------------------------------------------


def test_flagged_finding_structure_from_real_engine():
    data = _sample_image_bytes("fakes")
    pipeline = ScreeningPipeline(StubOCREngine())  # default real forensic service
    resp = pipeline.screen(data)

    assert resp.forensics.status == ForensicStatus.COMPLETED
    assert resp.forensics.findings
    for f in resp.forensics.findings:
        assert f.layer_name
        assert f.finding_type
        assert isinstance(f.score, float)
        assert f.severity in {"low", "medium", "high"}
        assert isinstance(f.confidence, float)
        assert f.message  # explanation exists
        assert f.signals  # supporting signal tag(s)
    # A fake passport should flag at least one layer.
    assert resp.forensics.summary.flagged_count >= 1


# ---------------------------------------------------------------------------
# Test 5 — clean / no-finding state on a genuine sample (not required == 0)
# ---------------------------------------------------------------------------


def test_genuine_sample_produces_clean_completed_result():
    data = _sample_image_bytes("reals")
    pipeline = ScreeningPipeline(StubOCREngine())
    resp = pipeline.screen(data)

    assert resp.forensics.status == ForensicStatus.COMPLETED
    # Findings are produced; a clean genuine sample should pass every layer.
    assert resp.forensics.findings
    assert resp.forensics.summary.layers_failed == 0
    assert resp.forensics.summary.all_passed is True
    assert resp.forensics.summary.flagged_count == 0


# ---------------------------------------------------------------------------
# Test 6 — forensic failure isolation
# ---------------------------------------------------------------------------


def test_forensic_failure_is_isolated_from_analysis():
    pipeline = ScreeningPipeline(StubOCREngine(), forensic_service=_RaisingForensicService())
    resp = pipeline.screen(make_image_bytes())

    # Phase 1 + Phase 2 still returned.
    assert resp.fields
    assert resp.rules is not None and resp.rules.findings
    # Forensics marked UNAVAILABLE (an outage), NOT a clean/passing result.
    assert resp.forensics is not None
    assert resp.forensics.status == ForensicStatus.UNAVAILABLE
    assert resp.forensics.error
    assert resp.forensics.summary.all_passed is False


def test_adapter_reports_unavailable_when_layer_missing(tmp_path):
    service = ForensicLayerService(forensic_layer_path=tmp_path / "does_not_exist")
    result = service.analyze(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.status == ForensicStatus.UNAVAILABLE
    assert result.error


# ---------------------------------------------------------------------------
# Test 7 — bounding-box validity (real engine, [x1,y1,x2,y2] within image)
# ---------------------------------------------------------------------------


def test_forensic_bboxes_are_valid_within_image():
    data = _sample_image_bytes("reals")
    pipeline = ScreeningPipeline(StubOCREngine())
    resp = pipeline.screen(data)

    width, height = resp.image.width, resp.image.height
    regions = [f.region for f in resp.forensics.findings if f.region is not None]
    # The genuine sample localises a portrait region (Layer 2).
    assert regions, "expected at least one localised region on the sample image"
    for x1, y1, x2, y2 in regions:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
