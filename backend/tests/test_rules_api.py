"""Phase 2 integration: rule findings are exposed through POST /api/screen and
the full Phase 1 -> Phase 2 pipeline runs end to end (with stubbed OCR)."""
from __future__ import annotations

from app.pipeline.pipeline import ScreeningPipeline
from tests.conftest import StubOCREngine, make_image_bytes


def test_screen_response_includes_rule_findings(client, image_bytes):
    r = client.post(
        "/api/screen",
        files={"document": ("passport.png", image_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Phase 1 structure preserved.
    assert body["document_type"] == "passport"
    assert body["mrz"]["detected"] is True

    # Phase 2 findings present and well-formed.
    assert body["rules"] is not None
    findings = body["rules"]["findings"]
    assert findings
    for f in findings:
        assert f["rule_id"] and f["message"]
        assert f["status"] in {"PASS", "WARNING", "FAIL", "NOT_APPLICABLE"}
        assert f["severity"] in {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}

    # Deterministic MRZ check digits on the ICAO specimen all pass.
    check_digits = [f for f in findings if f["rule_id"].startswith("MRZ_CHECK_")]
    assert check_digits and all(f["status"] == "PASS" for f in check_digits)

    # Summary is a tally, not a verdict.
    assert body["rules"]["summary"]["total"] == len(findings)


def test_pipeline_attaches_rules_directly():
    pipeline = ScreeningPipeline(StubOCREngine())
    result = pipeline.screen(make_image_bytes())
    assert result.rules is not None
    assert result.rules.findings
    # The engine ran after Phase 1: fields + MRZ still populated.
    assert result.fields
    assert result.mrz.detected is True
