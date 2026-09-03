"""End-to-end API tests for POST /api/screen (with stubbed OCR)."""
from __future__ import annotations

from tests.conftest import make_image_bytes


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ocr_backend"] == "tesseract"


def test_screen_valid_image_returns_structured_result(client, image_bytes):
    r = client.post(
        "/api/screen",
        files={"document": ("passport.png", image_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Structure required by later phases.
    assert body["document_type"] == "passport"
    assert body["image"]["width"] == 1000 and body["image"]["height"] == 1000
    assert "fields" in body and body["fields"]
    assert body["mrz"]["detected"] is True
    assert body["mrz"]["fields"]["nationality"] == "UTO"
    assert "confidence" in body["ocr"] and body["ocr"]["word_count"] > 0

    # Fields carry bboxes (non-optional for Phase 2/3).
    for field in body["fields"].values():
        assert "value" in field and "confidence" in field
        assert field["bbox"] is not None and len(field["bbox"]) == 4


def test_screen_invalid_image_returns_clean_error(client):
    r = client.post(
        "/api/screen",
        files={"document": ("bad.png", b"not-an-image", "image/png")},
    )
    assert r.status_code == 422
    assert "detail" in r.json()


def test_screen_unsupported_content_type(client):
    r = client.post(
        "/api/screen",
        files={"document": ("doc.pdf", make_image_bytes(), "application/pdf")},
    )
    assert r.status_code == 415
