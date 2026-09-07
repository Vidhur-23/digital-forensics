"""Persistence tests: /api/screen stores an analysis; the read endpoints
return it. Uses a throwaway SQLite DB and the project's stub OCR engine, so no
Tesseract/network is required. Exercises the REAL routes and repository.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import connection
from app.database.connection import get_session
from app.dependencies import get_pipeline
from app.main import app
from app.pipeline.pipeline import ScreeningPipeline
from tests.conftest import StubOCREngine, make_image_bytes


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolated temp DB + evidence dir; point the app's engine at them so the
    # startup table-create and the route both use this database.
    url = f"sqlite:///{tmp_path / 'test.db'}"
    eng = create_engine(url, connect_args={"check_same_thread": False})
    test_session = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    monkeypatch.setattr(connection, "engine", eng)
    monkeypatch.setattr(connection, "SessionLocal", test_session)
    monkeypatch.setattr(settings, "evidence_dir", str(tmp_path / "ev"))
    # Re-enable persistence for these tests (the autouse fixture turns it off).
    monkeypatch.setattr(settings, "persist_analyses", True)
    connection.Base.metadata.create_all(bind=eng)

    def _get_session():
        db = test_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = _get_session
    app.dependency_overrides[get_pipeline] = lambda: ScreeningPipeline(StubOCREngine())
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _screen(client, with_reference=False):
    files = {"document": ("doc.png", make_image_bytes(), "image/png")}
    if with_reference:
        files["reference_face"] = ("ref.png", make_image_bytes(), "image/png")
    r = client.post("/api/screen", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def test_screen_persists_and_returns_analysis_id(client):
    body = _screen(client)
    assert body["analysis_id"], "screen response should carry the saved record id"

    # It appears in the list endpoint.
    lst = client.get("/api/analyses").json()
    assert len(lst) == 1
    assert lst[0]["id"] == body["analysis_id"]


def test_list_summary_columns_match_response(client):
    body = _screen(client)
    row = client.get("/api/analyses").json()[0]
    intel = body["intelligence"]
    # Extracted columns are denormalised copies of the actual response values.
    assert row["risk_level"] == intel["risk"]["level"]
    assert row["risk_score"] == intel["risk"]["score"]
    assert row["recommendation"] == intel["recommendation"]["action"]
    assert row["biometric_status"] == body["biometrics"]["status"]
    assert row["forensic_status"] == body["forensics"]["status"]
    assert row["llm_status"] == intel["llm"]["status"]


def test_get_one_returns_full_result_and_evidence(client):
    body = _screen(client, with_reference=True)
    rec = client.get(f"/api/analyses/{body['analysis_id']}").json()

    # Lossless: the stored full_result reproduces the original response.
    assert rec["full_result"]["document_type"] == body["document_type"]
    assert rec["full_result"]["intelligence"]["risk"]["score"] == (
        body["intelligence"]["risk"]["score"]
    )
    # Both uploaded images are recorded (document + reference_face) as pointers.
    kinds = {f["kind"] for f in rec["evidence_files"]}
    assert kinds == {"document", "reference_face"}
    assert rec["has_reference_face"] is True
    for f in rec["evidence_files"]:
        assert len(f["sha256"]) == 64
        assert "path" not in f  # internal path is never exposed to clients


def test_evidence_written_to_disk(client, tmp_path):
    _screen(client)
    stored = list((tmp_path / "ev").glob("*"))
    assert stored, "the uploaded image should be written to the evidence dir"
    # Content-addressed filename = sha256 + extension.
    assert stored[0].name.endswith(".png")


def test_get_missing_returns_404(client):
    r = client.get("/api/analyses/deadbeef")
    assert r.status_code == 404


def test_persistence_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "persist_analyses", False)
    body = _screen(client)
    assert body["analysis_id"] is None
    assert client.get("/api/analyses").json() == []


def test_persistence_failure_does_not_break_screening(client, monkeypatch):
    # Simulate a DB write failure; the analysis must still be returned (200).
    from app.api.routes import documents

    def _boom(*a, **k):
        raise RuntimeError("simulated DB outage")

    monkeypatch.setattr(documents, "persist_screening", _boom)
    body = _screen(client)
    assert body["analysis_id"] is None  # not saved
    assert body["intelligence"]["risk"]["level"]  # but the analysis is intact
