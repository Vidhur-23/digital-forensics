"""FastAPI application entry point (Phase 1: Document + OCR pipeline)."""
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import documents, health
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="SIH26188 — Phase 1: document capture, OCR, passport field "
    "extraction and MRZ detection.",
)

# Health at root; screening under the API prefix -> POST /api/screen.
app.include_router(health.router)
app.include_router(documents.router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict:
    return {
        "app": settings.app_name,
        "docs": "/docs",
        "screen_endpoint": f"{settings.api_prefix}/screen",
    }
