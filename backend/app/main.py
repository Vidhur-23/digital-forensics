"""FastAPI application entry point.

Phase 1: document capture, OCR, passport field extraction and MRZ detection.
Phase 2: deterministic Rules Engine (findings are attached to the screening
response by the pipeline — see ``app.pipeline.pipeline``).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="SIH26188 — document capture, OCR, passport field extraction, "
    "MRZ detection (Phase 1) and deterministic rule findings (Phase 2).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
