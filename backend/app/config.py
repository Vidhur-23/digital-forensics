"""Application configuration (Phase 1).

Kept intentionally small — only what the document/OCR pipeline needs.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    app_name: str = "SIH26188 Document Screening"
    api_prefix: str = "/api"

    # Accepted upload content types for the screening endpoint.
    allowed_image_types: tuple[str, ...] = (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/bmp",
        "image/tiff",
        "image/webp",
    )
    # Reject absurdly large uploads early (bytes). 20 MB default.
    max_upload_bytes: int = 20 * 1024 * 1024

    # Optional explicit path to the tesseract binary (else taken from PATH).
    tesseract_cmd: str | None = None


settings = Settings()
