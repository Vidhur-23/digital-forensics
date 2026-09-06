"""Application configuration (Phase 1).

Kept intentionally small — only what the document/OCR pipeline needs.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The forensic layer now lives inside the forensics package:
# <repo>/backend/app/config.py -> parent == <repo>/backend/app; /forensics.
_FORENSICS_DIR = Path(__file__).resolve().parent / "forensics"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    app_name: str = "SIH26188 Document Screening"
    api_prefix: str = "/api"

    # Phase 3: root of the forensic layer, now migrated into the forensics
    # package (``app/forensics``). Its orchestrator is
    # ``pipeline/forensics_engine.py`` and its calibration artifacts live in
    # ``results/``. Overridable via APP_FORENSIC_LAYER_PATH.
    forensic_layer_path: Path = _FORENSICS_DIR

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
