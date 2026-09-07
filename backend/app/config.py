"""Application configuration.

Every setting's VALUE lives in the ``.env`` file (``backend/.env``), not here —
this module only declares the fields and their types and loads them from that
file. There are intentionally no inline defaults: the ``.env`` file is the single
source of configuration values (see ``.env.example`` for the template).
"""
from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env — resolved from this module's location so it is found regardless
# of the current working directory the app/tests are launched from.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str
    api_prefix: str
    # Accepted upload content types for the screening endpoint (JSON list in .env).
    allowed_image_types: tuple[str, ...]
    # Reject absurdly large uploads early (bytes).
    max_upload_bytes: int
    # Optional explicit path to the tesseract binary (else taken from PATH).
    tesseract_cmd: str | None

    # --- Phase 5: Intelligence Layer LLM ---
    llm_enabled: bool
    llm_provider: str
    llm_model: str
    # No API key is ever hard-coded; it comes from APP_LLM_API_KEY (or the SDK's
    # own env var when this is empty).
    llm_api_key: str | None
    llm_base_url: str | None
    llm_timeout_seconds: float
    llm_max_output_tokens: int

    # --- Persistence (analysis records) ---
    database_url: str
    evidence_dir: str
    persist_analyses: bool
    db_auto_create: bool

    @field_validator("tesseract_cmd", "llm_api_key", "llm_base_url", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        """Treat an empty .env value (e.g. ``APP_LLM_API_KEY=``) as unset."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


settings = Settings()
