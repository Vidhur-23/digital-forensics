"""Application configuration (Phase 1).

Kept intentionally small — only what the document/OCR pipeline needs.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    app_name: str = "SIH26188 Document Screening"
    api_prefix: str = "/api"

    # Note: the Phase 3 forensic layer lives inside the forensics package
    # (``app/forensics``); its adapter locates it from its own module path, so
    # no path setting is needed here.

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

    # ── Phase 5: Intelligence Layer LLM configuration ──────────────────────
    # The LLM is an *advisory* evidence-explanation layer. It is OFF by default
    # so the deterministic pipeline (fusion + risk + recommendation) runs fully
    # offline; enable it by setting APP_LLM_ENABLED=true and supplying a key.
    #
    # No API key is ever hard-coded — it comes from APP_LLM_API_KEY (or, when
    # that is unset, the provider SDK's own env var, e.g. ANTHROPIC_API_KEY).
    llm_enabled: bool = False
    # Provider behind the small LLMProvider interface. "anthropic" is the only
    # implemented HTTP provider; "none" forces the always-unavailable provider.
    llm_provider: str = "anthropic"
    # Model id. Default is the current most-capable Claude model.
    llm_model: str = "claude-opus-5"
    # Explicit key; when empty the anthropic SDK reads ANTHROPIC_API_KEY itself.
    llm_api_key: str | None = None
    # Optional override for a self-hosted / gateway endpoint.
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_output_tokens: int = 1200


settings = Settings()
