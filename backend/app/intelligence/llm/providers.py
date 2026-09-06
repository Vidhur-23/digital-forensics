"""Concrete LLM providers + factory (Phase 5).

Two providers:

* :class:`NullLLMProvider` — always unavailable. The default, so the whole
  pipeline runs fully offline with no API key and no network. LLM explanation is
  simply reported as UNAVAILABLE.
* :class:`AnthropicLLMProvider` — backed by the official ``anthropic`` SDK
  (imported lazily so the package is an *optional* dependency: if it is not
  installed, or no key is configured, the provider reports itself unavailable
  instead of crashing). The API key is read from configuration / the SDK's own
  ``ANTHROPIC_API_KEY`` environment variable — never hard-coded.

:func:`build_provider` selects the provider from :class:`Settings`.
"""
from __future__ import annotations

from typing import Optional

from app.config import Settings, settings as default_settings
from app.intelligence.llm.base import LLMProvider, LLMProviderError


class NullLLMProvider(LLMProvider):
    """A provider that is never available (deterministic offline default)."""

    model = "none"

    def __init__(self, reason: str = "LLM is disabled or not configured."):
        self._reason = reason

    def is_available(self) -> bool:
        return False

    def complete(self, system: str, user: str) -> str:  # noqa: D401
        raise LLMProviderError(self._reason)


class AnthropicLLMProvider(LLMProvider):
    """LLM provider using the official Anthropic Python SDK.

    The SDK client is created lazily on first use; if the ``anthropic`` package
    is not installed or no credential is resolvable, the provider stays
    unavailable rather than raising at construction time.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 1200,
    ):
        self.model = model
        self._api_key = api_key or None
        self._base_url = base_url or None
        self._timeout = timeout_seconds
        self._max_tokens = max_output_tokens
        self._client = None
        self._init_error: Optional[str] = None

    # -- lazy client construction ------------------------------------------
    def _ensure_client(self):
        if self._client is not None or self._init_error is not None:
            return self._client
        try:
            import anthropic  # optional dependency
        except Exception as exc:  # pragma: no cover - import guard
            self._init_error = (
                f"anthropic SDK not installed ({exc!r}); "
                "install it or disable the LLM."
            )
            return None
        try:
            kwargs = {"timeout": self._timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            # With no explicit key the SDK resolves ANTHROPIC_API_KEY / a
            # profile itself; a missing credential surfaces on the first call.
            self._client = anthropic.Anthropic(**kwargs)
        except Exception as exc:
            self._init_error = f"Could not initialise Anthropic client: {exc!r}"
            return None
        return self._client

    def is_available(self) -> bool:
        # Available if we can construct a client (credential is validated on the
        # actual call, where failure is handled gracefully).
        return self._ensure_client() is not None

    def complete(self, system: str, user: str) -> str:
        client = self._ensure_client()
        if client is None:
            raise LLMProviderError(self._init_error or "LLM client unavailable.")
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # Keep this structured-reasoning call cheap and deterministic-ish.
                output_config={"effort": "low"},
            )
        except TypeError:
            # Older SDK without output_config — retry without it.
            try:
                message = client.messages.create(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except Exception as exc:  # pragma: no cover - network/runtime guard
                raise LLMProviderError(f"LLM request failed: {exc!r}") from exc
        except Exception as exc:  # any API/network/auth error
            raise LLMProviderError(f"LLM request failed: {exc!r}") from exc

        text = "".join(
            getattr(block, "text", "")
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise LLMProviderError("LLM returned an empty response.")
        return text


def build_provider(cfg: Optional[Settings] = None) -> LLMProvider:
    """Select the configured provider (defaults to the offline Null provider)."""
    cfg = cfg or default_settings
    if not cfg.llm_enabled:
        return NullLLMProvider("LLM disabled (APP_LLM_ENABLED is false).")
    provider = (cfg.llm_provider or "").lower()
    if provider in ("", "none", "null"):
        return NullLLMProvider("No LLM provider configured (APP_LLM_PROVIDER).")
    if provider == "anthropic":
        return AnthropicLLMProvider(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            timeout_seconds=cfg.llm_timeout_seconds,
            max_output_tokens=cfg.llm_max_output_tokens,
        )
    return NullLLMProvider(f"Unknown LLM provider '{cfg.llm_provider}'.")
