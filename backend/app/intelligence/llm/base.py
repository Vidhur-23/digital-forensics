"""LLM provider interface (Phase 5).

The Intelligence Layer talks to the LLM only through this small interface, so
the provider is replaceable and every call site stays free of vendor detail.
A provider must never raise on a normal failure path — an unavailable/errored
LLM returns via :class:`LLMProviderError`, which the evaluator turns into an
explicit ``UNAVAILABLE`` result so the deterministic analysis keeps working.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Raised by a provider when it cannot produce a completion.

    The evaluator catches this and reports the LLM as UNAVAILABLE; it never
    becomes a successful/failed document verdict.
    """


class LLMProvider(ABC):
    """Minimal text-in / text-out interface for one structured LLM call."""

    #: Human-readable model identifier (for reporting/traceability).
    model: str = "unknown"

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap check: is this provider configured and usable at all?"""
        raise NotImplementedError

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the raw text completion for the given prompt.

        Must raise :class:`LLMProviderError` (not an arbitrary exception) on any
        failure so the caller can isolate LLM outages cleanly.
        """
        raise NotImplementedError
