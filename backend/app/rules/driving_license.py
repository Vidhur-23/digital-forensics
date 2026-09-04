"""Driving-licence-specific rules — intentionally not implemented in Phase 2.

The Phase 1 pipeline currently provides no driving-licence extraction: there is
no DL field extractor (``app.ocr.extractor`` only registers a passport
extractor) and ``app.document.schemas.driving_license`` is empty. Per the task,
we do NOT fabricate placeholder rules for data the pipeline cannot supply.

This module exists as the extension point: when a DL extractor is added, register
a ``check_driving_license(result)`` function here and add it to the engine's
rule-set registry keyed by ``"driving_license"``. The engine architecture
(document type -> applicable rule set) already supports this without changes to
the passport path.
"""
from __future__ import annotations

from typing import List

from app.api.schemas.document import ScreeningResponse
from app.rules.schemas import RuleFinding


def check_driving_license(result: ScreeningResponse) -> List[RuleFinding]:
    """Placeholder: no DL rules until Phase 1 provides DL fields."""
    return []
