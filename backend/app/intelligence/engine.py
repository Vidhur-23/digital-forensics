"""Intelligence Layer orchestrator (Phase 5).

Ties the deterministic sub-modules and the advisory LLM into one
:class:`IntelligenceResult`, in the order the architecture requires:

    fusion  ->  risk  ->  explanation  ->  LLM evaluation  ->  recommendation

The deterministic chain (fusion/risk/explanation/recommendation) runs FIRST and
always; the LLM call is advisory and fully isolated inside the evaluator, so an
LLM outage leaves a complete, defensible result. This engine is injected into
the pipeline exactly like the Phase 2/3/4 services, so the LLM provider is
swappable in tests.
"""
from __future__ import annotations

from typing import Optional

from app.api.schemas.document import ScreeningResponse
from app.intelligence.explanation import build_explanation
from app.intelligence.fusion import fuse
from app.intelligence.llm.service import LLMEvaluator
from app.intelligence.recommendation import recommend
from app.intelligence.risk import assess_risk
from app.intelligence.schemas import IntelligenceResult


class IntelligenceEngine:
    """Runs the full Phase-5 evaluation over a completed screening response."""

    def __init__(self, llm_evaluator: Optional[LLMEvaluator] = None):
        # Default evaluator wraps the offline Null provider -> LLM UNAVAILABLE.
        self._llm = llm_evaluator or LLMEvaluator()

    def evaluate(self, response: ScreeningResponse) -> IntelligenceResult:
        # 1. Deterministic evidence fusion over the actual engine outputs.
        fused = fuse(response)

        # 2. Transparent decision-support risk (fills item weights).
        risk = assess_risk(fused)

        # 3. Deterministic, evidence-traceable explanation.
        explanation = build_explanation(fused, risk)

        # 4. Advisory LLM interpretation (isolated: never breaks the analysis).
        llm = self._llm.evaluate(response, fused, risk)

        # 5. Human-review recommendation from the deterministic assessment.
        recommendation = recommend(fused, risk, llm)

        return IntelligenceResult(
            fused_evidence=fused,
            risk=risk,
            explanation=explanation,
            recommendation=recommendation,
            llm=llm,
        )
