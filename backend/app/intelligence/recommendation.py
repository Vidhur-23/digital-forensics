"""Officer recommendation (Phase 5).

Maps the deterministic risk assessment and fused evidence onto a human-review
action. The recommendation is review-oriented and never punitive/legal — the
human officer remains the final decision-maker.

Escalation is reserved for the strongest case: multiple independent engines
corroborating an inconsistency. A single-engine HIGH concern (e.g. an identity
mismatch on its own) calls for enhanced verification rather than escalation.
"""
from __future__ import annotations

from typing import List

from app.intelligence.schemas import (
    Correlation,
    CorrelationType,
    EvidenceItem,
    EvidenceLocation,
    FusedEvidence,
    LLMEvaluation,
    Polarity,
    Recommendation,
    RecommendedAction,
    RiskAssessment,
    RiskLevel,
)


def recommend(
    fused: FusedEvidence,
    risk: RiskAssessment,
    llm: LLMEvaluation | None = None,
) -> Recommendation:
    concern = [i for i in fused.items if i.polarity == Polarity.CONCERN]
    corroborations = [
        c for c in fused.correlations if c.type == CorrelationType.CORROBORATION
    ]
    multi_engine_corroboration = any(len(set(c.sources)) >= 2 for c in corroborations)

    action = _action_for(risk.level, concern, multi_engine_corroboration)

    reasons = _reasons(action, risk, concern, corroborations, fused)
    based_on = [i.id for i in concern] + [
        "+".join(c.evidence_ids) for c in corroborations
    ]
    review_targets = [
        EvidenceLocation(
            source=i.source, field=i.field, region=i.region, note=i.message
        )
        for i in sorted(concern, key=lambda x: x.weight, reverse=True)
        if i.field or i.region
    ]

    return Recommendation(
        action=action,
        reasons=reasons,
        based_on=based_on,
        review_targets=review_targets,
    )


def _action_for(
    level: RiskLevel,
    concern: List[EvidenceItem],
    multi_engine_corroboration: bool,
) -> RecommendedAction:
    if level == RiskLevel.HIGH:
        # Corroborated, multi-engine inconsistency -> escalate for senior /
        # enhanced review; otherwise enhanced verification.
        return (
            RecommendedAction.ESCALATE
            if multi_engine_corroboration
            else RecommendedAction.ENHANCED_VERIFICATION
        )
    if level == RiskLevel.MEDIUM:
        return RecommendedAction.MANUAL_REVIEW
    # LOW
    return (
        RecommendedAction.MANUAL_REVIEW
        if concern
        else RecommendedAction.NO_ADDITIONAL_ACTION
    )


def _reasons(
    action: RecommendedAction,
    risk: RiskAssessment,
    concern: List[EvidenceItem],
    corroborations: List[Correlation],
    fused: FusedEvidence,
) -> List[str]:
    reasons: List[str] = []
    if action == RecommendedAction.NO_ADDITIONAL_ACTION:
        reasons.append("No significant inconsistencies detected across engines.")
    else:
        top = sorted(concern, key=lambda x: x.weight, reverse=True)
        for i in top[:3]:
            where = f" on {i.field}" if i.field else ""
            reasons.append(f"[{i.source.value}] {i.message}".strip() or
                           f"{i.severity.value} concern{where}.")
    if corroborations:
        reasons.append(
            "Multiple independent engines identify corroborating inconsistencies."
        )
    if fused.sources_unavailable:
        reasons.append(
            "Some engines were unavailable ("
            + ", ".join(s.value for s in fused.sources_unavailable)
            + "); their evidence is unknown, not clean."
        )
    return reasons
