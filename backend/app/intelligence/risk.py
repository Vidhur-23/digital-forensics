"""Risk assessment (Phase 5).

Turns :class:`FusedEvidence` into a transparent, inspectable
:class:`RiskAssessment`. Every design constraint from the task spec is honoured:

* **Not a fraud probability.** ``score`` is a 0–100 *decision-support* number
  built from explicit weights; it is never described as a calibrated probability
  of fraud.
* **Not "count the failures".** Each concern item contributes points from a
  severity base, scaled by how reliable its source is and how confident the
  finding is. One HIGH forensic finding corroborated by a HIGH rules finding on
  the same region therefore outweighs three LOW informational warnings.
* **Scores are not blindly summed.** Different engine scores (OCR confidence,
  forensic anomaly score, biometric similarity) are NEVER added together. Only
  the semantic *severity/polarity* of each finding feeds the weight; the raw
  engine scores are used only as per-source confidence, never cross-added.
* **Contradictory / reassuring evidence is handled explicitly.** Reassuring
  findings (PASS / clean / MATCH) never *subtract* from a concern — a biometric
  MATCH cannot erase a forensic anomaly — they simply fail to add concern, so a
  fully clean document lands at score 0 / LOW.

All weights live here as named constants so the whole calculation is auditable.
"""
from __future__ import annotations

from typing import List

from app.intelligence.schemas import (
    Correlation,
    CorrelationType,
    EvidenceItem,
    FusedEvidence,
    NormalizedSeverity,
    Polarity,
    RiskAssessment,
    RiskContribution,
    RiskLevel,
)

# --- Centralised, inspectable weights --------------------------------------

# Base concern points per normalised severity.
SEVERITY_POINTS = {
    NormalizedSeverity.INFO: 0.0,
    NormalizedSeverity.LOW: 5.0,
    NormalizedSeverity.MEDIUM: 12.0,
    NormalizedSeverity.HIGH: 25.0,
    NormalizedSeverity.CRITICAL: 40.0,
}

# How much to trust each source. Deterministic rules and a biometric identity
# decision are strong; a forensic anomaly is softer, corroborating evidence;
# OCR quality is weak context.
SOURCE_RELIABILITY = {
    "rules": 1.0,
    "biometrics": 1.0,
    "forensics": 0.8,
    "ocr": 0.5,
}

# Corroboration between independent engines adds a bonus proportional to the
# strength of the corroborated findings (independent agreement is worth more
# than the sum of the parts).
CORROBORATION_BONUS_FACTOR = 0.5

# Score -> level thresholds (decision-support bands, not probabilities).
MEDIUM_THRESHOLD = 15
HIGH_THRESHOLD = 35

MAX_SCORE = 100


def _confidence_multiplier(confidence: float) -> float:
    """Low-confidence findings count less; a full-confidence finding counts
    fully. Maps confidence 0..1 onto 0.5..1.0 so nothing is fully discounted."""
    c = max(0.0, min(1.0, confidence))
    return 0.5 + 0.5 * c


def _item_points(item: EvidenceItem) -> float:
    base = SEVERITY_POINTS.get(item.severity, 0.0)
    reliability = SOURCE_RELIABILITY.get(item.source.value, 1.0)
    return base * reliability * _confidence_multiplier(item.confidence)


def assess_risk(fused: FusedEvidence) -> RiskAssessment:
    """Compute the decision-support risk assessment from fused evidence.

    Mutates each concern :class:`EvidenceItem`'s ``weight`` in place so the
    frontend/explanation can show exactly how much each finding mattered.
    """
    contributions: List[RiskContribution] = []
    concern_items = [i for i in fused.items if i.polarity == Polarity.CONCERN]

    total = 0.0
    for item in concern_items:
        points = round(_item_points(item), 2)
        item.weight = points
        total += points
        if points > 0:
            contributions.append(
                RiskContribution(
                    evidence_id=item.id,
                    source=item.source,
                    points=points,
                    reason=(
                        f"{item.severity.value} {item.source.value} finding "
                        f"({item.finding_type})"
                        + (f" on {item.field}" if item.field else "")
                    ),
                )
            )

    corroboration_bonus = _corroboration_bonus(fused.correlations, concern_items)
    total += corroboration_bonus

    score = int(min(MAX_SCORE, round(total)))
    level = _level_for(score)

    return RiskAssessment(
        level=level,
        score=score,
        contributions=sorted(contributions, key=lambda c: c.points, reverse=True),
        corroboration_bonus=round(corroboration_bonus, 2),
        rationale=_rationale(level, score, concern_items, corroboration_bonus, fused),
    )


def _corroboration_bonus(
    correlations: List[Correlation], concern_items: List[EvidenceItem]
) -> float:
    by_id = {i.id: i for i in concern_items}
    bonus = 0.0
    for c in correlations:
        if c.type != CorrelationType.CORROBORATION:
            continue
        linked = [by_id[e] for e in c.evidence_ids if e in by_id]
        if len(linked) < 2:
            continue
        bonus += CORROBORATION_BONUS_FACTOR * sum(_item_points(i) for i in linked)
    return round(bonus, 2)


def _level_for(score: int) -> RiskLevel:
    if score >= HIGH_THRESHOLD:
        return RiskLevel.HIGH
    if score >= MEDIUM_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _rationale(
    level: RiskLevel,
    score: int,
    concern_items: List[EvidenceItem],
    bonus: float,
    fused: FusedEvidence,
) -> str:
    if not concern_items:
        base = (
            f"No concern-raising findings across the available engines; "
            f"decision-support score {score}/100 ({level.value})."
        )
    else:
        n = len(concern_items)
        sources = sorted({i.source.value for i in concern_items})
        base = (
            f"{n} concern-raising finding(s) from {', '.join(sources)} "
            f"produce a decision-support score of {score}/100 ({level.value})."
        )
        if bonus > 0:
            base += (
                f" Independent engines corroborate on the same evidence "
                f"(+{bonus:.0f} corroboration weight)."
            )
    if fused.sources_unavailable:
        base += (
            " Note: "
            + ", ".join(s.value for s in fused.sources_unavailable)
            + " evidence was unavailable and is treated as unknown, not clean."
        )
    return base
