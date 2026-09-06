"""Explainability (Phase 5).

Two jobs:

1. :func:`build_explanation` — a fully deterministic, LLM-independent
   explanation. Every line traces back to a real engine finding, answering:
   what was found, which engine found it, where, how strong it is, whether it
   is corroborated or contradicted, and why it moved the risk assessment. This
   is what makes the system defensible when the LLM is unavailable.

2. :func:`build_llm_context` — assembles the *grounding* JSON handed to the LLM:
   the actual structured engine outputs (no raw images), plus the fused evidence
   and the deterministic risk assessment. The LLM interprets this; it is never
   asked to rediscover evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.api.schemas.document import ScreeningResponse
from app.intelligence.schemas import (
    CorrelationType,
    EvidenceItem,
    EvidenceLocation,
    Explanation,
    FusedEvidence,
    Polarity,
    RiskAssessment,
)


def build_explanation(
    fused: FusedEvidence, risk: RiskAssessment
) -> Explanation:
    """Deterministic explanation derived solely from fused evidence + risk."""
    concern = [i for i in fused.items if i.polarity == Polarity.CONCERN]
    concern.sort(key=lambda i: i.weight, reverse=True)
    reassuring = [i for i in fused.items if i.polarity == Polarity.REASSURING]

    overall = (
        f"{risk.level.value} review priority "
        f"(decision-support score {risk.score}/100). {risk.disclaimer}"
    )

    primary = None
    if concern:
        top = concern[0]
        primary = f"[{top.source.value}] {top.message}".strip()

    corroborating = [
        c.description
        for c in fused.correlations
        if c.type == CorrelationType.CORROBORATION
    ]
    counter = [
        c.description
        for c in fused.correlations
        if c.type == CorrelationType.CONTRADICTION
    ]
    # Also surface reassuring findings as counter-evidence context.
    for r in reassuring:
        if r.message:
            counter.append(f"[{r.source.value}] {r.message}")

    reasoning = _build_reasoning(concern, fused, risk)

    locations = [
        EvidenceLocation(
            source=i.source,
            field=i.field,
            region=i.region,
            note=i.message,
        )
        for i in concern
        if i.field or i.region
    ]

    return Explanation(
        overall_assessment=overall,
        primary_concern=primary,
        corroborating_evidence=corroborating,
        counter_evidence=counter,
        reasoning=reasoning,
        evidence_locations=locations,
    )


def _build_reasoning(
    concern: List[EvidenceItem], fused: FusedEvidence, risk: RiskAssessment
) -> List[str]:
    lines: List[str] = []
    if not concern:
        lines.append(
            "No concern-raising findings were produced by the available engines."
        )
    for i in concern:
        where = f" on {i.field}" if i.field else ""
        lines.append(
            f"{i.source.value} found '{i.finding_type}'{where} "
            f"(severity {i.severity.value}, confidence {i.confidence:.2f}) "
            f"→ contributed {i.weight:.1f} points."
        )
    if risk.corroboration_bonus > 0:
        lines.append(
            f"Corroboration between independent engines added "
            f"{risk.corroboration_bonus:.1f} points — agreement across engines "
            "is stronger than any single finding."
        )
    for c in fused.correlations:
        if c.type == CorrelationType.CONTRADICTION:
            lines.append("Contradiction/mixed evidence: " + c.description)
    if fused.sources_unavailable:
        lines.append(
            "Unavailable engines ("
            + ", ".join(s.value for s in fused.sources_unavailable)
            + ") are treated as unknown, never as clean."
        )
    lines.append(f"Overall: {risk.rationale}")
    return lines


# ---------------------------------------------------------------------------
# LLM grounding context
# ---------------------------------------------------------------------------
def build_llm_context(
    response: ScreeningResponse, fused: FusedEvidence, risk: RiskAssessment
) -> Dict[str, Any]:
    """Assemble the authoritative JSON evidence handed to the LLM.

    Uses the ACTUAL engine outputs (trimmed of bulky, non-semantic data such as
    the full OCR word list and raw image pixels). No raw document image is ever
    included — only structured findings and minimal metadata.
    """
    document = {
        "document_type": response.document_type,
        "document_type_confidence": response.document_type_confidence,
        # Field values only (labels + reads) — no pixels.
        "fields": {
            name: {"value": fv.value, "source": fv.source, "confidence": fv.confidence}
            for name, fv in response.fields.items()
        },
        "mrz": {
            "detected": response.mrz.detected if response.mrz else False,
            "fields": response.mrz.fields.model_dump() if response.mrz else {},
        },
        "ocr_confidence": response.ocr.confidence if response.ocr else None,
    }

    rules = response.rules.model_dump(mode="json") if response.rules else None

    forensics = None
    if response.forensics is not None:
        forensics = response.forensics.model_dump(mode="json")
        # Drop verbose per-layer raw arrays that carry no interpretive value.
        for f in forensics.get("findings", []):
            f.pop("raw_metric", None)

    biometrics = None
    if response.biometrics is not None:
        biometrics = response.biometrics.model_dump(mode="json")
        # Never expose raw quality signal internals to the LLM.
        for face in ("document_face", "reference_face"):
            if isinstance(biometrics.get(face), dict):
                biometrics[face].pop("quality_signals", None)

    return {
        "document": document,
        "engine_outputs": {
            "rules": rules,
            "forensics": forensics,
            "biometrics": biometrics,
        },
        "fused_evidence": fused.model_dump(mode="json"),
        "deterministic_risk": {
            "level": risk.level.value,
            "score": risk.score,
            "score_kind": risk.score_kind,
            "rationale": risk.rationale,
        },
    }
