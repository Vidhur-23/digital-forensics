"""Evidence fusion (Phase 5).

Combines the ACTUAL structured outputs already on a :class:`ScreeningResponse`
— ``rules`` (:class:`RuleResults`), ``forensics`` (:class:`ForensicResults`),
``biometrics`` (:class:`BiometricResults`) and the Phase-1 ``ocr`` summary —
into a single normalised evidence view (:class:`FusedEvidence`).

Two responsibilities, and nothing more:

1. **Normalise** every engine finding to a common :class:`EvidenceItem` shape
   while *retaining its source* and a pointer back to the real finding. This is
   translation, not reinterpretation — no engine is re-run and no finding is
   invented.
2. **Correlate** across engines: recognise when independent engines flag the
   same field/region (CORROBORATION), when they disagree (CONTRADICTION, e.g. a
   forensic photo anomaly alongside a biometric MATCH), or when they are broadly
   consistent (CONSISTENCY). This is stronger than counting findings.

The fusion layer never assigns a risk score — that is :mod:`app.intelligence.risk`.
"""
from __future__ import annotations

from typing import List, Optional

from app.api.schemas.biometric import BiometricResults, BiometricStatus
from app.api.schemas.document import ScreeningResponse
from app.api.schemas.evidence import (
    ForensicFinding,
    ForensicResults,
    ForensicStatus,
)
from app.intelligence.schemas import (
    Correlation,
    CorrelationType,
    EvidenceItem,
    EvidenceSource,
    FusedEvidence,
    NormalizedSeverity,
    Polarity,
)
from app.ocr.schemas import BBox
from app.rules.schemas import RuleFinding, RuleResults, RuleStatus

# Forensic finding_type values that represent a per-layer *error*, not a
# clean/anomalous result (see the forensic engine's finding-type vocabulary).
_FORENSIC_ERROR_TYPES = {
    "processing_error",
    "calibration_missing",
    "feature_extraction_failure",
    "roi_extraction_failure",
    "insufficient_image_quality",
    "invalid_input",
}

# Map the forensic engine's lowercase severity to the normalised scale.
_FORENSIC_SEVERITY = {
    "low": NormalizedSeverity.LOW,
    "medium": NormalizedSeverity.MEDIUM,
    "high": NormalizedSeverity.HIGH,
}

# Biometric decision strength -> a confidence proxy for the comparison.
_STRENGTH_CONFIDENCE = {"HIGH": 0.95, "MEDIUM": 0.8, "LOW": 0.6}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def fuse(response: ScreeningResponse) -> FusedEvidence:
    """Normalise + correlate all engine outputs on ``response``."""
    items: List[EvidenceItem] = []
    available: List[EvidenceSource] = []
    unavailable: List[EvidenceSource] = []

    # --- Rules (Phase 2) ---------------------------------------------------
    if response.rules is not None:
        available.append(EvidenceSource.RULES)
        items.extend(_normalize_rules(response.rules, response))

    # --- Forensics (Phase 3) ----------------------------------------------
    if response.forensics is not None:
        if response.forensics.status == ForensicStatus.UNAVAILABLE:
            unavailable.append(EvidenceSource.FORENSICS)
        else:
            available.append(EvidenceSource.FORENSICS)
        items.extend(_normalize_forensics(response.forensics))

    # --- Biometrics (Phase 4) ---------------------------------------------
    if response.biometrics is not None:
        if response.biometrics.status in (
            BiometricStatus.MATCH,
            BiometricStatus.MISMATCH,
        ):
            available.append(EvidenceSource.BIOMETRICS)
        else:
            unavailable.append(EvidenceSource.BIOMETRICS)
        items.append(_normalize_biometrics(response.biometrics))

    # --- OCR context (Phase 1) --------------------------------------------
    ocr_item = _normalize_ocr(response)
    if ocr_item is not None:
        items.append(ocr_item)

    correlations = _correlate(items)

    concern = sum(1 for i in items if i.polarity == Polarity.CONCERN)
    reassuring = sum(1 for i in items if i.polarity == Polarity.REASSURING)
    neutral = sum(1 for i in items if i.polarity == Polarity.NEUTRAL)

    return FusedEvidence(
        items=items,
        correlations=correlations,
        concern_count=concern,
        reassuring_count=reassuring,
        neutral_count=neutral,
        sources_available=available,
        sources_unavailable=unavailable,
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _field_region(response: ScreeningResponse, field: Optional[str]) -> Optional[BBox]:
    """Region of a document field, so a rule about it can be localised /
    correlated against a forensic anomaly by pixel overlap."""
    if not field:
        return None
    fv = response.fields.get(field)
    if fv is not None and fv.bbox:
        return fv.bbox
    if field == "mrz" and response.mrz is not None and response.mrz.bbox:
        return response.mrz.bbox
    return None


def _normalize_rules(
    rules: RuleResults, response: ScreeningResponse
) -> List[EvidenceItem]:
    out: List[EvidenceItem] = []
    for f in rules.findings:
        out.append(_rule_to_item(f, response))
    return out


def _rule_to_item(f: RuleFinding, response: ScreeningResponse) -> EvidenceItem:
    if f.status == RuleStatus.PASS:
        polarity = Polarity.REASSURING
        severity = NormalizedSeverity.INFO
    elif f.status == RuleStatus.NOT_APPLICABLE:
        polarity = Polarity.NEUTRAL
        severity = NormalizedSeverity.INFO
    else:  # WARNING / FAIL
        polarity = Polarity.CONCERN
        # Rule severities already use the normalised vocabulary.
        severity = NormalizedSeverity(f.severity.value)

    return EvidenceItem(
        id=f"rules:{f.rule_id}",
        source=EvidenceSource.RULES,
        finding_type=f.rule_id,
        status=f.status.value,
        severity=severity,
        polarity=polarity,
        confidence=1.0,  # deterministic given the extracted values
        field=f.field,
        region=_field_region(response, f.field),
        message=f.message,
        signals=[f.category],
        evidence=dict(f.evidence),
    )


def _normalize_forensics(forensics: ForensicResults) -> List[EvidenceItem]:
    if forensics.status == ForensicStatus.UNAVAILABLE:
        return [
            EvidenceItem(
                id="forensics:unavailable",
                source=EvidenceSource.FORENSICS,
                finding_type="unavailable",
                status="UNAVAILABLE",
                severity=NormalizedSeverity.INFO,
                polarity=Polarity.NEUTRAL,
                confidence=0.0,
                message=forensics.error or "Forensic analysis was unavailable.",
            )
        ]
    return [_forensic_to_item(f) for f in forensics.findings]


def _forensic_to_item(f: ForensicFinding) -> EvidenceItem:
    is_error = f.status == "error" or f.finding_type in _FORENSIC_ERROR_TYPES
    if is_error:
        polarity = Polarity.NEUTRAL
        severity = NormalizedSeverity.INFO
    elif f.passed or f.finding_type == "normal":
        polarity = Polarity.REASSURING
        severity = NormalizedSeverity.INFO
    else:  # a localised/flagged anomaly
        polarity = Polarity.CONCERN
        severity = _FORENSIC_SEVERITY.get(f.severity.lower(), NormalizedSeverity.MEDIUM)

    # The implemented forensic layers localise the portrait/photo region; a
    # located anomaly is therefore about the "photo" area of the document.
    field = "photo" if (f.region and polarity == Polarity.CONCERN) else None

    return EvidenceItem(
        id=f"forensics:{f.layer_name}",
        source=EvidenceSource.FORENSICS,
        finding_type=f.finding_type,
        status="anomaly" if polarity == Polarity.CONCERN else f.finding_type,
        severity=severity,
        polarity=polarity,
        confidence=float(f.confidence),
        field=field,
        region=f.region,
        message=f.message,
        signals=list(f.signals),
        evidence=dict(f.raw_metric),
    )


def _normalize_biometrics(bio: BiometricResults) -> EvidenceItem:
    confidence = _STRENGTH_CONFIDENCE.get(bio.decision_strength or "", 0.7)
    region = bio.document_face.bbox if bio.document_face else None

    if bio.status == BiometricStatus.MATCH:
        polarity = Polarity.REASSURING
        severity = NormalizedSeverity.INFO
        field = "photo"
    elif bio.status == BiometricStatus.MISMATCH:
        polarity = Polarity.CONCERN
        # A confident identity mismatch is the strongest single-engine concern.
        severity = (
            NormalizedSeverity.CRITICAL
            if bio.decision_strength == "HIGH"
            else NormalizedSeverity.HIGH
        )
        field = "identity"
    else:
        # NO_FACE / INSUFFICIENT_QUALITY / AMBIGUOUS / UNAVAILABLE == not compared.
        polarity = Polarity.NEUTRAL
        severity = NormalizedSeverity.INFO
        confidence = 0.0
        field = None

    return EvidenceItem(
        id=f"biometrics:{bio.status.value}",
        source=EvidenceSource.BIOMETRICS,
        finding_type=bio.status.value,
        status=bio.status.value,
        severity=severity,
        polarity=polarity,
        confidence=confidence,
        field=field,
        region=region,
        message=bio.message,
        signals=[bio.status.value],
        evidence={
            "similarity": bio.similarity,
            "threshold": bio.threshold,
            "decision_strength": bio.decision_strength,
        },
    )


def _normalize_ocr(response: ScreeningResponse) -> Optional[EvidenceItem]:
    """A light OCR-confidence context item (never a standalone verdict)."""
    if response.ocr is None:
        return None
    conf = float(response.ocr.confidence)
    # Very poor OCR undermines every downstream comparison — surface it as a
    # low concern; otherwise it is neutral context for the LLM.
    if conf < 0.5 and response.ocr.word_count > 0:
        polarity = Polarity.CONCERN
        severity = NormalizedSeverity.LOW
        message = (
            f"Overall OCR confidence is low ({conf:.2f}); field reads used by "
            "other checks may be unreliable."
        )
    else:
        polarity = Polarity.NEUTRAL
        severity = NormalizedSeverity.INFO
        message = f"Overall OCR confidence {conf:.2f} across {response.ocr.word_count} words."

    return EvidenceItem(
        id="ocr:confidence",
        source=EvidenceSource.OCR,
        finding_type="ocr_confidence",
        status="ok",
        severity=severity,
        polarity=polarity,
        confidence=conf,
        message=message,
        signals=["ocr_quality"],
        evidence={"mean_confidence": conf, "word_count": response.ocr.word_count},
    )


# ---------------------------------------------------------------------------
# Cross-engine correlation
# ---------------------------------------------------------------------------
def _iou(a: Optional[BBox], b: Optional[BBox]) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap(a: EvidenceItem, b: EvidenceItem) -> bool:
    """Two items concern the same place: same non-null field, or overlapping
    pixel regions (IoU > 0.1)."""
    if a.field and b.field and a.field == b.field:
        return True
    return _iou(a.region, b.region) > 0.1


def _correlate(items: List[EvidenceItem]) -> List[Correlation]:
    concern = [i for i in items if i.polarity == Polarity.CONCERN]
    reassuring = [i for i in items if i.polarity == Polarity.REASSURING]
    correlations: List[Correlation] = []
    seen_pairs: set = set()

    # 1. CORROBORATION — two concern items from *different* engines about the
    #    same field/region. Independent engines agreeing is stronger evidence.
    for i, a in enumerate(concern):
        for b in concern[i + 1:]:
            if a.source == b.source:
                continue
            if not _overlap(a, b):
                continue
            key = tuple(sorted((a.id, b.id)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            field = a.field or b.field
            where = f" affecting the {field}" if field else " affecting the same region"
            correlations.append(
                Correlation(
                    type=CorrelationType.CORROBORATION,
                    field=field,
                    sources=[a.source, b.source],
                    evidence_ids=[a.id, b.id],
                    description=(
                        f"{a.source.value} and {b.source.value} independently "
                        f"flag an issue{where}."
                    ),
                )
            )

    # 2. CONTRADICTION — a concern item and a reassuring item from different
    #    engines about the same place (mixed evidence). A biometric MATCH must
    #    NOT erase a forensic photo anomaly — it is recorded as a contradiction.
    for a in concern:
        for b in reassuring:
            if a.source == b.source or not _overlap(a, b):
                continue
            key = tuple(sorted((a.id, b.id)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            field = a.field or b.field
            where = f" on the {field}" if field else " on the same region"
            correlations.append(
                Correlation(
                    type=CorrelationType.CONTRADICTION,
                    field=field,
                    sources=[a.source, b.source],
                    evidence_ids=[a.id, b.id],
                    description=(
                        f"Mixed evidence{where}: {a.source.value} flags a concern "
                        f"while {b.source.value} is reassuring — the reassuring "
                        "signal does not eliminate the concern."
                    ),
                )
            )

    # 3. Identity vs document-consistency contradiction — a biometric MISMATCH
    #    while the document's own fields are internally consistent (no rule
    #    FAIL) is a distinct mixed case worth calling out even without a shared
    #    region.
    bio_mismatch = next(
        (i for i in concern if i.source == EvidenceSource.BIOMETRICS), None
    )
    doc_concern = any(
        i.source in (EvidenceSource.RULES, EvidenceSource.FORENSICS) for i in concern
    )
    if bio_mismatch is not None and not doc_concern:
        correlations.append(
            Correlation(
                type=CorrelationType.CONTRADICTION,
                field="identity",
                sources=[EvidenceSource.BIOMETRICS],
                evidence_ids=[bio_mismatch.id],
                description=(
                    "Document information appears internally consistent, but the "
                    "biometric identity comparison does not correspond — identity "
                    "correspondence is questionable and requires review."
                ),
            )
        )

    # 4. CONSISTENCY — no concern anywhere and at least two engines reassuring.
    if not concern and len({r.source for r in reassuring}) >= 2:
        correlations.append(
            Correlation(
                type=CorrelationType.CONSISTENCY,
                sources=sorted({r.source for r in reassuring}, key=lambda s: s.value),
                evidence_ids=[r.id for r in reassuring],
                description=(
                    "Rules, forensics and biometrics evidence are broadly "
                    "consistent; no inconsistencies were detected."
                ),
            )
        )

    return correlations
