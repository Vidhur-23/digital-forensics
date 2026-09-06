"""Phase 5 Intelligence-Layer schemas.

These models are the *output contract* of the Intelligence Layer — the fifth
peer alongside Phase 2 ``RuleResults``, Phase 3 ``ForensicResults`` and Phase 4
``BiometricResults``. The whole Phase-5 result is attached to the shared
:class:`~app.api.schemas.document.ScreeningResponse` as a single optional
``intelligence`` field, so the analysis response stays one coherent object.

Design rules (mirroring the earlier phases):

* **Not a verdict.** The risk ``score`` is an inspectable *decision-support*
  number, never a calibrated probability of fraud. The human officer decides.
* **No duplicate engine schemas.** The Intelligence Layer does NOT re-represent
  a rule / forensic / biometric finding. It produces one small, normalised
  :class:`EvidenceItem` per underlying finding that always retains its
  ``source`` and points back (via ``id`` / ``evidence``) to the real engine
  output — the engines remain the source of truth.
* **Reused primitives.** Regions use the project-wide ``BBox`` type
  (``[x1, y1, x2, y2]`` in original-image pixels), exactly like OCR / field /
  forensic / face regions.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.ocr.schemas import BBox


# ---------------------------------------------------------------------------
# Normalised evidence (small internal representation, retains source)
# ---------------------------------------------------------------------------
class EvidenceSource(str, Enum):
    """Which engine produced the underlying finding."""

    RULES = "rules"
    FORENSICS = "forensics"
    BIOMETRICS = "biometrics"
    OCR = "ocr"


class Polarity(str, Enum):
    """How a normalised item bears on review concern."""

    CONCERN = "CONCERN"       # increases review concern (fail / anomaly / mismatch)
    REASSURING = "REASSURING"  # reduces concern (pass / clean / match)
    NEUTRAL = "NEUTRAL"       # inconclusive / not-applicable / unavailable


class NormalizedSeverity(str, Enum):
    """A single severity vocabulary the different engines are mapped onto.

    Deliberately the same five levels the Rules Engine already uses, so rule
    severities pass through unchanged and forensic (``low/medium/high``) and
    biometric (by decision strength) severities normalise onto the same scale.
    """

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceItem(BaseModel):
    """One engine finding normalised to a common shape.

    Every field mirrors a value the source engine already produced — this is
    normalisation, not reinterpretation. ``id`` is stable and traceable
    (``"<source>:<finding key>"``) so an explanation / the frontend can point
    back to the exact underlying finding.
    """

    id: str
    source: EvidenceSource
    finding_type: str            # rule_id / forensic finding_type / biometric status
    status: str                  # the source engine's own status string
    severity: NormalizedSeverity
    polarity: Polarity
    confidence: float = 1.0      # 0..1 (deterministic rules == 1.0)
    field: Optional[str] = None  # document field / conceptual region ("photo", "identity")
    region: Optional[BBox] = None
    message: str = ""
    signals: List[str] = Field(default_factory=list)
    # Concern weight assigned by the risk engine (0 for non-concern items).
    weight: float = 0.0
    # Small pointer back to the concrete engine values behind the finding.
    evidence: Dict[str, Any] = Field(default_factory=dict)


class CorrelationType(str, Enum):
    CORROBORATION = "CORROBORATION"   # independent engines flag the same issue
    CONTRADICTION = "CONTRADICTION"   # engines disagree (mixed evidence)
    CONSISTENCY = "CONSISTENCY"       # engines are broadly consistent (all clear)


class Correlation(BaseModel):
    """A cross-engine relationship between normalised evidence items."""

    type: CorrelationType
    field: Optional[str] = None
    sources: List[EvidenceSource] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    description: str = ""


class FusedEvidence(BaseModel):
    """Output of :mod:`app.intelligence.fusion`."""

    items: List[EvidenceItem] = Field(default_factory=list)
    correlations: List[Correlation] = Field(default_factory=list)
    concern_count: int = 0
    reassuring_count: int = 0
    neutral_count: int = 0
    sources_available: List[EvidenceSource] = Field(default_factory=list)
    sources_unavailable: List[EvidenceSource] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskContribution(BaseModel):
    """One evidence item's transparent contribution to the risk score."""

    evidence_id: str
    source: EvidenceSource
    points: float
    reason: str


class RiskAssessment(BaseModel):
    """Output of :mod:`app.intelligence.risk`.

    ``score`` is a 0–100 **decision-support** number derived from inspectable
    weights (see ``contributions``). It is explicitly NOT a calibrated
    probability that the document is fraudulent.
    """

    level: RiskLevel
    score: int = Field(ge=0, le=100)
    score_kind: str = "decision_support"
    disclaimer: str = (
        "Decision-support score derived from transparent evidence weights — "
        "NOT a calibrated probability that the document is fraudulent."
    )
    contributions: List[RiskContribution] = Field(default_factory=list)
    corroboration_bonus: float = 0.0
    rationale: str = ""


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
class EvidenceLocation(BaseModel):
    """Where a reviewer should look, traced to a real engine finding."""

    source: EvidenceSource
    field: Optional[str] = None
    region: Optional[BBox] = None
    note: str = ""


class Explanation(BaseModel):
    """Deterministic, evidence-traceable explanation (LLM-independent)."""

    overall_assessment: str = ""
    primary_concern: Optional[str] = None
    corroborating_evidence: List[str] = Field(default_factory=list)
    counter_evidence: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    evidence_locations: List[EvidenceLocation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
class RecommendedAction(str, Enum):
    NO_ADDITIONAL_ACTION = "NO_ADDITIONAL_ACTION"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    ENHANCED_VERIFICATION = "ENHANCED_VERIFICATION"
    ESCALATE = "ESCALATE"


class Recommendation(BaseModel):
    """Human-review-oriented recommendation. Never punitive/legal advice."""

    action: RecommendedAction
    reasons: List[str] = Field(default_factory=list)
    based_on: List[str] = Field(default_factory=list)     # evidence / correlation ids
    review_targets: List[EvidenceLocation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM evaluation (advisory)
# ---------------------------------------------------------------------------
class LLMStatus(str, Enum):
    COMPLETED = "COMPLETED"
    UNAVAILABLE = "UNAVAILABLE"


class LLMKeyFinding(BaseModel):
    source: str
    finding: str
    importance: str = "MEDIUM"  # HIGH / MEDIUM / LOW


class LLMEvaluation(BaseModel):
    """Structured LLM interpretation of the evidence — advisory only.

    The LLM never sets the risk level/score; ``agreement_with_risk`` merely
    records whether its reading lines up with the deterministic assessment.
    """

    status: LLMStatus
    model: Optional[str] = None
    summary: str = ""
    key_findings: List[LLMKeyFinding] = Field(default_factory=list)
    corroboration: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    # Advisory recommended action — does NOT override the deterministic one.
    suggested_action: Optional[str] = None
    # AGREE / PARTIAL / DISAGREE / UNKNOWN vs the deterministic risk level.
    agreement_with_risk: Optional[str] = None
    # Populated when status == UNAVAILABLE.
    error: Optional[str] = None

    @classmethod
    def unavailable(cls, error: str, model: Optional[str] = None) -> "LLMEvaluation":
        return cls(
            status=LLMStatus.UNAVAILABLE,
            model=model,
            error=error,
            summary=(
                "LLM evidence explanation was unavailable; the deterministic "
                "risk assessment and recommendation remain authoritative."
            ),
        )


# ---------------------------------------------------------------------------
# Final Phase-5 result
# ---------------------------------------------------------------------------
class IntelligenceResult(BaseModel):
    """The Intelligence Layer's complete output, attached as
    ``ScreeningResponse.intelligence``."""

    fused_evidence: FusedEvidence
    risk: RiskAssessment
    explanation: Explanation
    recommendation: Recommendation
    llm: LLMEvaluation
