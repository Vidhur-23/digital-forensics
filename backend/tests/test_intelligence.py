"""Phase 5 Intelligence-Layer tests.

Two distinct concerns are tested separately (task requirement #28):

* **Engine evaluation** — given fixed engine outputs, the deterministic fusion /
  risk / recommendation produce the expected level, corroboration and mixed-
  evidence handling. These build a :class:`ScreeningResponse` directly from the
  real Phase 2-4 schemas (no images, no models) so they are fully deterministic.
* **LLM evaluation** — given fixed engine JSON, a *stub* provider exercises
  grounding, structured output, the anti-hallucination guard, and the mandatory
  graceful-degradation path when the LLM is unavailable.

No real LLM is called.
"""
from __future__ import annotations

import json

import pytest

from app.api.schemas.biometric import (
    BiometricResults,
    BiometricStatus,
    FaceEvidence,
    FaceQuality,
)
from app.api.schemas.document import (
    FieldValue,
    ImageInfo,
    MRZFieldsOut,
    MRZInfo,
    OCRInfo,
    ScreeningResponse,
)
from app.api.schemas.evidence import (
    ForensicFinding,
    ForensicResults,
    ForensicStatus,
    ForensicSummary,
)
from app.intelligence.engine import IntelligenceEngine
from app.intelligence.llm.base import LLMProvider, LLMProviderError
from app.intelligence.llm.service import LLMEvaluator
from app.intelligence.schemas import (
    CorrelationType,
    EvidenceSource,
    LLMStatus,
    Polarity,
    RecommendedAction,
    RiskLevel,
)
from app.rules.schemas import RuleFinding, RuleResults, RuleSeverity, RuleStatus


# ---------------------------------------------------------------------------
# Builders — assemble a ScreeningResponse from the ACTUAL engine schemas
# ---------------------------------------------------------------------------
_DOB_BBOX = [100, 300, 400, 340]
_PHOTO_BBOX = [400, 100, 600, 300]


def _response(rules=None, forensics=None, biometrics=None) -> ScreeningResponse:
    resp = ScreeningResponse(
        document_type="passport",
        document_type_confidence=0.95,
        image=ImageInfo(width=1000, height=1000),
        fields={
            "date_of_birth": FieldValue(
                value="12 AUG 1974", confidence=0.9, bbox=_DOB_BBOX, source="visual"
            )
        },
        mrz=MRZInfo(detected=True, bbox=[100, 900, 900, 980], fields=MRZFieldsOut()),
        ocr=OCRInfo(confidence=0.92, word_count=40),
    )
    resp.rules = rules
    resp.forensics = forensics
    resp.biometrics = biometrics
    return resp


def _rules_pass() -> RuleResults:
    return RuleResults.from_findings(
        [
            RuleFinding.make(
                "CONSISTENCY_DOB", "consistency", RuleStatus.PASS, RuleSeverity.INFO,
                "Visual DOB matches MRZ DOB.", field="date_of_birth",
            )
        ]
    )


def _rules_dob_mismatch() -> RuleResults:
    return RuleResults.from_findings(
        [
            RuleFinding.make(
                "CONSISTENCY_DOB", "consistency", RuleStatus.FAIL, RuleSeverity.HIGH,
                "Visual date of birth differs from MRZ date of birth.",
                field="date_of_birth",
                evidence={"visual_value": "1974-08-12", "mrz_value": "1980-01-01"},
            )
        ]
    )


def _forensics_clean() -> ForensicResults:
    return ForensicResults(
        status=ForensicStatus.COMPLETED,
        findings=[
            ForensicFinding(
                layer_name="ela", finding_type="normal", passed=True, score=0.1,
                severity="low", confidence=0.9, message="No compression anomaly.",
            )
        ],
        summary=ForensicSummary(layers_completed=4, all_passed=True),
    )


def _forensics_dob_anomaly() -> ForensicResults:
    return ForensicResults(
        status=ForensicStatus.COMPLETED,
        findings=[
            ForensicFinding(
                layer_name="ela", finding_type="compression_artifact_anomaly",
                passed=False, score=0.82, severity="high", confidence=0.85,
                region=[110, 305, 390, 338],  # overlaps the DOB field bbox
                signals=["compression_artifact_anomaly"],
                message="DOB region shows compression inconsistency vs neighbours.",
            )
        ],
        summary=ForensicSummary(layers_completed=4, flagged_count=1),
    )


def _forensics_photo_anomaly() -> ForensicResults:
    return ForensicResults(
        status=ForensicStatus.COMPLETED,
        findings=[
            ForensicFinding(
                layer_name="photo_boundary", finding_type="photo_boundary_anomaly",
                passed=False, score=0.7, severity="medium", confidence=0.8,
                region=_PHOTO_BBOX, signals=["photo_boundary_anomaly"],
                message="Photo boundary shows splicing characteristics.",
            )
        ],
        summary=ForensicSummary(layers_completed=4, flagged_count=1),
    )


def _bio_match() -> BiometricResults:
    return BiometricResults(
        status=BiometricStatus.MATCH, similarity=0.98, threshold=0.5,
        decision_strength="HIGH",
        document_face=FaceEvidence(detected=True, bbox=_PHOTO_BBOX, quality=FaceQuality.GOOD),
        reference_face=FaceEvidence(detected=True, quality=FaceQuality.GOOD),
        message="Document and reference faces correspond.",
    )


def _bio_mismatch() -> BiometricResults:
    return BiometricResults(
        status=BiometricStatus.MISMATCH, similarity=0.14, threshold=0.5,
        decision_strength="HIGH",
        document_face=FaceEvidence(detected=True, bbox=_PHOTO_BBOX, quality=FaceQuality.GOOD),
        reference_face=FaceEvidence(detected=True, quality=FaceQuality.GOOD),
        message="Document and reference faces do not correspond.",
    )


def _evaluate(resp) -> "IntelligenceResult":  # noqa: F821
    return IntelligenceEngine().evaluate(resp)  # default = offline Null LLM


# ---------------------------------------------------------------------------
# Engine evaluation — the six required scenarios
# ---------------------------------------------------------------------------
def test_scenario1_clean_is_low():
    r = _evaluate(_response(_rules_pass(), _forensics_clean(), _bio_match()))
    assert r.risk.level == RiskLevel.LOW
    assert r.risk.score == 0
    assert r.recommendation.action == RecommendedAction.NO_ADDITIONAL_ACTION
    # Broad consistency is recognised.
    assert any(c.type == CorrelationType.CONSISTENCY for c in r.fused_evidence.correlations)


def test_scenario2_logical_inconsistency_elevated_and_named():
    r = _evaluate(_response(_rules_dob_mismatch(), _forensics_clean(), _bio_match()))
    # A single deterministic mismatch elevates concern above LOW.
    assert r.risk.level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
    assert r.recommendation.action != RecommendedAction.NO_ADDITIONAL_ACTION
    # The DOB mismatch is clearly identified as the primary concern.
    assert "date of birth" in (r.explanation.primary_concern or "").lower()
    assert any(
        i.source == EvidenceSource.RULES and i.field == "date_of_birth"
        and i.polarity == Polarity.CONCERN
        for i in r.fused_evidence.items
    )


def test_scenario3_corroborated_manipulation_is_high():
    r = _evaluate(_response(_rules_dob_mismatch(), _forensics_dob_anomaly(), _bio_match()))
    assert r.risk.level == RiskLevel.HIGH
    # Rules + forensics are identified as corroborating on the same region.
    corr = [c for c in r.fused_evidence.correlations if c.type == CorrelationType.CORROBORATION]
    assert corr, "expected a corroboration correlation"
    assert {EvidenceSource.RULES, EvidenceSource.FORENSICS} <= set(corr[0].sources)
    assert r.risk.corroboration_bonus > 0
    assert r.recommendation.action == RecommendedAction.ESCALATE


def test_scenario4_biometric_mismatch_is_high_and_identity_named():
    r = _evaluate(_response(_rules_pass(), _forensics_clean(), _bio_mismatch()))
    assert r.risk.level == RiskLevel.HIGH
    assert any(
        i.source == EvidenceSource.BIOMETRICS and i.polarity == Polarity.CONCERN
        and i.field == "identity"
        for i in r.fused_evidence.items
    )
    # Single-engine high concern -> enhanced verification (not escalation).
    assert r.recommendation.action == RecommendedAction.ENHANCED_VERIFICATION
    # The identity/document-consistency mismatch is called out.
    assert any(
        c.type == CorrelationType.CONTRADICTION for c in r.fused_evidence.correlations
    )


def test_scenario5_mixed_evidence_match_does_not_erase_anomaly():
    r = _evaluate(_response(_rules_pass(), _forensics_photo_anomaly(), _bio_match()))
    # The forensic photo anomaly survives despite the biometric MATCH.
    assert any(
        i.source == EvidenceSource.FORENSICS and i.polarity == Polarity.CONCERN
        for i in r.fused_evidence.items
    )
    assert r.risk.score > 0
    assert r.recommendation.action == RecommendedAction.MANUAL_REVIEW
    # Recorded as mixed evidence (contradiction), not collapsed away.
    assert any(
        c.type == CorrelationType.CONTRADICTION for c in r.fused_evidence.correlations
    )


def test_scenario6_llm_unavailable_still_produces_full_result():
    # Default engine uses the offline Null provider -> LLM UNAVAILABLE.
    r = _evaluate(_response(_rules_dob_mismatch(), _forensics_dob_anomaly(), _bio_match()))
    assert r.llm.status == LLMStatus.UNAVAILABLE
    assert r.llm.error
    # Deterministic analysis is complete and authoritative.
    assert r.risk.level == RiskLevel.HIGH
    assert r.recommendation.action
    assert r.explanation.reasoning


def test_score_is_not_presented_as_fraud_probability():
    r = _evaluate(_response(_rules_dob_mismatch(), _forensics_dob_anomaly(), _bio_match()))
    assert r.risk.score_kind == "decision_support"
    assert "not a calibrated probability" in r.risk.disclaimer.lower()


def test_corroborated_pair_outweighs_three_low_warnings():
    # 1 HIGH rules + 1 HIGH forensics (same region) vs 3 LOW warnings.
    corroborated = _evaluate(
        _response(_rules_dob_mismatch(), _forensics_dob_anomaly(), _bio_match())
    )
    low_warnings = RuleResults.from_findings(
        [
            RuleFinding.make(
                f"LOW_{i}", "format", RuleStatus.WARNING, RuleSeverity.LOW,
                "Minor formatting note.", field=None,
            )
            for i in range(3)
        ]
    )
    three_low = _evaluate(_response(low_warnings, _forensics_clean(), _bio_match()))
    assert corroborated.risk.score > three_low.risk.score


def test_unavailable_forensics_is_not_treated_as_clean():
    resp = _response(_rules_pass(), ForensicResults.unavailable("engine down"), _bio_match())
    r = _evaluate(resp)
    assert EvidenceSource.FORENSICS in r.fused_evidence.sources_unavailable
    assert "unavailable" in r.risk.rationale.lower() or "unknown" in r.risk.rationale.lower()


# ---------------------------------------------------------------------------
# LLM evaluation — stub providers (no real API call)
# ---------------------------------------------------------------------------
class _StubProvider(LLMProvider):
    """Returns a canned completion and records the prompt it was given."""

    model = "stub-model"

    def __init__(self, response_text: str):
        self._text = response_text
        self.last_system = None
        self.last_user = None

    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self._text


class _RaisingProvider(LLMProvider):
    model = "raising-model"

    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str) -> str:
        raise LLMProviderError("simulated network failure")


def _fused_and_risk(resp):
    from app.intelligence.fusion import fuse
    from app.intelligence.risk import assess_risk

    fused = fuse(resp)
    return fused, assess_risk(fused)


def test_llm_receives_grounded_engine_json():
    resp = _response(_rules_dob_mismatch(), _forensics_dob_anomaly(), _bio_match())
    fused, risk = _fused_and_risk(resp)
    stub = _StubProvider(json.dumps({"summary": "ok", "key_findings": []}))
    LLMEvaluator(stub).evaluate(resp, fused, risk)
    # The engine JSON (not vague prose) is what the LLM is grounded on...
    assert "engine_outputs" in stub.last_user
    assert "CONSISTENCY_DOB" in stub.last_user
    # ...and the system prompt forbids inventing evidence / probabilities.
    assert "Do NOT invent" in stub.last_system
    assert "probabilit" in stub.last_system.lower()


def test_llm_structured_output_parsed_and_hallucinations_dropped():
    resp = _response(_rules_dob_mismatch(), _forensics_clean(), _bio_match())
    fused, risk = _fused_and_risk(resp)
    payload = {
        "summary": "DOB mismatch is the key concern.",
        "key_findings": [
            {"source": "rules", "finding": "DOB mismatch", "importance": "HIGH"},
            # Hallucinated source -> must be dropped by the guard.
            {"source": "government_database", "finding": "flagged", "importance": "HIGH"},
        ],
        "corroboration": [],
        "contradictions": [],
        "uncertainties": ["Reference face quality unknown"],
        "suggested_action": "MANUAL_REVIEW",
        "agreement_with_risk": "AGREE",
    }
    ev = LLMEvaluator(_StubProvider(json.dumps(payload))).evaluate(resp, fused, risk)
    assert ev.status == LLMStatus.COMPLETED
    assert ev.model == "stub-model"
    sources = {kf.source for kf in ev.key_findings}
    assert sources == {"rules"}  # invented source removed
    assert ev.suggested_action == "MANUAL_REVIEW"
    assert ev.agreement_with_risk == "AGREE"


def test_llm_bad_json_degrades_gracefully():
    resp = _response(_rules_pass(), _forensics_clean(), _bio_match())
    fused, risk = _fused_and_risk(resp)
    ev = LLMEvaluator(_StubProvider("I think this looks fine, no JSON here")).evaluate(
        resp, fused, risk
    )
    assert ev.status == LLMStatus.UNAVAILABLE
    assert ev.error


def test_llm_provider_error_degrades_gracefully():
    resp = _response(_rules_pass(), _forensics_clean(), _bio_match())
    fused, risk = _fused_and_risk(resp)
    ev = LLMEvaluator(_RaisingProvider()).evaluate(resp, fused, risk)
    assert ev.status == LLMStatus.UNAVAILABLE
    assert "simulated network failure" in ev.error


def test_llm_context_excludes_raw_images_and_ocr_words():
    resp = _response(_rules_pass(), _forensics_clean(), _bio_match())
    from app.intelligence.explanation import build_llm_context

    fused, risk = _fused_and_risk(resp)
    ctx = build_llm_context(resp, fused, risk)
    blob = json.dumps(ctx)
    # No raw pixels / no bulky OCR word list / no raw quality signals.
    assert "words" not in ctx["document"]
    assert "quality_signals" not in blob
