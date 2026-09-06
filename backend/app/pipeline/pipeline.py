"""Phase 1 screening pipeline.

Orchestrates the flow without embedding any OCR/CV details in the API layer:

    bytes
      -> load_document        (decode + preprocess)
      -> OCREngine.extract     (words + bbox + confidence)
      -> detect_mrz            (MRZ region + raw fields)
      -> classify_document     (document type)
      -> FieldExtractor        (visual + MRZ field values)
      -> ScreeningResponse     (structured Phase 1 JSON)
      -> RulesEngine.evaluate  (Phase 2 deterministic rule findings)
      -> ForensicService       (Phase 3 forensic manipulation evidence)
      -> BiometricService      (Phase 4 face-verification evidence)

The OCR engine is injected, so tests can supply a deterministic stub and a
later phase can swap the backend. The Rules Engine (Phase 2) runs *after* the
Phase 1 structured result is assembled and consumes that result; it never
touches OCR/CV. The Forensic Service (Phase 3) runs on the ORIGINAL decoded
image. The Biometric Service (Phase 4) runs last, comparing the document face
against a separately supplied reference face image. Every later-phase service
is evidence-only (never a fraud verdict) and failure-isolated, so an error in
one does not destroy the earlier phases' results.
"""
from __future__ import annotations

from typing import Optional

from app.api.schemas.biometric import BiometricResults
from app.api.schemas.document import (
    FieldValue,
    ImageInfo,
    MRZFieldsOut,
    MRZInfo,
    OCRInfo,
    OCRWordOut,
    ScreeningResponse,
)
from app.api.schemas.evidence import ForensicResults
from app.biometrics.service import BiometricService, InsightFaceBiometricService
from app.document.classifier import classify_document
from app.document.preprocessing import decode_image, load_document
from app.forensics.engine import ForensicLayerService, ForensicService
from app.intelligence.engine import IntelligenceEngine
from app.ocr.engine import OCREngine
from app.ocr.extractor import get_extractor
from app.ocr.mrz import detect_mrz
from app.rules.engine import RulesEngine


class ScreeningPipeline:
    def __init__(
        self,
        ocr_engine: OCREngine,
        rules_engine: Optional[RulesEngine] = None,
        forensic_service: Optional[ForensicService] = None,
        biometric_service: Optional[BiometricService] = None,
        intelligence_engine: Optional[IntelligenceEngine] = None,
    ):
        self._ocr = ocr_engine
        # Phase 2 orchestrator. Deterministic and stateless, so a default
        # instance is fine; still injectable for testing/extension.
        self._rules = rules_engine or RulesEngine()
        # Phase 3 forensic adapter (bridges to the in-backend forensic_layer
        # project). Injectable so tests can stub it or exercise failure paths.
        self._forensics = forensic_service or ForensicLayerService()
        # Phase 4 biometric service (InsightFace-backed). Injectable so tests
        # can stub it or exercise failure paths.
        self._biometrics = biometric_service or InsightFaceBiometricService()
        # Phase 5 Intelligence Layer. Deterministic fusion/risk/recommendation
        # plus an advisory LLM (disabled by default). Consumes the Phase 2-4
        # outputs already on the response; never re-runs an engine. Injectable so
        # tests can supply a stub LLM provider.
        self._intelligence = intelligence_engine or IntelligenceEngine()

    def screen(
        self, data: bytes, reference_data: Optional[bytes] = None
    ) -> ScreeningResponse:
        # 1. Decode + preprocess (raises ImageDecodeError on bad input).
        doc = load_document(data)

        # 2. OCR on the enhanced image (bboxes are in original coordinates).
        ocr_result = self._ocr.extract(doc.ocr_image)

        # 3. MRZ detection/extraction (separate from visual fields).
        mrz = detect_mrz(ocr_result)

        # 4. Document type identification.
        classification = classify_document(ocr_result, mrz)

        # 5. Field extraction (type-specific; passport in Phase 1).
        extractor = get_extractor(classification.document_type)
        extracted = extractor.extract(ocr_result, mrz)

        # 6. Assemble structured response.
        fields = {
            name: FieldValue(
                value=f.value,
                confidence=f.confidence,
                bbox=f.bbox,
                source=f.source,
            )
            for name, f in extracted.items()
        }

        mrz_out = MRZInfo(
            detected=mrz.detected,
            format=mrz.format,
            text=mrz.text,
            bbox=mrz.bbox,
            fields=MRZFieldsOut(**mrz.fields.model_dump()),
        )

        ocr_out = OCRInfo(
            confidence=round(ocr_result.mean_confidence, 4),
            word_count=len(ocr_result.words),
            words=[
                OCRWordOut(text=w.text, confidence=w.confidence, bbox=w.bbox)
                for w in ocr_result.words
            ],
        )

        response = ScreeningResponse(
            document_type=classification.document_type,
            document_type_confidence=classification.confidence,
            image=ImageInfo(width=doc.width, height=doc.height),
            fields=fields,
            mrz=mrz_out,
            ocr=ocr_out,
        )

        # 7. Phase 2: deterministic Rules Engine over the structured result.
        response.rules = self._rules.evaluate(response)

        # 8. Phase 3: forensic manipulation evidence on the ORIGINAL image
        #    (BGR, original resolution — same coordinate space as OCR bboxes).
        #    Failure-isolated: a forensic error must not destroy the Phase 1/2
        #    result, and is reported as UNAVAILABLE (never a clean result).
        try:
            response.forensics = self._forensics.analyze(doc.original)
        except Exception as exc:  # last-resort guard around the adapter
            response.forensics = ForensicResults.unavailable(
                f"Forensic analysis failed: {exc!r}"
            )

        # 9. Phase 4: biometric face verification — document face (from the
        #    ORIGINAL BGR image) vs. a separately supplied reference face image.
        #    Failure-isolated and evidence-only. A missing/undecodable reference
        #    yields an UNAVAILABLE biometric result, never a MISMATCH, and never
        #    affects Phases 1-3.
        response.biometrics = self._run_biometrics(doc.original, reference_data)

        # 10. Phase 5: Intelligence Layer over the Phase 2-4 outputs now on the
        #     response. It re-runs no engine — it fuses the existing JSON, scores
        #     risk transparently, produces an advisory LLM explanation and an
        #     officer recommendation. Failure-isolated: any error leaves Phases
        #     1-4 intact (intelligence stays None); an LLM outage is captured
        #     inside the result as an UNAVAILABLE ``llm``, never a verdict.
        try:
            response.intelligence = self._intelligence.evaluate(response)
        except Exception:  # last-resort guard around the whole layer
            response.intelligence = None

        return response

    def _run_biometrics(
        self, document_image, reference_data: Optional[bytes]
    ) -> BiometricResults:
        reference_image = None
        if reference_data:
            try:
                reference_image = decode_image(reference_data)  # BGR, same as doc
            except Exception as exc:
                return BiometricResults.unavailable(
                    f"Reference image could not be decoded: {exc}",
                    message="The supplied reference face image is not a readable image.",
                )
        try:
            return self._biometrics.verify(document_image, reference_image)
        except Exception as exc:  # last-resort guard around the service
            return BiometricResults.unavailable(
                f"Biometric analysis failed: {exc!r}",
                message="Biometric analysis could not be completed.",
            )
