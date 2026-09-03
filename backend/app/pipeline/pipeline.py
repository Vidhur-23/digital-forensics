"""Phase 1 screening pipeline.

Orchestrates the flow without embedding any OCR/CV details in the API layer:

    bytes
      -> load_document        (decode + preprocess)
      -> OCREngine.extract     (words + bbox + confidence)
      -> detect_mrz            (MRZ region + raw fields)
      -> classify_document     (document type)
      -> FieldExtractor        (visual + MRZ field values)
      -> ScreeningResponse     (structured JSON)

The OCR engine is injected, so tests can supply a deterministic stub and a
later phase can swap the backend.
"""
from __future__ import annotations

from app.api.schemas.document import (
    FieldValue,
    ImageInfo,
    MRZFieldsOut,
    MRZInfo,
    OCRInfo,
    OCRWordOut,
    ScreeningResponse,
)
from app.document.classifier import classify_document
from app.document.preprocessing import load_document
from app.ocr.engine import OCREngine
from app.ocr.extractor import get_extractor
from app.ocr.mrz import detect_mrz


class ScreeningPipeline:
    def __init__(self, ocr_engine: OCREngine):
        self._ocr = ocr_engine

    def screen(self, data: bytes) -> ScreeningResponse:
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

        return ScreeningResponse(
            document_type=classification.document_type,
            document_type_confidence=classification.confidence,
            image=ImageInfo(width=doc.width, height=doc.height),
            fields=fields,
            mrz=mrz_out,
            ocr=ocr_out,
        )
