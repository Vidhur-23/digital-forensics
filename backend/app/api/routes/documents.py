"""Document screening route — Phase 1 (`POST /api/screen`).

Thin HTTP layer: validate the upload, delegate to the pipeline, translate
domain errors into clean HTTP responses. No OCR/CV logic lives here.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.schemas.document import ScreeningResponse
from app.config import settings
from app.database.connection import get_session
from app.database.repository import persist_screening
from app.dependencies import get_pipeline
from app.document.preprocessing import ImageDecodeError
from app.ocr.engine import OCRError
from app.pipeline.pipeline import ScreeningPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["screening"])


def _validate_content_type(content_type: str | None) -> None:
    """Reject uploads whose declared content type is not an accepted image."""
    if content_type and content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type '{content_type}'. "
                f"Allowed: {', '.join(settings.allowed_image_types)}."
            ),
        )


@router.post("/screen", response_model=ScreeningResponse)
async def screen_document(
    document: UploadFile = File(..., description="Document image (passport)"),
    reference_face: UploadFile | None = File(
        None,
        description="Optional reference/live face image for Phase 4 biometric "
        "verification against the document photo.",
    ),
    pipeline: ScreeningPipeline = Depends(get_pipeline),
    db: Session = Depends(get_session),
) -> ScreeningResponse:
    """Run the full screening pipeline (Phases 1-4) on an uploaded image.

    An optional ``reference_face`` image enables Phase 4 biometric verification;
    without it, biometrics is reported as UNAVAILABLE and Phases 1-3 run as
    before.
    """
    _validate_content_type(document.content_type)

    data = await document.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file is too large.",
        )

    reference_data: bytes | None = None
    if reference_face is not None:
        _validate_content_type(reference_face.content_type)
        reference_data = await reference_face.read()
        if len(reference_data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Reference face image is too large.",
            )

    try:
        response = pipeline.screen(data, reference_data=reference_data)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except OCRError as exc:
        # OCR backend unavailable (e.g. tesseract binary not installed).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    # Persist the analysis (best-effort). A storage failure must not lose the
    # user's result — the analysis itself already succeeded — so we log and
    # return the response with analysis_id left unset.
    if settings.persist_analyses:
        try:
            record = persist_screening(
                db,
                response,
                document_bytes=data,
                document_content_type=document.content_type,
                reference_bytes=reference_data,
                reference_content_type=(
                    reference_face.content_type if reference_face else None
                ),
            )
            response.analysis_id = record.id
        except Exception:  # pragma: no cover - persistence is non-critical
            db.rollback()
            logger.exception("Failed to persist analysis; returning result anyway.")

    return response
