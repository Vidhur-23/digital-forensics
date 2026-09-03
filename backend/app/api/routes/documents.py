"""Document screening route — Phase 1 (`POST /api/screen`).

Thin HTTP layer: validate the upload, delegate to the pipeline, translate
domain errors into clean HTTP responses. No OCR/CV logic lives here.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.schemas.document import ScreeningResponse
from app.config import settings
from app.dependencies import get_pipeline
from app.document.preprocessing import ImageDecodeError
from app.ocr.engine import OCRError
from app.pipeline.pipeline import ScreeningPipeline

router = APIRouter(tags=["screening"])


@router.post("/screen", response_model=ScreeningResponse)
async def screen_document(
    document: UploadFile = File(..., description="Document image (passport)"),
    pipeline: ScreeningPipeline = Depends(get_pipeline),
) -> ScreeningResponse:
    """Run the Phase 1 document + OCR pipeline on an uploaded image."""
    if document.content_type and document.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type '{document.content_type}'. "
                f"Allowed: {', '.join(settings.allowed_image_types)}."
            ),
        )

    data = await document.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file is too large.",
        )

    try:
        return pipeline.screen(data)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except OCRError as exc:
        # OCR backend unavailable (e.g. tesseract binary not installed).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
