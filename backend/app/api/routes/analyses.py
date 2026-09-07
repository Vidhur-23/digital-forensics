"""Read endpoints for persisted analyses.

These serve stored screening results — they never run the pipeline (screening
stays on ``POST /api/screen``). This is retrieval only:

    GET /api/analyses        -> recent analyses (light summary rows)
    GET /api/analyses/{id}   -> one analysis, including the full response JSON
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.schemas.persistence import AnalysisRecordOut, AnalysisSummaryOut
from app.database import repository
from app.database.connection import get_session

router = APIRouter(tags=["analyses"])


@router.get("/analyses", response_model=list[AnalysisSummaryOut])
def list_analyses(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_session),
) -> list[AnalysisSummaryOut]:
    """Most-recent-first summaries for a review queue / dashboard."""
    return repository.list_analyses(db, limit=limit, offset=offset)


@router.get("/analyses/{analysis_id}", response_model=AnalysisRecordOut)
def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_session),
) -> AnalysisRecordOut:
    """One stored analysis, including the complete original response JSON."""
    record = repository.get_analysis(db, analysis_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis with id '{analysis_id}'.",
        )
    return record
