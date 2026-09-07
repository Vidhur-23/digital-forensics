"""Response models for the stored-analysis read endpoints.

These describe rows of the ``analyses`` table. The heavy ``full_result`` (the
complete original screening response) is included only on the single-record
endpoint, not on the list, to keep listings light.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class EvidenceFileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kind: str
    sha256: str
    content_type: Optional[str] = None
    size: Optional[int] = None


class AnalysisSummaryOut(BaseModel):
    """One row for the analyses list — the extracted, queryable columns."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    document_type: Optional[str] = None
    risk_level: Optional[str] = None
    risk_score: Optional[int] = None
    recommendation: Optional[str] = None
    concern_count: Optional[int] = None
    rules_has_failures: Optional[bool] = None
    forensic_status: Optional[str] = None
    forensic_flagged_count: Optional[int] = None
    biometric_status: Optional[str] = None
    biometric_similarity: Optional[float] = None
    llm_status: Optional[str] = None
    has_reference_face: bool = False


class AnalysisRecordOut(AnalysisSummaryOut):
    """A single stored analysis, including the full original response JSON."""

    image_width: Optional[int] = None
    image_height: Optional[int] = None
    evidence_files: List[EvidenceFileOut] = []
    # The complete ScreeningResponse as returned at analysis time — lets the
    # frontend re-render the whole result view from a saved record.
    full_result: Dict[str, Any] = {}
