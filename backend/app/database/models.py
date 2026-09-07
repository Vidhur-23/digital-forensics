"""ORM models for persisted analyses.

One :class:`Analysis` row per ``/api/screen`` call. It keeps a handful of
extracted, queryable columns (for dashboards / triage / filtering) plus the
COMPLETE screening response as JSON (``full_result``) so the whole UI can be
re-rendered losslessly from a stored record. Uploaded images are NOT stored in
the database — only a path + SHA-256 pointer in :class:`EvidenceFile`.

The extracted columns are denormalised copies of values that already live in the
response JSON; the response schemas remain the source of truth.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base

# Portable JSON: plain JSON on SQLite, indexable JSONB on Postgres.
JSONType = JSON().with_variant(JSONB, "postgresql")


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True
    )

    # --- document / image ---
    document_type: Mapped[str | None] = mapped_column(String(64), index=True)
    document_type_confidence: Mapped[float | None] = mapped_column(Float)
    image_width: Mapped[int | None] = mapped_column(Integer)
    image_height: Mapped[int | None] = mapped_column(Integer)
    has_reference_face: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- extracted, queryable outcome columns (denormalised from full_result) ---
    risk_level: Mapped[str | None] = mapped_column(String(16), index=True)
    risk_score: Mapped[int | None] = mapped_column(Integer, index=True)
    recommendation: Mapped[str | None] = mapped_column(String(32), index=True)
    concern_count: Mapped[int | None] = mapped_column(Integer)

    rules_has_failures: Mapped[bool | None] = mapped_column(Boolean)
    forensic_status: Mapped[str | None] = mapped_column(String(16))
    forensic_flagged_count: Mapped[int | None] = mapped_column(Integer)
    biometric_status: Mapped[str | None] = mapped_column(String(24), index=True)
    biometric_similarity: Mapped[float | None] = mapped_column(Float)
    llm_status: Mapped[str | None] = mapped_column(String(16))

    # --- lossless full response ---
    full_result: Mapped[dict] = mapped_column(JSONType)

    evidence_files: Mapped[list["EvidenceFile"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class EvidenceFile(Base):
    """Pointer to an uploaded image on disk (never the raw pixels in the DB)."""

    __tablename__ = "evidence_files"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(24))  # "document" | "reference_face"
    path: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    content_type: Mapped[str | None] = mapped_column(String(64))
    size: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    analysis: Mapped["Analysis"] = relationship(back_populates="evidence_files")
