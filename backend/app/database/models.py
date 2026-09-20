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


class Block(Base):
    """One block in the tamper-evident chain (a real, permissioned blockchain).

    Blocks are mined in order: each commits to the previous block's ``hash``, to
    a Merkle root over its transactions, and to a proof-of-work ``nonce`` that
    makes ``hash`` start with ``difficulty`` zeros. Change, insert, delete or
    reorder anything and every subsequent block hash stops matching — the chain
    no longer verifies (:func:`app.audit.ledger.verify_chain`).

    Crucially, blocks carry only **fingerprints** (SHA-256 hashes) and
    non-identifying metadata via their transactions — never PII, images or
    biometrics. The confidential evidence stays off-chain, encrypted and
    *deletable* (DPDP / GDPR erasure); the chain keeps permanent, tamper-evident
    proof that a record with a given fingerprint existed at a given time.
    """

    __tablename__ = "blocks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    # Block height (0 = genesis). Unique so a concurrent double-mine collides at
    # the DB instead of silently forking the chain.
    index: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True
    )
    # Miner / operator identity that sealed the block (not the data subject).
    miner: Mapped[str | None] = mapped_column(String(128))

    # Consensus fields.
    prev_hash: Mapped[str] = mapped_column(String(64))
    merkle_root: Mapped[str] = mapped_column(String(64))
    difficulty: Mapped[int] = mapped_column(Integer, default=0)
    nonce: Mapped[int] = mapped_column(Integer, default=0)
    hash: Mapped[str] = mapped_column(String(64), unique=True)

    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="block",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Transaction.tx_index",
    )


class Transaction(Base):
    """A single record committed inside a :class:`Block`.

    Holds fingerprints + non-identifying detail only (never PII/biometrics). Its
    ``tx_hash`` is a leaf of the block's Merkle tree.
    """

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    block_id: Mapped[str] = mapped_column(
        ForeignKey("blocks.id", ondelete="CASCADE"), index=True
    )
    tx_index: Mapped[int] = mapped_column(Integer)  # position within the block
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Who / what / on what.
    actor: Mapped[str | None] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(64), index=True)
    subject_type: Mapped[str] = mapped_column(String(32), index=True)
    subject_id: Mapped[str | None] = mapped_column(String(128), index=True)

    # Fingerprints + non-identifying detail only. NEVER PII/biometrics.
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)

    # Merkle leaf: SHA-256 over this transaction's canonical contents.
    tx_hash: Mapped[str] = mapped_column(String(64), index=True)

    block: Mapped["Block"] = relationship(back_populates="transactions")
