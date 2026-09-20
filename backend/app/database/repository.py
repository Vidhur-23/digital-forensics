"""Persistence for screening analyses.

Extracts a few queryable columns from the actual :class:`ScreeningResponse`,
stores the full response JSON losslessly, and writes uploaded images to disk
(recording only a path + SHA-256 in the DB). All reads/writes go through here so
the route and pipeline stay unaware of the ORM.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.document import ScreeningResponse
from app.audit import ledger
from app.config import settings
from app.database.models import Analysis, EvidenceFile

_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


def _store_file(kind: str, data: bytes, content_type: Optional[str]) -> dict:
    """Write bytes under the evidence dir as <sha256><ext>; return a pointer.

    Content-addressed, so re-uploading the same image does not duplicate it.
    """
    digest = hashlib.sha256(data).hexdigest()
    ext = _EXT.get((content_type or "").lower()) or (
        mimetypes.guess_extension(content_type or "") or ".bin"
    )
    base = Path(settings.evidence_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{digest}{ext}"
    if not path.exists():
        with open(path, "wb") as fh:
            fh.write(data)
    return {
        "kind": kind,
        "path": str(path),
        "sha256": digest,
        "content_type": content_type,
        "size": len(data),
    }


def _extract_columns(resp: ScreeningResponse) -> dict:
    """Pull the denormalised, queryable columns out of the response."""
    intel = resp.intelligence
    risk = intel.risk if intel else None
    rec = intel.recommendation if intel else None
    fused = intel.fused_evidence if intel else None
    llm = intel.llm if intel else None

    return {
        "document_type": resp.document_type,
        "document_type_confidence": resp.document_type_confidence,
        "image_width": resp.image.width if resp.image else None,
        "image_height": resp.image.height if resp.image else None,
        "risk_level": risk.level.value if risk else None,
        "risk_score": risk.score if risk else None,
        "recommendation": rec.action.value if rec else None,
        "concern_count": fused.concern_count if fused else None,
        "rules_has_failures": (
            resp.rules.summary.has_failures if resp.rules else None
        ),
        "forensic_status": (
            resp.forensics.status.value if resp.forensics else None
        ),
        "forensic_flagged_count": (
            resp.forensics.summary.flagged_count if resp.forensics else None
        ),
        "biometric_status": (
            resp.biometrics.status.value if resp.biometrics else None
        ),
        "biometric_similarity": (
            resp.biometrics.similarity if resp.biometrics else None
        ),
        "llm_status": llm.status.value if llm else None,
    }


def persist_screening(
    db: Session,
    response: ScreeningResponse,
    *,
    document_bytes: bytes,
    document_content_type: Optional[str] = None,
    reference_bytes: Optional[bytes] = None,
    reference_content_type: Optional[str] = None,
    actor: Optional[str] = None,
) -> Analysis:
    """Store one screening result + its evidence images. Returns the row."""
    files = [_store_file("document", document_bytes, document_content_type)]
    if reference_bytes:
        files.append(
            _store_file("reference_face", reference_bytes, reference_content_type)
        )

    full_result = response.model_dump(mode="json")
    record = Analysis(
        **_extract_columns(response),
        has_reference_face=bool(reference_bytes),
        full_result=full_result,
        evidence_files=[EvidenceFile(**f) for f in files],
    )
    db.add(record)
    db.flush()  # assign record.id before we fingerprint it for the chain

    # Mine a block committing this screening onto the chain, in the SAME
    # transaction, so the record and its block commit together. Fingerprints
    # only — no PII/biometrics ever go on-chain.
    ledger.add_block(
        db,
        [
            {
                "action": "screening.created",
                "subject_type": "analysis",
                "subject_id": record.id,
                "actor": actor,
                "payload": {
                    "result_sha256": ledger.result_fingerprint(full_result),
                    "evidence": [
                        {"kind": f["kind"], "sha256": f["sha256"]} for f in files
                    ],
                    "risk_level": record.risk_level,
                    "risk_score": record.risk_score,
                    "recommendation": record.recommendation,
                },
            }
        ],
        miner=actor,
        commit=False,
    )

    db.commit()
    db.refresh(record)
    return record


def list_analyses(db: Session, limit: int = 50, offset: int = 0) -> list[Analysis]:
    stmt = (
        select(Analysis)
        .order_by(Analysis.created_at.desc())
        .limit(max(1, min(limit, 200)))
        .offset(max(0, offset))
    )
    return list(db.scalars(stmt).all())


def get_analysis(db: Session, analysis_id: str) -> Optional[Analysis]:
    return db.get(Analysis, analysis_id)


# --- blockchain -------------------------------------------------------------

def list_blocks(db: Session, limit: int = 50, offset: int = 0) -> list:
    """Blocks, newest (highest) first, for a block explorer."""
    from app.database.models import Block  # local import: chain is optional

    stmt = (
        select(Block)
        .order_by(Block.index.desc())
        .limit(max(1, min(limit, 200)))
        .offset(max(0, offset))
    )
    return list(db.scalars(stmt).all())


def get_block(db: Session, index: int):
    """One block by height, or None."""
    from app.database.models import Block

    return db.scalars(select(Block).where(Block.index == index)).first()


def verify_chain(db: Session) -> dict:
    """Recompute the whole chain and report whether it is intact."""
    return ledger.verify_chain(db)


def chain_stats(db: Session) -> dict:
    """Chain header stats for a dashboard banner."""
    return ledger.chain_stats(db)


def demo_tamper(db: Session, index: Optional[int] = None) -> dict:
    """Demo-only: alter a sealed block so verification fails."""
    return ledger.demo_tamper(db, index)


def demo_restore(db: Session) -> dict:
    """Demo-only: undo any demo tampering."""
    return ledger.demo_restore(db)
