"""Block-explorer / verification endpoints for the chain.

    GET /api/chain              -> chain header stats (height, head, difficulty)
    GET /api/chain/verify       -> recompute & validate the whole chain
    GET /api/chain/blocks       -> blocks, newest first (with transactions)
    GET /api/chain/blocks/{ix}  -> one block by height

The chain is append-only and mined by the screening pipeline; there is no write
endpoint here by design — blocks are only ever produced as a side effect of a
real action (e.g. a screening), never by hand.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.schemas.persistence import BlockOut, ChainStatsOut, ChainVerifyOut
from app.database import repository
from app.database.connection import get_session

router = APIRouter(prefix="/chain", tags=["chain"])


def _demo_enabled() -> bool:
    """Demo endpoints are on by default in dev, off if APP_DEMO_MODE is falsy."""
    return os.getenv("APP_DEMO_MODE", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _require_demo() -> None:
    if not _demo_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo endpoints are disabled (set APP_DEMO_MODE=true to enable).",
        )


@router.get("", response_model=ChainStatsOut)
def chain_stats(db: Session = Depends(get_session)) -> ChainStatsOut:
    """Header stats for a block-explorer banner."""
    return repository.chain_stats(db)


@router.get("/verify", response_model=ChainVerifyOut)
def verify_chain(db: Session = Depends(get_session)) -> ChainVerifyOut:
    """Recompute the whole chain. ``ok=false`` means it was tampered with."""
    return repository.verify_chain(db)


@router.get("/blocks", response_model=list[BlockOut])
def list_blocks(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_session),
) -> list[BlockOut]:
    """Blocks newest (highest) first, each with its transactions."""
    return repository.list_blocks(db, limit=limit, offset=offset)


@router.get("/blocks/{index}", response_model=BlockOut)
def get_block(index: int, db: Session = Depends(get_session)) -> BlockOut:
    """One block by height."""
    block = repository.get_block(db, index)
    if block is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No block at height {index}.",
        )
    return block


# --- demonstration endpoints (gated by APP_DEMO_MODE) -----------------------
# For live demos only: prove the chain catches tampering. Never enabled in prod.

@router.post("/demo/tamper")
def demo_tamper(
    index: int | None = Query(None, description="Block height to tamper; default = newest"),
    db: Session = Depends(get_session),
) -> dict:
    """Simulate an attacker altering a sealed block (a 'risk downgrade').

    After calling this, ``GET /api/chain/verify`` will report the chain broken.
    Undo with ``POST /api/chain/demo/restore``.
    """
    _require_demo()
    return repository.demo_tamper(db, index)


@router.post("/demo/restore")
def demo_restore(db: Session = Depends(get_session)) -> dict:
    """Undo any demo tampering, restoring the chain so verification passes."""
    _require_demo()
    return repository.demo_restore(db)
