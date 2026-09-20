"""The chain — a self-contained, tamper-evident blockchain.

Why a blockchain here
---------------------
This is a *forensics* system, so the valuable property is **tamper-evidence and
non-repudiable chain of custody**: proving that a record existed at a point in
time and that nothing in the history has been altered. A blockchain is exactly
the right data structure for that — a chain of blocks, each committing to the
previous block's hash, sealed by proof of work.

This chain is *permissioned and single-node* (it runs inside the app, no
wallets / gas / network), which keeps demos reliable while remaining a genuine
blockchain: genesis block, block height, Merkle root over transactions, a
proof-of-work nonce, and full hash linkage.

What goes on-chain
------------------
Only **fingerprints** (SHA-256 hashes) and non-identifying metadata — never PII,
passport numbers, images or biometrics. The confidential data stays off-chain,
encrypted at rest and *deletable*; the chain keeps a permanent, tamper-evident
proof that a record with a given fingerprint was processed at a given time. When
the off-chain data is erased for a DPDP / GDPR request, the on-chain hash
becomes a harmless orphan and the chain still verifies.

Mechanics
---------
* transaction hash = SHA256( canonical(tx fields) )              (a Merkle leaf)
* merkle_root      = Merkle tree over the block's transaction hashes
* block hash       = SHA256( canonical(header) ) with ``difficulty`` leading
  zeros, found by incrementing ``nonce`` (proof of work)
* each block's ``prev_hash`` = the previous block's ``hash``; genesis uses 64
  zeros.

Difficulty is intentionally small so mining is sub-millisecond (demo-safe) while
still showing a real nonce search. :func:`verify_chain` recomputes every Merkle
root and every block hash and checks all linkage.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Block, Transaction

# prev_hash of the genesis block.
GENESIS_PREV_HASH = "0" * 64
# Proof-of-work target: block hash must start with this many hex zeros. Kept low
# so mining is effectively instant (a real nonce search, but demo-safe).
DIFFICULTY = 4


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace.

    Hashes are only reproducible if serialization is byte-stable, so every
    producer and verifier goes through here.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --- hashing primitives -----------------------------------------------------

def tx_hash(
    *,
    tx_index: int,
    timestamp: datetime,
    actor: Optional[str],
    action: str,
    subject_type: str,
    subject_id: Optional[str],
    payload: dict,
) -> str:
    """SHA-256 over a transaction's canonical contents (a Merkle leaf)."""
    return _sha256(
        _canonical(
            {
                "tx_index": tx_index,
                "timestamp": timestamp.isoformat(),
                "actor": actor,
                "action": action,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "payload": payload,
            }
        )
    )


def merkle_root(leaf_hashes: list[str]) -> str:
    """Merkle root of the transaction hashes.

    Empty block -> hash of the empty string. Odd layers duplicate the last node
    (Bitcoin-style), so the root commits to every transaction.
    """
    if not leaf_hashes:
        return _sha256("")
    layer = list(leaf_hashes)
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [_sha256(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def _block_header(
    *,
    index: int,
    timestamp: datetime,
    prev_hash: str,
    merkle: str,
    difficulty: int,
    nonce: int,
) -> str:
    return _canonical(
        {
            "index": index,
            "timestamp": timestamp.isoformat(),
            "prev_hash": prev_hash,
            "merkle_root": merkle,
            "difficulty": difficulty,
            "nonce": nonce,
        }
    )


def _mine(
    *,
    index: int,
    timestamp: datetime,
    prev_hash: str,
    merkle: str,
    difficulty: int,
) -> tuple[int, str]:
    """Proof of work: find a nonce whose block hash has ``difficulty`` zeros."""
    target = "0" * difficulty
    nonce = 0
    while True:
        h = _sha256(
            _block_header(
                index=index,
                timestamp=timestamp,
                prev_hash=prev_hash,
                merkle=merkle,
                difficulty=difficulty,
                nonce=nonce,
            )
        )
        if h.startswith(target):
            return nonce, h
        nonce += 1


# --- chain access -----------------------------------------------------------

def _head(db: Session) -> Optional[Block]:
    """The current top block (highest index), or None on an empty chain."""
    return db.scalars(select(Block).order_by(Block.index.desc()).limit(1)).first()


def _ensure_genesis(db: Session) -> Block:
    """Return the genesis block, mining it if the chain is empty."""
    head = _head(db)
    if head is not None:
        # Walk back to block 0 (genesis is always index 0).
        return db.scalars(select(Block).where(Block.index == 0)).first()
    return _seal_block(db, transactions=[], miner="system", commit=False)


def _seal_block(
    db: Session,
    *,
    transactions: list[dict],
    miner: Optional[str],
    commit: bool,
) -> Block:
    """Build, mine and persist the next block containing ``transactions``.

    Each transaction dict has: action, subject_type, subject_id, actor, payload.
    """
    head = _head(db)
    index = (head.index + 1) if head else 0
    prev_hash = head.hash if head else GENESIS_PREV_HASH
    # Naive-UTC so the value round-trips byte-identically on SQLite (a tz-aware
    # value reads back naive and its isoformat would no longer match the hash).
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    tx_rows: list[Transaction] = []
    leaf_hashes: list[str] = []
    for i, t in enumerate(transactions):
        payload = t.get("payload") or {}
        h = tx_hash(
            tx_index=i,
            timestamp=timestamp,
            actor=t.get("actor"),
            action=t["action"],
            subject_type=t["subject_type"],
            subject_id=t.get("subject_id"),
            payload=payload,
        )
        leaf_hashes.append(h)
        tx_rows.append(
            Transaction(
                tx_index=i,
                timestamp=timestamp,
                actor=t.get("actor"),
                action=t["action"],
                subject_type=t["subject_type"],
                subject_id=t.get("subject_id"),
                payload=payload,
                tx_hash=h,
            )
        )

    merkle = merkle_root(leaf_hashes)
    nonce, block_hash = _mine(
        index=index,
        timestamp=timestamp,
        prev_hash=prev_hash,
        merkle=merkle,
        difficulty=DIFFICULTY,
    )

    block = Block(
        index=index,
        timestamp=timestamp,
        miner=miner,
        prev_hash=prev_hash,
        merkle_root=merkle,
        difficulty=DIFFICULTY,
        nonce=nonce,
        hash=block_hash,
        transactions=tx_rows,
    )
    db.add(block)
    db.flush()  # surface index/hash uniqueness collisions immediately
    if commit:
        db.commit()
        db.refresh(block)
    return block


# --- public API -------------------------------------------------------------

def add_block(
    db: Session,
    transactions: list[dict],
    *,
    miner: Optional[str] = None,
    commit: bool = True,
) -> Block:
    """Mine a new block committing ``transactions`` onto the chain.

    Auto-creates the genesis block first if the chain is empty. Pass
    ``commit=False`` to enlist this in a caller's surrounding transaction (e.g.
    write the analysis and its block atomically).
    """
    _ensure_genesis(db)
    return _seal_block(db, transactions=transactions, miner=miner, commit=commit)


def verify_chain(db: Session) -> dict:
    """Recompute the whole chain and confirm it is intact.

    Returns::

        {
          "ok": bool,
          "blocks": int,           # blocks checked
          "transactions": int,     # total transactions
          "head": str | None,      # hash of the top block (chain fingerprint)
          "difficulty": int,
          "broken_at": int | None, # index of the first bad block
          "reason": str | None,
        }

    Detects: non-contiguous ``index`` (a block was deleted), a broken
    ``prev_hash`` link (reorder / insertion), a Merkle root that no longer
    matches the block's transactions (a transaction was edited), an invalid
    proof of work, and any recomputed block hash that no longer matches.
    """
    blocks = list(db.scalars(select(Block).order_by(Block.index.asc())).all())
    total_txns = sum(len(b.transactions) for b in blocks)

    def report(ok, broken_at=None, reason=None):
        return {
            "ok": ok,
            "blocks": len(blocks),
            "transactions": total_txns,
            "head": blocks[-1].hash if blocks else None,
            "difficulty": DIFFICULTY,
            "broken_at": broken_at,
            "reason": reason,
        }

    prev_hash = GENESIS_PREV_HASH
    for expected_index, block in enumerate(blocks):
        if block.index != expected_index:
            return report(
                False,
                broken_at=block.index,
                reason=(
                    f"non-contiguous block index: expected {expected_index}, "
                    f"got {block.index} (a block was deleted or reordered)"
                ),
            )
        if block.prev_hash != prev_hash:
            return report(
                False,
                broken_at=block.index,
                reason="prev_hash does not match the previous block's hash",
            )

        # Recompute the Merkle root from the block's transactions.
        leaves = []
        for tx in sorted(block.transactions, key=lambda t: t.tx_index):
            recomputed_leaf = tx_hash(
                tx_index=tx.tx_index,
                timestamp=tx.timestamp,
                actor=tx.actor,
                action=tx.action,
                subject_type=tx.subject_type,
                subject_id=tx.subject_id,
                payload=tx.payload,
            )
            if recomputed_leaf != tx.tx_hash:
                return report(
                    False,
                    broken_at=block.index,
                    reason="a transaction hash does not match its contents "
                    "(transaction edited)",
                )
            leaves.append(tx.tx_hash)
        if merkle_root(leaves) != block.merkle_root:
            return report(
                False,
                broken_at=block.index,
                reason="merkle_root does not match the block's transactions",
            )

        # Recompute the block hash and check the proof of work.
        recomputed = _sha256(
            _block_header(
                index=block.index,
                timestamp=block.timestamp,
                prev_hash=block.prev_hash,
                merkle=block.merkle_root,
                difficulty=block.difficulty,
                nonce=block.nonce,
            )
        )
        if recomputed != block.hash:
            return report(
                False,
                broken_at=block.index,
                reason="stored block hash does not match recomputed hash "
                "(block edited)",
            )
        if not block.hash.startswith("0" * block.difficulty):
            return report(
                False,
                broken_at=block.index,
                reason="block hash does not satisfy its proof-of-work difficulty",
            )

        prev_hash = block.hash

    return report(True)


def chain_stats(db: Session) -> dict:
    """Lightweight header stats for a dashboard / explorer banner."""
    head = _head(db)
    n_blocks = int(db.scalar(select(func.count(Block.id))) or 0)
    n_txns = int(db.scalar(select(func.count(Transaction.id))) or 0)
    return {
        "height": head.index if head else -1,
        "blocks": n_blocks,
        "transactions": n_txns,
        "head": head.hash if head else None,
        "difficulty": DIFFICULTY,
    }


# --- demonstration helpers --------------------------------------------------
# These exist ONLY to show the tamper-evidence working in a live demo. They
# mutate a committed transaction (an attack the app itself never performs) so a
# presenter can run verify_chain and watch it fail, then restore. Gated behind
# APP_DEMO_MODE at the route layer so they can never run in production.

# block_index -> [{tx_id, payload}] originals, kept in-process for the session.
_TAMPER_BACKUP: dict[int, list[dict]] = {}


def demo_tamper(db: Session, index: Optional[int] = None) -> dict:
    """Simulate an attacker altering sealed evidence (a 'risk downgrade').

    Edits the target block's first transaction payload in place *without*
    updating its hash, so :func:`verify_chain` will detect it. The original is
    backed up so :func:`demo_restore` can undo it exactly.
    """
    if index is None:
        head = _head(db)
        if head is None or head.index == 0:
            return {
                "tampered": False,
                "reason": "No non-genesis block yet — run a screening first.",
            }
        index = head.index

    block = db.scalars(select(Block).where(Block.index == index)).first()
    if block is None:
        return {"tampered": False, "reason": f"No block at height {index}."}
    if block.index == 0 or not block.transactions:
        return {
            "tampered": False,
            "reason": "The genesis block has no transactions to tamper.",
        }

    tx = sorted(block.transactions, key=lambda t: t.tx_index)[0]
    original = dict(tx.payload or {})
    _TAMPER_BACKUP.setdefault(index, []).append(
        {"tx_id": tx.id, "payload": original}
    )

    tampered = dict(original)
    before = tampered.get("risk_level")
    tampered["risk_level"] = "LOW"  # the "attack": downgrade a flagged result
    tampered["_tampered"] = True
    tx.payload = tampered  # new dict object -> SQLAlchemy marks it dirty
    db.commit()

    return {
        "tampered": True,
        "block_index": index,
        "field": "risk_level",
        "before": before,
        "after": "LOW",
    }


def demo_restore(db: Session) -> dict:
    """Undo every demo tamper, restoring original payloads exactly."""
    restored = 0
    for entries in _TAMPER_BACKUP.values():
        for e in entries:
            tx = db.get(Transaction, e["tx_id"])
            if tx is not None:
                tx.payload = e["payload"]
                restored += 1
    _TAMPER_BACKUP.clear()
    db.commit()
    return {"restored": restored}


def result_fingerprint(full_result: dict) -> str:
    """SHA-256 of a screening response's canonical JSON.

    This is the fingerprint committed on-chain. Recompute it from the stored
    ``Analysis.full_result`` to prove the record has not been altered since it
    was screened.
    """
    return _sha256(_canonical(full_result))
