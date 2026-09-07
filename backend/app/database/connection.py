"""Database engine / session wiring.

Backend-agnostic: the same models run on SQLite (local dev, zero setup) and
Postgres (production) — only ``APP_DATABASE_URL`` changes. Engine construction
is kept here so the rest of the app depends on ``get_session`` / ``Base`` and
never on a specific driver.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# Declarative base shared by all ORM models.
Base = declarative_base()


def _engine_kwargs(url: str) -> dict:
    # SQLite needs check_same_thread off for FastAPI's threadpool; server DBs
    # benefit from pre-ping to drop dead pooled connections.
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


engine = create_engine(settings.database_url, **_engine_kwargs(settings.database_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they don't exist (dev/SQLite convenience).

    In production prefer Alembic migrations; this is guarded by
    ``settings.db_auto_create``.
    """
    # Import models so they are registered on ``Base.metadata`` before create.
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
