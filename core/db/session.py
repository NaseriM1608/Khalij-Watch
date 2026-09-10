"""Engine and session management.

SQLite is the current backend by decision; PostgreSQL + PostGIS is the intended
destination. Everything dialect-specific is confined to this module and
``core.db.spatial``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from core.db.models import Base

DEFAULT_SQLITE_PATH = Path("data") / "khalij.db"

# SQLite defaults are tuned for safety on a single embedded writer. This
# workload is a continuous AIS ingest plus batch SAR jobs, so WAL and a
# relaxed sync are needed to avoid the writer blocking readers.
_SQLITE_PRAGMAS = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "foreign_keys": "ON",
    "busy_timeout": "5000",
    # Negative cache_size is KiB, so this is a 64 MB page cache. Deliberately
    # modest: the host has 10 GB of RAM in total.
    "cache_size": "-64000",
}


def database_url() -> str:
    """Resolve the database URL, defaulting to a local SQLite file."""
    configured = os.environ.get("DATABASE_URL")
    if configured:
        return configured
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+pysqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


def _apply_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        for pragma, value in _SQLITE_PRAGMAS.items():
            cursor.execute(f"PRAGMA {pragma}={value}")
        cursor.close()


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Build an engine with dialect-appropriate settings."""
    resolved = url or database_url()
    is_sqlite = resolved.startswith("sqlite")
    engine = create_engine(
        resolved,
        echo=echo,
        future=True,
        # A file-backed SQLite database is used from several threads (ingest
        # worker plus API), so the default same-thread check must be relaxed.
        connect_args={"check_same_thread": False} if is_sqlite else {},
    )
    if is_sqlite:
        _apply_sqlite_pragmas(engine)
    return engine


def create_all(engine: Engine) -> None:
    """Create every table plus the SQLite spatial index, if applicable."""
    Base.metadata.create_all(engine)
    if engine.dialect.name == "sqlite":
        from core.db.spatial import create_rtree_tables

        create_rtree_tables(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
