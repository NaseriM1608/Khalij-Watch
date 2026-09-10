"""Create the database schema. Idempotent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect

from core.db.session import create_all, create_db_engine, database_url


def main() -> int:
    url = database_url()
    engine = create_db_engine(url)
    create_all(engine)

    inspector = inspect(engine)
    names = inspector.get_table_names()
    # SQLite backs each R*Tree with _node/_parent/_rowid shadow tables; those
    # are engine internals and only noise to an operator.
    shadow_suffixes = ("_rtree", "_rtree_node", "_rtree_parent", "_rtree_rowid")
    tables = sorted(t for t in names if not t.endswith(shadow_suffixes))
    spatial = sorted(t for t in names if t.endswith("_rtree"))

    print(f"database: {url}")
    print(f"tables  : {', '.join(tables)}")
    print(f"spatial : {', '.join(spatial) or 'none'}")
    print(f"dialect : {engine.dialect.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
