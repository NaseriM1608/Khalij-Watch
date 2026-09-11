"""Existing databases must pick up columns added to the models.

``create_all`` creates missing tables but never alters existing ones, so
without an additive step a column added to a model is absent from any
database created earlier, and the first write touching it fails.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from core.db.session import create_all, create_db_engine

# The detections table exactly as databases created before Phase 3 hold it.
LEGACY_DETECTIONS = """
CREATE TABLE detections (
    id INTEGER NOT NULL PRIMARY KEY,
    scene_id INTEGER NOT NULL,
    lon FLOAT NOT NULL,
    lat FLOAT NOT NULL,
    length_m FLOAT,
    width_m FLOAT,
    heading_deg FLOAT,
    confidence FLOAT NOT NULL,
    is_vessel_prob FLOAT,
    is_fishing_prob FLOAT,
    pred_class VARCHAR(24),
    class_probs JSON,
    chip_path VARCHAR(512),
    detector VARCHAR(64) NOT NULL,
    detector_version VARCHAR(32)
)
"""


def column_names(engine, table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def test_new_nullable_columns_are_added_to_an_existing_table():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(LEGACY_DETECTIONS))
        conn.execute(
            text(
                "INSERT INTO detections (id, scene_id, lon, lat, confidence, detector) "
                "VALUES (1, 1, 56.5, 25.2, 1.0, 'cfar')"
            )
        )

    create_all(engine)

    assert {"polarisation", "pixel_count", "peak_dn"} <= column_names(engine, "detections")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT lon FROM detections WHERE id = 1")).scalar() == 56.5


def test_create_all_is_idempotent():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")

    create_all(engine)
    create_all(engine)

    assert "polarisation" in column_names(engine, "detections")


def test_a_missing_required_column_needs_an_explicit_migration():
    """A NOT NULL column cannot be added to rows that already exist."""
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    without_detector = LEGACY_DETECTIONS.replace(",\n    detector VARCHAR(64) NOT NULL", "")
    with engine.begin() as conn:
        conn.execute(text(without_detector))

    with pytest.raises(RuntimeError, match="migration"):
        create_all(engine)
