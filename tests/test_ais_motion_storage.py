"""Provenance survives storage and legacy positions remain unknown-time."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import inspect, text

from core.ais.messages import AisPositionReport, VesselStatic
from core.db.repository import PositionRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox
from services.ais_ingest.consumer import write_batch

NOW = dt.datetime(2026, 10, 29, 12, tzinfo=dt.UTC)


def test_native_provenance_round_trips_and_static_does_not_freshen_position():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    sessions = make_session_factory(engine)
    observed = NOW - dt.timedelta(seconds=90)
    report = AisPositionReport(
        mmsi=422123456,
        ts=observed,
        lon=56.5,
        lat=26.5,
        source="openwaters",
        sog_kn=12,
        cog_deg=90,
        timestamp_basis="provider",
        received_at=NOW,
        provider_source="aishub",
        provider_reencoded=True,
    )
    with session_scope(sessions) as session:
        write_batch(session, [report], [], now=NOW)
        write_batch(
            session,
            [],
            [VesselStatic(mmsi=report.mmsi, name="NEW NAME")],
            now=NOW + dt.timedelta(seconds=30),
        )
        fixes = PositionRepository(session).latest_in_bbox(BBox(56, 26, 57, 27), now=NOW)
        assert len(fixes) == 1
        fix = fixes[0]
        assert fix.ts == observed
        assert fix.received_at == NOW
        assert fix.timestamp_basis == "provider"
        assert fix.provider_source == "aishub"
        assert fix.provider_reencoded is True


def test_nullable_migration_preserves_old_positions_without_inventing_provenance():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO vessels (mmsi,sanctioned) VALUES (422123456,0)"))
        conn.execute(
            text(
                "INSERT INTO ais_positions (mmsi,ts,lon,lat,source) VALUES (422123456,'2026-10-29 11:59:00',56.5,26.5,'aisstream')"
            )
        )
        for column in ("timestamp_basis", "received_at", "provider_source", "provider_reencoded"):
            conn.execute(text(f"ALTER TABLE ais_positions DROP COLUMN {column}"))
    create_all(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("ais_positions")}
    assert {"timestamp_basis", "received_at", "provider_source", "provider_reencoded"} <= columns
    with session_scope(make_session_factory(engine)) as session:
        fix = PositionRepository(session).latest_in_bbox(BBox(56, 26, 57, 27), now=NOW)[0]
        assert fix.timestamp_basis is None and fix.received_at is None
        assert fix.provider_source is None and fix.provider_reencoded is None
