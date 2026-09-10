"""Tests for the data-access layer."""

from __future__ import annotations

import datetime as dt

import pytest

from core.ais.messages import AisPositionReport, VesselStatic
from core.db.repository import PositionRepository, VesselRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import GULF_BBOX, HORMUZ_BBOX, BBox

MMSI = 422123456
OTHER_MMSI = 470123456
NOW = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)


@pytest.fixture
def session():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as sess:
        yield sess


@pytest.fixture
def vessels(session):
    return VesselRepository(session)


@pytest.fixture
def positions(session):
    return PositionRepository(session)


def fix(mmsi=MMSI, lon=56.5, lat=26.5, minutes_ago=0, **kwargs) -> AisPositionReport:
    return AisPositionReport(
        mmsi=mmsi,
        ts=NOW - dt.timedelta(minutes=minutes_ago),
        lon=lon,
        lat=lat,
        source=kwargs.pop("source", "aisstream"),
        **kwargs,
    )


# --------------------------------------------------------------------------
# Vessel identity
# --------------------------------------------------------------------------


def test_sparse_static_messages_merge_rather_than_overwrite(vessels, session):
    """AIS type 24 arrives in two parts; neither may erase the other."""
    vessels.upsert_static(VesselStatic(mmsi=MMSI, name="EVER GIVEN"))  # Part A
    session.flush()
    vessels.upsert_static(VesselStatic(mmsi=MMSI, ship_type=70, callsign="H3RC"))  # Part B
    session.flush()

    vessel = vessels.get(MMSI)

    assert vessel.name == "EVER GIVEN"  # survived Part B
    assert vessel.callsign == "H3RC"
    assert vessel.coarse_class == "cargo_tanker"


def test_static_upsert_derives_coarse_class(vessels, session):
    vessels.upsert_static(VesselStatic(mmsi=MMSI, ship_type=30))
    session.flush()

    assert vessels.get(MMSI).coarse_class == "fishing"


def test_ensure_exists_is_idempotent(vessels, session):
    vessels.ensure_exists({MMSI, OTHER_MMSI}, seen_at=NOW)
    session.flush()
    vessels.ensure_exists({MMSI, OTHER_MMSI}, seen_at=NOW)
    session.flush()

    assert vessels.count() == 2


def test_ensure_exists_does_not_clobber_known_identity(vessels, session):
    vessels.upsert_static(VesselStatic(mmsi=MMSI, name="EVER GIVEN"))
    session.flush()
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()

    assert vessels.get(MMSI).name == "EVER GIVEN"


# --------------------------------------------------------------------------
# Position ingest
# --------------------------------------------------------------------------


def test_add_batch_writes_positions(vessels, positions, session):
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()

    written = positions.add_batch([fix(minutes_ago=2), fix(minutes_ago=1)])

    assert written == 2
    assert positions.count() == 2


def test_duplicate_fixes_are_ignored_not_fatal(vessels, positions, session):
    """The same fix arrives via multiple receivers; that is normal."""
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    duplicate = fix()

    positions.add_batch([duplicate])
    session.flush()
    positions.add_batch([duplicate])
    session.flush()

    assert positions.count() == 1


def test_empty_batch_is_a_no_op(positions):
    assert positions.add_batch([]) == 0


# --------------------------------------------------------------------------
# Live map query
# --------------------------------------------------------------------------


def test_latest_in_bbox_returns_one_row_per_vessel(vessels, positions, session):
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(minutes_ago=10, lat=26.1), fix(minutes_ago=1, lat=26.9)])
    session.flush()

    results = positions.latest_in_bbox(HORMUZ_BBOX, now=NOW)

    assert len(results) == 1
    assert results[0].lat == 26.9  # the newer fix


def test_latest_in_bbox_excludes_vessels_outside_the_box(vessels, positions, session):
    vessels.ensure_exists({MMSI, OTHER_MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(), fix(mmsi=OTHER_MMSI, lon=10.0, lat=50.0)])
    session.flush()

    results = positions.latest_in_bbox(HORMUZ_BBOX, now=NOW)

    assert [r.mmsi for r in results] == [MMSI]


def test_latest_in_bbox_applies_the_age_cutoff(vessels, positions, session):
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(minutes_ago=60 * 24)])  # a day old
    session.flush()

    assert positions.latest_in_bbox(HORMUZ_BBOX, now=NOW) == []
    stale = positions.latest_in_bbox(HORMUZ_BBOX, now=NOW, max_age=None)
    assert len(stale) == 1


def test_latest_in_bbox_joins_vessel_identity(vessels, positions, session):
    vessels.upsert_static(VesselStatic(mmsi=MMSI, name="EVER GIVEN", ship_type=70))
    session.flush()
    positions.add_batch([fix()])
    session.flush()

    result = positions.latest_in_bbox(HORMUZ_BBOX, now=NOW)[0]

    assert result.name == "EVER GIVEN"
    assert result.coarse_class == "cargo_tanker"


def test_fix_reports_its_own_age(vessels, positions, session):
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(minutes_ago=30)])
    session.flush()

    result = positions.latest_in_bbox(HORMUZ_BBOX, now=NOW)[0]

    # The UI must be able to state age explicitly rather than imply freshness.
    assert result.age_seconds(NOW) == pytest.approx(1800, abs=1)


def test_source_is_preserved_for_attribution(vessels, positions, session):
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(source="aiscast")])
    session.flush()

    assert positions.latest_in_bbox(HORMUZ_BBOX, now=NOW)[0].source == "aiscast"


def test_gulf_bbox_contains_hormuz_traffic(vessels, positions, session):
    """Guards the AOI bug found in Phase 0: Hormuz must sit inside the Gulf."""
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(lon=57.4, lat=26.5)])  # east of the strait
    session.flush()

    assert len(positions.latest_in_bbox(GULF_BBOX, now=NOW)) == 1


# --------------------------------------------------------------------------
# Tracks and retention
# --------------------------------------------------------------------------


def test_track_returns_positions_oldest_first(vessels, positions, session):
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(minutes_ago=1, lat=26.9), fix(minutes_ago=10, lat=26.1)])
    session.flush()

    track = positions.track(MMSI)

    assert [p.lat for p in track] == [26.1, 26.9]


def test_track_respects_a_time_window(vessels, positions, session):
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(minutes_ago=1), fix(minutes_ago=120, lat=26.2)])
    session.flush()

    track = positions.track(MMSI, since=NOW - dt.timedelta(minutes=30))

    assert len(track) == 1


def test_delete_older_than_trims_the_tail(vessels, positions, session):
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(minutes_ago=1), fix(minutes_ago=60 * 48, lat=26.2)])
    session.flush()

    removed = positions.delete_older_than(NOW - dt.timedelta(days=1))
    session.flush()

    assert removed == 1
    assert positions.count() == 1


def test_deleting_positions_keeps_the_spatial_index_consistent(vessels, positions, session):
    from sqlalchemy import text

    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(minutes_ago=60 * 48)])
    session.flush()

    positions.delete_older_than(NOW - dt.timedelta(days=1))
    session.flush()

    remaining = session.execute(text("SELECT count(*) FROM ais_positions_rtree")).scalar()
    assert remaining == 0


def test_bbox_rejects_nonsense_before_it_reaches_sql():
    with pytest.raises(ValueError):
        BBox(57.0, 23.5, 47.5, 30.5)


def test_non_utc_timestamps_are_normalised_before_storage(vessels, positions, session):
    """SQLite drops the UTC offset, so conversion must happen before the write.

    Tehran is UTC+3:30. Storing 15:30+03:30 without converting would record it
    as 15:30 UTC - three and a half hours in the future.
    """
    from sqlalchemy import text

    tehran = dt.timezone(dt.timedelta(hours=3, minutes=30))
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()

    positions.add_batch(
        [
            AisPositionReport(
                mmsi=MMSI,
                ts=dt.datetime(2026, 9, 10, 15, 30, tzinfo=tehran),
                lon=56.5,
                lat=26.5,
                source="test",
            )
        ]
    )
    session.flush()

    stored = session.execute(text("SELECT ts FROM ais_positions")).scalar()
    assert stored.startswith("2026-09-10 12:00:00")


def test_age_cutoff_matches_stored_timestamp_format(vessels, positions, session):
    """Regression: an ISO cutoff string silently matched nothing.

    SQLite compares these as text. "2026-09-10T06:00:00+00:00" sorts after
    every stored "2026-09-10 ..." value because "T" > " ", so the filter
    excluded everything instead of only stale rows.
    """
    vessels.ensure_exists({MMSI}, seen_at=NOW)
    session.flush()
    positions.add_batch([fix(minutes_ago=5)])
    session.flush()

    fresh = positions.latest_in_bbox(HORMUZ_BBOX, now=NOW, max_age=dt.timedelta(hours=6))

    assert len(fresh) == 1


def test_large_batch_is_chunked_under_the_bind_variable_limit(vessels, positions, session):
    """Regression: SQLite caps bound variables per statement (~32k).

    A 10-column insert therefore fails above ~3,200 rows with "too many SQL
    variables". A busy feed reaches that easily, so add_batch must chunk.
    """
    mmsis = {422_000_000 + i for i in range(1, 1201)}
    vessels.ensure_exists(mmsis, seen_at=NOW)
    session.flush()

    reports = [
        AisPositionReport(
            mmsi=mmsi,
            ts=NOW - dt.timedelta(seconds=offset),
            lon=56.5,
            lat=26.5,
            source="test",
        )
        for offset, mmsi in enumerate(sorted(mmsis))
        for _ in range(5)
    ]

    written = positions.add_batch(reports)
    session.flush()

    assert written == len(mmsis)  # 5 identical fixes per vessel dedupe to 1
    assert positions.count() == len(mmsis)


def test_ensure_exists_chunks_large_mmsi_sets(vessels, session):
    mmsis = {422_000_000 + i for i in range(1, 15_001)}

    vessels.ensure_exists(mmsis, seen_at=NOW)
    session.flush()

    assert vessels.count() == len(mmsis)
