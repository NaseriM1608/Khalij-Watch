"""Tests for the database layer and its SQLite spatial index.

The R*Tree companion tables are maintained by triggers, so these tests prove
the index cannot silently drift from the rows it indexes.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

from core.db.models import AisPosition, Vessel
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import (
    GULF_BBOX,
    HORMUZ_BBOX,
    BBox,
    bbox_filter_sql,
    bbox_params,
)


@pytest.fixture
def engine():
    # A file-backed temp DB would also work, but in-memory keeps tests fast and
    # forces the schema to be created from scratch every time.
    eng = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(eng)
    return eng


@pytest.fixture
def session_factory(engine):
    return make_session_factory(engine)


def _add_vessel(session, mmsi: int = 422000001) -> Vessel:
    vessel = Vessel(mmsi=mmsi, name="TEST VESSEL")
    session.add(vessel)
    session.flush()
    return vessel


def _add_position(session, mmsi: int, lon: float, lat: float) -> AisPosition:
    position = AisPosition(
        ts=dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC),
        mmsi=mmsi,
        lon=lon,
        lat=lat,
        source="test",
    )
    session.add(position)
    session.flush()
    return position


# --------------------------------------------------------------------------
# BBox
# --------------------------------------------------------------------------


def test_bbox_parse_reads_env_style_spec():
    bbox = BBox.parse("47.5,23.5,57.0,30.5")

    assert (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat) == (47.5, 23.5, 57.0, 30.5)


@pytest.mark.parametrize(
    "spec",
    ["47.5,23.5,57.0", "47.5,23.5,57.0,30.5,1", "", "a,b,c,d"],
)
def test_bbox_parse_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        BBox.parse(spec)


def test_bbox_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="min_lon"):
        BBox(57.0, 23.5, 47.5, 30.5)
    with pytest.raises(ValueError, match="min_lat"):
        BBox(47.5, 30.5, 57.0, 23.5)


def test_bbox_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError, match="latitude"):
        BBox(47.5, -91.0, 57.0, 30.5)
    with pytest.raises(ValueError, match="longitude"):
        BBox(-181.0, 23.5, 57.0, 30.5)


def test_bbox_contains_is_inclusive_at_edges():
    assert GULF_BBOX.contains(52.0, 27.0)
    assert GULF_BBOX.contains(47.5, 23.5)
    assert not GULF_BBOX.contains(60.0, 27.0)


def test_aisstream_bbox_puts_latitude_first():
    """aisstream.io orders coordinates lat-then-lon, unlike everything else."""
    payload = HORMUZ_BBOX.as_aisstream()

    assert payload == [[[24.5, 54.0], [27.5, 57.5]]]
    # Guard the specific mistake: first value must be the latitude.
    assert payload[0][0][0] == HORMUZ_BBOX.min_lat


def test_hormuz_sits_inside_the_gulf_aoi():
    assert GULF_BBOX.contains(HORMUZ_BBOX.min_lon, HORMUZ_BBOX.min_lat)
    assert GULF_BBOX.contains(HORMUZ_BBOX.max_lon, HORMUZ_BBOX.max_lat)


# --------------------------------------------------------------------------
# Spatial index
# --------------------------------------------------------------------------


def test_rtree_finds_position_inside_bbox(session_factory):
    with session_scope(session_factory) as session:
        _add_vessel(session)
        inside = _add_position(session, 422000001, lon=56.5, lat=26.5)
        _add_position(session, 422000001, lon=10.0, lat=50.0)  # North Sea

        rows = (
            session.execute(text(bbox_filter_sql("ais_positions")), bbox_params(HORMUZ_BBOX))
            .scalars()
            .all()
        )

        assert rows == [inside.id]


def test_rtree_is_updated_when_a_row_moves(session_factory):
    with session_scope(session_factory) as session:
        _add_vessel(session)
        position = _add_position(session, 422000001, lon=56.5, lat=26.5)

        position.lon = 10.0
        position.lat = 50.0
        session.flush()

        rows = (
            session.execute(text(bbox_filter_sql("ais_positions")), bbox_params(HORMUZ_BBOX))
            .scalars()
            .all()
        )

        assert rows == []


def test_rtree_drops_entries_for_deleted_rows(session_factory):
    with session_scope(session_factory) as session:
        _add_vessel(session)
        position = _add_position(session, 422000001, lon=56.5, lat=26.5)
        session.delete(position)
        session.flush()

        count = session.execute(text("SELECT count(*) FROM ais_positions_rtree")).scalar()

        assert count == 0


def test_bbox_filter_sql_rejects_unindexed_table():
    with pytest.raises(ValueError, match="no spatial index"):
        bbox_filter_sql("vessels")


# --------------------------------------------------------------------------
# Session handling
# --------------------------------------------------------------------------


def test_session_scope_rolls_back_on_error(session_factory):
    class BoomError(Exception):
        pass

    with pytest.raises(BoomError), session_scope(session_factory) as session:
        _add_vessel(session, mmsi=422000002)
        raise BoomError

    with session_scope(session_factory) as session:
        assert session.get(Vessel, 422000002) is None


def test_foreign_keys_are_enforced(session_factory):
    """PRAGMA foreign_keys is off by default in SQLite; we switch it on."""
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError), session_scope(session_factory) as session:
        _add_position(session, mmsi=999999999, lon=56.5, lat=26.5)
