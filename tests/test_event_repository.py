"""Tests for storing and querying analyst events (DARK_VESSEL and later types).

Every event carries its evidence card: an alert that cannot be explained is
never stored (PLAN §9). Events belong to a scene and to the fusion run that
made them, so re-running a pass replaces that run's events instead of
doubling them, as detections are replaced per window.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest

from core.db.repository import EventRepository, NewEvent, SceneRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox

ACQ = dt.datetime(2026, 9, 25, 14, 15, 51, tzinfo=dt.UTC)
BOX = BBox(56.43, 25.13, 56.63, 25.31)


@dataclass(frozen=True)
class Extent:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


@dataclass(frozen=True)
class Product:
    """Shaped like core.sar.catalogue.SarProduct."""

    product_id: str
    name: str
    acq_start: dt.datetime = ACQ
    satellite: str = "S1C"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "ASCENDING"
    s3_path: str = "/eodata/Sentinel-1/x"
    footprint: tuple = ((55.8, 24.0), (58.7, 24.0), (58.7, 26.0), (55.8, 26.0), (55.8, 24.0))
    bbox: Extent = Extent(55.8, 24.0, 58.7, 26.0)


@pytest.fixture
def session():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    with session_scope(make_session_factory(engine)) as sess:
        yield sess


def scene(session, name: str = "S1C_A") -> int:
    SceneRepository(session).upsert(Product(product_id=name, name=name))
    return SceneRepository(session).id_for(name)


def dark(lon: float, lat: float, confidence: str = "high") -> NewEvent:
    return NewEvent(
        event_type="DARK_VESSEL",
        ts=ACQ,
        lon=lon,
        lat=lat,
        severity=confidence,
        detection_id=None,
        evidence={"status": "dark", "confidence": confidence},
    )


def test_events_are_stored_and_read_back_with_their_evidence(session):
    repo = EventRepository(session)
    scene_id = scene(session)
    assert repo.replace_for_scene(scene_id, "fusion-v1", [dark(56.50, 25.20)]) == 1
    (event,) = repo.in_bbox(BOX)
    assert (event.event_type, event.lon, event.lat, event.severity) == (
        "DARK_VESSEL",
        56.50,
        25.20,
        "high",
    )
    assert event.ts == ACQ and event.ts.tzinfo is not None
    assert event.evidence == {"status": "dark", "confidence": "high"}
    assert (event.scene_id, event.run) == (scene_id, "fusion-v1")


def test_rerunning_a_scene_replaces_only_that_runs_events(session):
    repo = EventRepository(session)
    first, second = scene(session, "S1C_A"), scene(session, "S1C_B")
    repo.replace_for_scene(first, "fusion-v1", [dark(56.50, 25.20), dark(56.51, 25.21)])
    repo.replace_for_scene(second, "fusion-v1", [dark(56.52, 25.22)])
    repo.replace_for_scene(first, "fusion-v0", [dark(56.53, 25.23)])
    repo.replace_for_scene(first, "fusion-v1", [dark(56.54, 25.24)])
    stored = sorted((e.scene_id, e.run, e.lon) for e in repo.in_bbox(BOX))
    assert stored == [
        (first, "fusion-v0", 56.53),
        (first, "fusion-v1", 56.54),
        (second, "fusion-v1", 56.52),
    ]


def test_queries_filter_by_box_and_type_newest_first(session):
    repo = EventRepository(session)
    scene_id = scene(session)
    later = NewEvent(**{**dark(56.55, 25.25).__dict__, "ts": ACQ + dt.timedelta(days=1)})
    other = NewEvent(**{**dark(56.56, 25.26).__dict__, "event_type": "ZONE_ENTRY"})
    repo.replace_for_scene(
        scene_id, "fusion-v1", [dark(56.50, 25.20), later, other, dark(57.5, 26.5)]
    )
    found = repo.in_bbox(BOX, event_type="DARK_VESSEL")
    assert [(e.lon, e.ts) for e in found] == [(56.55, later.ts), (56.50, ACQ)]


def test_an_event_without_evidence_is_refused_and_nothing_is_written(session):
    repo = EventRepository(session)
    scene_id = scene(session)
    unexplained = NewEvent(**{**dark(56.51, 25.21).__dict__, "evidence": {}})
    with pytest.raises(ValueError, match="evidence"):
        repo.replace_for_scene(scene_id, "fusion-v1", [dark(56.50, 25.20), unexplained])
    assert repo.in_bbox(BOX) == []


def test_one_event_is_read_by_id(session):
    repo = EventRepository(session)
    repo.replace_for_scene(scene(session), "fusion-v1", [dark(56.50, 25.20)])
    (stored,) = repo.in_bbox(BOX)
    assert repo.get(stored.id) == stored
    assert repo.get(stored.id + 1) is None


def test_a_passes_events_of_one_run_are_listed_in_id_order(session):
    repo = EventRepository(session)
    first, second = scene(session, "S1C_A"), scene(session, "S1C_B")
    repo.replace_for_scene(first, "fusion-v1", [dark(56.50, 25.20), dark(56.51, 25.21)])
    repo.replace_for_scene(first, "fusion-v0", [dark(56.52, 25.22)])
    repo.replace_for_scene(second, "fusion-v1", [dark(56.53, 25.23)])
    listed = repo.for_scene(first, "fusion-v1")
    assert [e.lon for e in listed] == [56.50, 56.51]
    assert [e.id for e in listed] == sorted(e.id for e in listed)
    assert sorted(repo.scene_ids("fusion-v1")) == sorted([first, second])
    assert repo.scene_ids("fusion-v0") == [first]


def test_an_events_evidence_is_completed_only_if_unchanged_and_never_emptied(session):
    """Compare and set: the card is replaced only if it is still the one the caller read."""
    repo = EventRepository(session)
    repo.replace_for_scene(scene(session), "fusion-v1", [dark(56.50, 25.20)])
    (stored,) = repo.in_bbox(BOX)
    completed = {**stored.evidence, "chip": {"image": {"pols": ["vv"]}}}
    assert repo.update_evidence(stored.id, expected=stored.evidence, evidence=completed)
    assert repo.get(stored.id).evidence == completed
    # A writer still holding the old card changes nothing.
    stale = {**stored.evidence, "chip": {"image": {"pols": ["vh"]}}}
    assert not repo.update_evidence(stored.id, expected=stored.evidence, evidence=stale)
    assert repo.get(stored.id).evidence == completed
    assert not repo.update_evidence(stored.id + 1, expected=completed, evidence=completed)
    with pytest.raises(ValueError, match="evidence"):
        repo.update_evidence(stored.id, expected=completed, evidence={})
    assert repo.get(stored.id).evidence == completed


def test_a_scene_is_read_by_id_with_its_image_path(session):
    scene_id = scene(session, "S1C_A")
    stored = SceneRepository(session).get(scene_id)
    assert (stored.product_name, stored.cog_url) == ("S1C_A", "/eodata/Sentinel-1/x")
    assert SceneRepository(session).get(scene_id + 1) is None
