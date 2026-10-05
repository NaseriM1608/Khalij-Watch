"""Tests for recording where the detector actually looked (coverage honesty, PLAN §10).

A radar footprint says the satellite imaged an area; only an analysed window
says a detector searched it. A window is recorded with every run, also when
it found nothing, so the console can tell "searched" from "not searched".
Windows are kept per pass, detector and polarisation, like the detections they
describe; their counts are those stored now; windows of fused passes stay
visible as long as their dark-vessel candidates do.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest

from core.areas import box_around
from core.db.repository import (
    AnalysedWindowRepository,
    DetectionRepository,
    EventRepository,
    NewEvent,
    SceneRepository,
)
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox
from core.sar.detect import Candidate

ACQ = dt.datetime(2026, 10, 3, 2, 14, 40, tzinfo=dt.UTC)
FUJAIRAH = box_around(56.53, 25.22, 20.0)
BANDAR_ABBAS = box_around(56.25, 27.03, 20.0)


@dataclass(frozen=True)
class Extent:
    min_lon: float = 55.8
    min_lat: float = 24.0
    max_lon: float = 58.7
    max_lat: float = 27.5


@dataclass(frozen=True)
class Product:
    """Shaped like core.sar.catalogue.SarProduct."""

    product_id: str
    name: str
    acq_start: dt.datetime = ACQ
    satellite: str = "S1C"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "DESCENDING"
    s3_path: str = "/eodata/Sentinel-1/x"
    footprint: tuple = ((55.8, 24.0), (58.7, 24.0), (58.7, 27.5), (55.8, 27.5), (55.8, 24.0))
    bbox: Extent = Extent()
    online: bool = True


@pytest.fixture
def session():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    with session_scope(make_session_factory(engine)) as sess:
        yield sess


def scene(session, name: str = "S1C_A", acq: dt.datetime = ACQ) -> int:
    SceneRepository(session).upsert(Product(product_id=name, name=name, acq_start=acq))
    return SceneRepository(session).id_for(name)


GULF = BBox(47.5, 23.5, 58.0, 30.5)


def record(repo, scene_id, box=FUJAIRAH, detector="pointdet", version="kaggle-v2", pol="DV"):
    repo.record(
        scene_id, box, detector=detector, detector_version=version, polarisation=pol, detections=0
    )


def ship(lon: float, lat: float) -> Candidate:
    return Candidate(
        row=10.0,
        col=10.0,
        length_m=120.0,
        width_m=20.0,
        heading_deg=30.0,
        pixel_count=30,
        peak_amplitude=1500.0,
        mean_amplitude=900.0,
        lon=lon,
        lat=lat,
    )


def store(session, scene_id, box, ships, detector="cfar", pol="VH") -> int:
    return DetectionRepository(session).replace_in_bbox(
        scene_id,
        box,
        ships,
        polarisation=pol,
        detector=detector,
        detector_version="L4.4",
        confidence=1.0,
    )


def test_a_window_is_recorded_with_its_pass_even_when_nothing_was_found(session):
    repo = AnalysedWindowRepository(session)
    record(repo, scene(session))
    (window,) = repo.recent(GULF, since=ACQ - dt.timedelta(days=1))
    assert window.detections == 0
    assert (window.detector, window.detector_version, window.polarisation) == (
        "pointdet",
        "kaggle-v2",
        "DV",
    )
    assert window.footprint_wkt.startswith("POLYGON")
    assert (window.product_name, window.satellite, window.acq_time) == ("S1C_A", "S1C", ACQ)
    assert window.bbox == FUJAIRAH


def test_re_running_a_window_replaces_its_record(session):
    repo = AnalysedWindowRepository(session)
    scene_id = scene(session)
    record(repo, scene_id, version="kaggle-v1")
    record(repo, scene_id, version="kaggle-v2")
    record(repo, scene_id, detector="cfar", version="L4.4", pol="VH")
    windows = repo.recent(GULF, since=ACQ - dt.timedelta(days=1))
    assert sorted((w.detector, w.detector_version) for w in windows) == [
        ("cfar", "L4.4"),
        ("pointdet", "kaggle-v2"),
    ]


def test_runs_on_each_polarisation_keep_their_own_window(session):
    """As detections are replaced per polarisation, so are their windows."""
    repo = AnalysedWindowRepository(session)
    scene_id = scene(session)
    record(repo, scene_id, detector="cfar", version="L4.4", pol="VH")
    record(repo, scene_id, detector="cfar", version="L4.4", pol="VV")
    windows = repo.recent(GULF, since=ACQ - dt.timedelta(days=1))
    assert sorted(w.polarisation for w in windows) == ["VH", "VV"]


def test_a_window_counts_the_detections_stored_now(session):
    """A later run over an overlapping box replaces rows the first window counted."""
    repo = AnalysedWindowRepository(session)
    scene_id = scene(session)
    first = [ship(56.50, 25.20), ship(56.60, 25.25)]
    record(repo, scene_id, FUJAIRAH, detector="cfar", version="L4.4", pol="VH")
    store(session, scene_id, FUJAIRAH, first)
    east = BBox(56.55, 25.15, 56.65, 25.30)
    store(session, scene_id, east, [])
    record(repo, scene_id, east, detector="cfar", version="L4.4", pol="VH")
    counts = {w.bbox: w.detections for w in repo.recent(GULF, since=ACQ - dt.timedelta(days=1))}
    assert counts == {FUJAIRAH: 1, east: 0}


def test_windows_of_fused_passes_stay_while_their_candidates_are_shown(session):
    """Dark-vessel candidates have no lookback, so neither has the window they came from."""
    repo = AnalysedWindowRepository(session)
    old = scene(session, "S1D_OLD", ACQ - dt.timedelta(days=25))
    stale = scene(session, "S1D_STALE", ACQ - dt.timedelta(days=25))
    record(repo, old)
    record(repo, stale)
    EventRepository(session).replace_for_scene(
        old,
        "fusion-v1",
        [
            NewEvent(
                event_type="DARK_VESSEL",
                ts=ACQ - dt.timedelta(days=25),
                lon=56.5,
                lat=25.2,
                severity="low",
                detection_id=None,
                evidence={"status": "dark"},
            )
        ],
    )
    windows = repo.recent(GULF, since=ACQ - dt.timedelta(days=14))
    assert [w.product_name for w in windows] == ["S1D_OLD"]


def test_recent_windows_are_those_of_passes_since_a_time_that_overlap_the_box(session):
    repo = AnalysedWindowRepository(session)
    new, old = scene(session, "S1C_NEW"), scene(session, "S1D_OLD", ACQ - dt.timedelta(days=20))
    record(repo, new, BANDAR_ABBAS)
    record(repo, new, FUJAIRAH)
    record(repo, old, FUJAIRAH)
    since = ACQ - dt.timedelta(days=14)
    gulf = repo.recent(GULF, since=since)
    assert [(w.product_name, w.bbox) for w in gulf] in (
        [("S1C_NEW", BANDAR_ABBAS), ("S1C_NEW", FUJAIRAH)],
        [("S1C_NEW", FUJAIRAH), ("S1C_NEW", BANDAR_ABBAS)],
    )
    iran = repo.recent(BBox(56.0, 26.8, 56.5, 27.3), since=since)
    assert [w.bbox for w in iran] == [BANDAR_ABBAS]


def test_detect_window_records_the_window_with_its_detections(tmp_path):
    """scripts/detect_window.py stores a run's detections and its window together."""
    from scripts.detect_window import store_detections

    url = f"sqlite+pysqlite:///{(tmp_path / 'khalij.db').as_posix()}"
    product = Product(product_id="S1C_B", name="S1C_B")
    assert (
        store_detections(
            url, product, BANDAR_ABBAS, [ship(56.25, 27.03)], "dv", "kaggle-v2", detector="pointdet"
        )
        == 1
    )
    store_detections(url, product, FUJAIRAH, [], "vh", "L4.4", detector="cfar")
    engine = create_db_engine(url)
    with session_scope(make_session_factory(engine)) as sess:
        windows = AnalysedWindowRepository(sess).recent(GULF, since=ACQ - dt.timedelta(days=1))
    assert sorted(
        (w.detector, w.polarisation, w.detections, w.bbox == BANDAR_ABBAS) for w in windows
    ) == [("cfar", "VH", 0, False), ("pointdet", "DV", 1, True)]
