"""Tests for storing and querying SAR detections."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest
from sqlalchemy import text

from core.db.repository import DetectionRepository, SceneRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox

ACQ = dt.datetime(2026, 9, 10, 2, 6, 40, tzinfo=dt.UTC)
WINDOW = BBox(56.4, 25.1, 56.6, 25.3)


@dataclass(frozen=True)
class Target:
    """Shaped like core.sar.detect.Candidate, without pulling in numpy."""

    lon: float | None
    lat: float | None
    length_m: float = 212.0
    width_m: float = 63.0
    heading_deg: float = 127.0
    pixel_count: int = 180
    peak_amplitude: float = 2911.0


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
    satellite: str = "S1D"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "DESCENDING"
    s3_path: str = "/eodata/Sentinel-1/x"
    footprint: tuple = ((55.8, 24.0), (58.7, 24.0), (58.7, 26.0), (55.8, 26.0), (55.8, 24.0))
    bbox: Extent = Extent(55.8, 24.0, 58.7, 26.0)


@pytest.fixture
def session():
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    with session_scope(make_session_factory(engine)) as sess:
        yield sess


@pytest.fixture
def detections(session):
    return DetectionRepository(session)


def add_scene(session, product_id: str = "s1d-a", acq: dt.datetime = ACQ) -> int:
    scenes = SceneRepository(session)
    scenes.upsert(Product(product_id=product_id, name=f"S1D_{product_id}_COG.SAFE", acq_start=acq))
    session.flush()
    return scenes.id_for(product_id)


def store(detections, scene_id, targets, *, bbox=WINDOW, polarisation="VH") -> int:
    return detections.replace_in_bbox(
        scene_id,
        bbox,
        targets,
        polarisation=polarisation,
        detector="cfar",
        detector_version="test",
        confidence=1.0,
    )


def test_stored_detections_come_back_with_their_scene(session, detections):
    scene_id = add_scene(session)
    written = store(detections, scene_id, [Target(56.5, 25.2)])
    session.flush()

    found = detections.in_bbox(WINDOW)

    assert written == 1
    assert len(found) == 1
    hit = found[0]
    assert (hit.lon, hit.lat) == (56.5, 25.2)
    assert hit.satellite == "S1D"
    assert hit.polarisation == "VH"
    assert hit.length_m == 212.0
    assert hit.pixel_count == 180
    assert hit.peak_dn == pytest.approx(2911.0)
    assert hit.detector == "cfar"


@dataclass(frozen=True)
class Point:
    """Shaped like a point-detector result: a position and a score, no hull."""

    lon: float
    lat: float
    confidence: float
    length_m: None = None
    width_m: None = None
    heading_deg: None = None
    pixel_count: None = None
    peak_amplitude: None = None


def test_point_detections_store_their_own_confidence_and_no_hull(session, detections):
    scene_id = add_scene(session)
    detections.replace_in_bbox(
        scene_id,
        WINDOW,
        [Point(56.5, 25.2, 0.83)],
        polarisation="VV+VH",
        detector="pointdet",
        detector_version="kaggle-v1",
        confidence=1.0,
    )
    session.flush()

    hit = detections.in_bbox(WINDOW)[0]

    assert hit.confidence == pytest.approx(0.83)
    assert hit.detector == "pointdet"
    assert (hit.length_m, hit.width_m, hit.heading_deg, hit.pixel_count, hit.peak_dn) == (
        None,
        None,
        None,
        None,
        None,
    )


def test_detection_age_is_measured_from_the_radar_pass(session, detections):
    """Not from when the row was written: radar is never live."""
    scene_id = add_scene(session)
    store(detections, scene_id, [Target(56.5, 25.2)])
    session.flush()

    hit = detections.in_bbox(WINDOW)[0]

    assert hit.acq_time == ACQ
    assert hit.age_seconds(ACQ + dt.timedelta(hours=1)) == pytest.approx(3600.0)


def test_rerunning_a_window_replaces_rather_than_duplicates(session, detections):
    scene_id = add_scene(session)
    targets = [Target(56.45, 25.2), Target(56.55, 25.2)]

    store(detections, scene_id, targets)
    session.flush()
    store(detections, scene_id, targets)
    session.flush()

    assert detections.count() == 2


def test_replacement_leaves_other_channels_scenes_and_areas_alone(session, detections):
    scene_id = add_scene(session)
    other_scene = add_scene(session, "s1d-b")
    elsewhere = BBox(56.7, 25.1, 56.9, 25.3)
    store(detections, scene_id, [Target(56.5, 25.2)])
    store(detections, scene_id, [Target(56.5, 25.2)], polarisation="VV")
    store(detections, other_scene, [Target(56.5, 25.2)])
    store(detections, scene_id, [Target(56.8, 25.2)], bbox=elsewhere)
    session.flush()

    # The first window re-run finds nothing: only its own rows may go.
    store(detections, scene_id, [])
    session.flush()

    assert detections.count() == 3


def test_ungeoreferenced_detections_are_refused(session, detections):
    scene_id = add_scene(session)

    with pytest.raises(ValueError, match="georeferenced"):
        store(detections, scene_id, [Target(None, None)])


def test_detections_outside_the_window_are_refused(session, detections):
    """A re-run replaces only inside its window, so rows outside it would duplicate."""
    scene_id = add_scene(session)

    with pytest.raises(ValueError, match="outside"):
        store(detections, scene_id, [Target(57.5, 25.2)])


def test_query_box_excludes_detections_outside_it(session, detections):
    scene_id = add_scene(session)
    store(detections, scene_id, [Target(56.5, 25.2)])
    session.flush()

    assert detections.in_bbox(BBox(0.0, 50.0, 5.0, 55.0)) == []


def test_since_excludes_older_passes(session, detections):
    old = add_scene(session, "old", acq=ACQ - dt.timedelta(days=20))
    new = add_scene(session, "new")
    store(detections, old, [Target(56.45, 25.2)])
    store(detections, new, [Target(56.55, 25.2)])
    session.flush()

    found = detections.in_bbox(WINDOW, since=ACQ - dt.timedelta(days=1))

    assert [d.lon for d in found] == [56.55]


def test_replacing_keeps_the_spatial_index_consistent(session, detections):
    scene_id = add_scene(session)
    store(detections, scene_id, [Target(56.45, 25.2), Target(56.55, 25.2)])
    session.flush()
    store(detections, scene_id, [Target(56.5, 25.2)])
    session.flush()

    indexed = session.execute(text("SELECT count(*) FROM detections_rtree")).scalar()

    assert indexed == detections.count() == 1


def test_rerun_never_removes_a_neighbouring_detection_just_outside_the_window(session, detections):
    """The R*Tree stores 32-bit floats rounded outward, so its box test also
    matches points up to ~0.4 m outside a window. Measured: 19,503 of 20,000
    points 1 cm outside a query box were returned. Deleting on that test alone
    would silently drop a neighbouring window's detection."""
    scene_id = add_scene(session)
    neighbour = BBox(WINDOW.max_lon, WINDOW.min_lat, WINDOW.max_lon + 0.2, WINDOW.max_lat)
    store(detections, scene_id, [Target(WINDOW.max_lon + 1e-7, 25.2)], bbox=neighbour)
    store(detections, scene_id, [Target(56.5, 25.2)])
    session.flush()

    store(detections, scene_id, [Target(56.5, 25.2)])
    session.flush()

    assert detections.count() == 2


def test_scene_id_lookup(session):
    scene_id = add_scene(session)

    assert SceneRepository(session).id_for("s1d-a") == scene_id
    assert SceneRepository(session).id_for("no-such-product") is None
