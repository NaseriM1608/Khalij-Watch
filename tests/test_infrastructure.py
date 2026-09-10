"""Tests for the offshore infrastructure mask."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from core.db.spatial import GULF_BBOX, HORMUZ_BBOX, BBox
from core.sar.infrastructure import (
    DEFAULT_PATH,
    PERSIAN_GULF,
    InfrastructureMask,
    Platform,
    active_platforms,
    load_platforms,
    parse_quarter,
    quarter_end,
)

DATASET_PRESENT = DEFAULT_PATH.is_file()
needs_dataset = pytest.mark.skipif(
    not DATASET_PRESENT,
    reason="run scripts/fetch_platforms.py to fetch the OPD dataset",
)


# --------------------------------------------------------------------------
# Quarter parsing
# --------------------------------------------------------------------------


def test_a_quarter_parses_to_its_first_day():
    assert parse_quarter("19Q3") == dt.date(2019, 7, 1)
    assert parse_quarter("23Q4") == dt.date(2023, 10, 1)
    assert parse_quarter("17Q1") == dt.date(2017, 1, 1)


def test_open_ended_markers_parse_to_none():
    """These mean 'before the record' and 'still standing', not a date."""
    assert parse_quarter("17Q1 or before") is None
    assert parse_quarter("25Q1 or still present") is None


@pytest.mark.parametrize("text", ["", None, "garbage", "19Q5", "19X3", "1900Q1"])
def test_malformed_quarters_parse_to_none(text):
    assert parse_quarter(text) is None


def test_quarter_end_is_the_last_day():
    assert quarter_end(dt.date(2019, 7, 1)) == dt.date(2019, 9, 30)
    assert quarter_end(dt.date(2023, 10, 1)) == dt.date(2023, 12, 31)


# --------------------------------------------------------------------------
# Lifespans: the property that makes the mask correct rather than merely present
# --------------------------------------------------------------------------


def platform(installed=None, last=None, lon=55.0, lat=26.0) -> Platform:
    return Platform(
        lon=lon,
        lat=lat,
        installed=parse_quarter(installed),
        last_observed=parse_quarter(last),
    )


def test_a_platform_is_absent_before_it_was_installed():
    """Masking a 2019 scene with a 2023 structure would hide a real vessel."""
    built_2023 = platform(installed="23Q4")

    assert not built_2023.was_present(dt.date(2019, 6, 1))
    assert built_2023.was_present(dt.date(2024, 6, 1))


def test_a_platform_is_absent_after_it_was_removed():
    removed = platform(installed="17Q2", last="21Q3")

    assert removed.was_present(dt.date(2020, 1, 1))
    assert not removed.was_present(dt.date(2024, 1, 1))


def test_a_removed_platform_still_counts_through_its_final_quarter():
    removed = platform(installed="17Q2", last="21Q3")

    assert removed.was_present(dt.date(2021, 9, 30))
    assert not removed.was_present(dt.date(2021, 10, 1))


def test_open_ended_bounds_mean_always_present():
    permanent = platform(installed="17Q1 or before", last="25Q1 or still present")

    assert permanent.was_present(dt.date(2017, 1, 1))
    assert permanent.was_present(dt.date(2026, 9, 1))


def test_active_platforms_filters_by_date():
    platforms = [
        platform(installed="23Q4", lon=55.0),
        platform(installed="17Q2", last="19Q1", lon=55.1),
        platform(lon=55.2),
    ]

    in_2018 = active_platforms(platforms, dt.date(2018, 6, 1))
    in_2024 = active_platforms(platforms, dt.date(2024, 6, 1))

    assert {p.lon for p in in_2018} == {55.1, 55.2}
    assert {p.lon for p in in_2024} == {55.0, 55.2}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_a_missing_dataset_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_platforms"):
        load_platforms(tmp_path / "absent.geojson")


def test_loading_filters_by_region_and_bbox(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [55.0, 26.0]},
                "properties": {"region": PERSIAN_GULF, "installed_quarter": "17Q2"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-90.0, 26.0]},
                "properties": {"region": "Gulf of Mexico"},
            },
        ],
    }
    path = tmp_path / "opd.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")

    gulf = load_platforms(path, region=PERSIAN_GULF)

    assert len(gulf) == 1
    assert gulf[0].lon == 55.0


def test_non_point_features_are_skipped(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[1, 2], [3, 4]]},
                "properties": {"region": PERSIAN_GULF},
            }
        ],
    }
    path = tmp_path / "opd.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_platforms(path) == []


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


def test_covers_reports_positions_near_a_structure():
    mask = InfrastructureMask([platform(lon=55.0, lat=26.0)], radius_m=250.0)

    assert mask.covers(55.0, 26.0)
    # About 100 m away in latitude.
    assert mask.covers(55.0, 26.0009)
    # About 1 km away: open water.
    assert not mask.covers(55.0, 26.01)


def test_for_date_narrows_the_mask():
    mask = InfrastructureMask(
        [platform(installed="23Q4", lon=55.0), platform(installed="17Q2", lon=55.5)]
    )

    assert len(mask.for_date(dt.date(2019, 1, 1))) == 1
    assert len(mask.for_date(dt.date(2024, 1, 1))) == 2


def test_rasterise_marks_a_disc_around_each_platform():
    from rasterio.transform import from_bounds

    # 1000 px across 1 degree of longitude at 26 N is about 111 m per pixel.
    transform = from_bounds(55.0, 26.0, 56.0, 27.0, 1000, 1000)
    mask = InfrastructureMask([platform(lon=55.5, lat=26.5)], radius_m=250.0)

    raster = mask.rasterise(transform, 1000, 1000)

    assert raster[500, 500]
    assert not raster[0, 0]
    # A 250 m radius at ~111 m per pixel is a disc a few pixels across.
    assert 5 < raster.sum() < 100


def test_rasterise_with_no_platforms_is_empty():
    from rasterio.transform import from_bounds

    transform = from_bounds(55.0, 26.0, 56.0, 27.0, 100, 100)

    assert not InfrastructureMask([]).rasterise(transform, 100, 100).any()


def test_rasterise_clips_platforms_outside_the_window():
    from rasterio.transform import from_bounds

    transform = from_bounds(55.0, 26.0, 56.0, 27.0, 100, 100)
    mask = InfrastructureMask([platform(lon=20.0, lat=60.0)])

    assert not mask.rasterise(transform, 100, 100).any()


# --------------------------------------------------------------------------
# The real dataset
# --------------------------------------------------------------------------


@needs_dataset
def test_the_gulf_holds_thousands_of_platforms():
    """The reason this mask exists at all."""
    platforms = load_platforms(region=PERSIAN_GULF, bbox=GULF_BBOX)

    assert len(platforms) > 2000


@needs_dataset
def test_every_gulf_platform_sits_inside_the_aoi():
    platforms = load_platforms(region=PERSIAN_GULF, bbox=GULF_BBOX)

    assert all(GULF_BBOX.contains(p.lon, p.lat) for p in platforms)


@needs_dataset
def test_the_strait_of_hormuz_has_far_fewer_platforms_than_the_shallows():
    """Sanity: production sits in the shallow south-west, not the deep strait."""
    hormuz = load_platforms(region=PERSIAN_GULF, bbox=HORMUZ_BBOX)
    western = load_platforms(region=PERSIAN_GULF, bbox=BBox(48.0, 26.0, 53.0, 30.0))

    assert len(western) > len(hormuz) * 3


@needs_dataset
def test_platforms_are_in_shallow_water():
    platforms = load_platforms(region=PERSIAN_GULF, bbox=GULF_BBOX)
    depths = [abs(p.water_depth_m) for p in platforms if p.water_depth_m]

    # The Gulf is shallow throughout; nothing should sit in deep water.
    assert max(depths) < 200


@needs_dataset
def test_a_real_mask_excludes_a_meaningful_area_of_a_scene():
    from rasterio.transform import from_bounds

    platforms = load_platforms(region=PERSIAN_GULF, bbox=GULF_BBOX)
    mask = InfrastructureMask(platforms).for_date(dt.date(2024, 6, 1))

    # A 1-degree window over the Saudi/Qatari shelf at ~110 m per pixel.
    transform = from_bounds(50.0, 26.0, 51.0, 27.0, 1000, 1000)
    raster = mask.rasterise(transform, 1000, 1000)

    assert raster.any(), "no platforms masked in a known production area"
    # Small in absolute terms, but exactly the pixels that would otherwise
    # dominate the detections.
    assert raster.mean() < 0.05


@needs_dataset
def test_the_mask_shrinks_when_asked_for_an_earlier_date():
    platforms = load_platforms(region=PERSIAN_GULF, bbox=GULF_BBOX)
    mask = InfrastructureMask(platforms)

    early = len(mask.for_date(dt.date(2017, 6, 1)))
    late = len(mask.for_date(dt.date(2025, 1, 1)))

    assert early < late


@needs_dataset
def test_detections_on_a_platform_would_be_suppressed():
    """End to end: a bright fixed target at a known platform is masked out."""
    from rasterio.transform import from_bounds

    from core.sar.cfar import CfarConfig
    from core.sar.detect import detect_candidates
    from core.sar.synthetic import Target, make_scene

    platforms = load_platforms(region=PERSIAN_GULF, bbox=GULF_BBOX)
    subject = next(p for p in platforms if p.last_observed is None)

    # A window centred on that platform, at roughly Sentinel-1 resolution.
    half = 0.02
    transform = from_bounds(
        subject.lon - half,
        subject.lat - half,
        subject.lon + half,
        subject.lat + half,
        400,
        400,
    )
    scene = make_scene((400, 400), targets=[Target(200, 200, length_px=12, width_px=8)], seed=5)
    mask = InfrastructureMask(platforms).for_date(dt.date(2024, 6, 1))
    raster = mask.rasterise(transform, 400, 400)

    assert raster[200, 200], "the platform should mask its own location"

    config = CfarConfig(pfa=1e-6)
    unmasked = detect_candidates(scene.amplitude, config=config)
    masked = detect_candidates(scene.amplitude, config=config, mask=raster)

    assert len(unmasked) == 1
    assert len(masked) == 0


def test_numpy_mask_dtype_is_boolean():
    from rasterio.transform import from_bounds

    transform = from_bounds(55.0, 26.0, 56.0, 27.0, 50, 50)
    raster = InfrastructureMask([platform(lon=55.5, lat=26.5)]).rasterise(transform, 50, 50)

    assert raster.dtype == np.bool_
