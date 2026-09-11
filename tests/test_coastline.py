"""Tests for the coastline land mask."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterio.control import GroundControlPoint

from core.db.spatial import BBox
from core.geo import haversine_m
from core.sar.coastline import CoastlineMask, LandPolygon, clip_ring, load_land, save_polygons
from core.sar.georef import SceneGeoref

FIXTURE = Path(__file__).parent / "fixtures" / "s1d_20260910T020640_gcps.json"

# A north-up lon/lat raster of 200 x 200 pixels, each 0.0001 deg (about 10 m
# along columns and 11 m along rows at this latitude), spanning 56.00-56.02 E
# and 25.00-25.02 N. Column 100 starts at 56.01 E; row 100 starts at 25.01 N.
SCENE = SceneGeoref.from_affine(Affine(0.0001, 0.0, 56.0, 0.0, -0.0001, 25.02))
SIZE = 200


def square(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> np.ndarray:
    return np.array(
        [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]
    )


# Land covering everything west of 56.01 E, reaching well beyond the raster.
WEST = LandPolygon(square(55.9, 24.9, 56.01, 25.1))
# An island with a lagoon whose centre is pixel (100, 100).
LAGOON_ISLAND = LandPolygon(
    square(56.002, 25.002, 56.018, 25.018), holes=(square(56.006, 25.006, 56.014, 25.014),)
)
FAR_AWAY = LandPolygon(square(10.0, 50.0, 10.1, 50.1))


def shoelace_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


@pytest.fixture(scope="module")
def real_scene() -> SceneGeoref:
    grid = json.loads(FIXTURE.read_text(encoding="utf-8"))
    gcps = tuple(
        GroundControlPoint(row=row, col=col, x=x, y=y, z=z) for row, col, x, y, z in grid["gcps"]
    )
    return SceneGeoref(width=grid["width"], height=grid["height"], gcps=gcps)


# --------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------

BOX = BBox(0.0, 0.0, 10.0, 10.0)


def test_a_ring_inside_the_box_is_unchanged():
    ring = np.array([(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)])

    assert np.allclose(clip_ring(ring, BOX), ring)


def test_a_ring_outside_the_box_vanishes():
    ring = np.array([(20.0, 20.0), (21.0, 20.0), (21.0, 21.0)])

    assert len(clip_ring(ring, BOX)) == 0


def test_a_crossing_ring_is_cut_at_the_box_edge():
    ring = np.array([(5.0, 5.0), (15.0, 5.0), (15.0, 8.0), (5.0, 8.0)])

    clipped = clip_ring(ring, BOX)

    assert clipped[:, 0].max() == pytest.approx(10.0)
    assert shoelace_area(clipped) == pytest.approx(15.0)


def test_a_polygon_needs_at_least_three_points():
    with pytest.raises(ValueError, match="three"):
        LandPolygon(np.array([(0.0, 0.0), (1.0, 1.0)]))


def test_the_buffer_cannot_be_negative():
    with pytest.raises(ValueError, match="buffer"):
        CoastlineMask([WEST], buffer_m=-1.0)


# --------------------------------------------------------------------------
# Rasterising
# --------------------------------------------------------------------------


def test_land_pixels_are_masked_and_sea_pixels_are_not():
    mask = CoastlineMask([WEST], buffer_m=0.0).rasterise(SCENE, SIZE, SIZE)

    assert mask.dtype == bool
    assert mask[:, :100].all()
    assert not mask[:, 100:].any()


def test_the_buffer_extends_the_mask_seaward():
    """100 m at about 10 m per column is 10 pixels."""
    mask = CoastlineMask([WEST], buffer_m=100.0).rasterise(SCENE, SIZE, SIZE)

    assert mask[:, :110].all()
    assert not mask[:, 110:].any()


def test_land_just_outside_the_window_still_buffers_into_it():
    """Shoreline structures sit on the land edge whichever window holds the land."""
    beyond = LandPolygon(square(55.9, 24.9, 55.9995, 25.1))  # ends 5 px west of the raster

    mask = CoastlineMask([beyond], buffer_m=100.0).rasterise(SCENE, SIZE, SIZE)

    assert mask[:, :5].all()
    assert not mask[:, 5:].any()


def test_a_lagoon_inside_land_stays_unmasked():
    mask = CoastlineMask([LAGOON_ISLAND], buffer_m=0.0).rasterise(SCENE, SIZE, SIZE)

    assert not mask[95:105, 95:105].any()
    assert mask[25, 100]  # on the island's rim


def test_a_sub_window_matches_the_same_pixels_of_the_whole_raster():
    coastline = CoastlineMask([WEST, LAGOON_ISLAND], buffer_m=100.0)

    whole = coastline.rasterise(SCENE, SIZE, SIZE)
    part = coastline.rasterise(SCENE, 100, SIZE, window_offset=(0, 50))

    assert np.array_equal(part, whole[:, 50:150])


def test_distant_land_leaves_the_window_empty():
    assert not CoastlineMask([FAR_AWAY]).rasterise(SCENE, SIZE, SIZE).any()


def test_land_is_placed_correctly_on_a_real_radar_scene(real_scene):
    """Radar geometry is rotated and mirrored; the mask must follow the GCPs."""
    lon, lat, half = 56.53, 25.22, 0.002
    rows, cols = real_scene.raster_coords([lon], [lat])
    row_off, col_off = int(rows[0]) - 100, int(cols[0]) - 100
    island = LandPolygon(square(lon - half, lat - half, lon + half, lat + half))

    mask = CoastlineMask([island], buffer_m=0.0).rasterise(
        real_scene, SIZE, SIZE, window_offset=(row_off, col_off)
    )

    masked_rows, masked_cols = np.nonzero(mask)
    centre_lon, centre_lat = real_scene.lonlat(
        [masked_rows.mean() + row_off], [masked_cols.mean() + col_off]
    )
    assert haversine_m(float(centre_lon[0]), float(centre_lat[0]), lon, lat) < 15.0
    # About 403 m by 442 m on a ~10 m grid: roughly 1,750 pixels.
    assert 1_500 < mask.sum() < 2_000


# --------------------------------------------------------------------------
# Reference file
# --------------------------------------------------------------------------


def test_polygons_round_trip_through_the_reference_file(tmp_path):
    path = tmp_path / "land.npz"

    save_polygons(path, [WEST, LAGOON_ISLAND], source="test")
    loaded = load_land(path)

    assert len(loaded) == 2
    assert np.allclose(loaded[0].exterior, WEST.exterior)
    assert len(loaded[1].holes) == 1
    assert np.allclose(loaded[1].holes[0], LAGOON_ISLAND.holes[0])


def test_loading_can_be_limited_to_a_box(tmp_path):
    path = tmp_path / "land.npz"
    save_polygons(path, [WEST, FAR_AWAY], source="test")

    loaded = load_land(path, bbox=BBox(55.0, 24.0, 57.0, 26.0))

    assert len(loaded) == 1


def test_a_missing_reference_file_says_how_to_fetch_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_coastline"):
        load_land(tmp_path / "absent.npz")
