"""Tests for pixel <-> geographic conversion.

Built on a real Sentinel-1 control-point grid - S1D IW GRDH, acquired
2026-09-10 on a descending pass over the Gulf of Oman - so these tests pin
the behaviour of genuine radar geometry rather than an idealised grid.
Contains modified Copernicus Sentinel data 2026.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.control import GroundControlPoint  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import Affine, from_bounds  # noqa: E402

from core.db.spatial import BBox  # noqa: E402
from core.geo import angular_difference_deg, haversine_m  # noqa: E402
from core.sar.detect import Candidate, georeference  # noqa: E402
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.infrastructure import InfrastructureMask, Platform  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "s1d_20260910T020640_gcps.json"


@pytest.fixture(scope="module")
def grid() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gcps(grid) -> tuple:
    return tuple(
        GroundControlPoint(row=row, col=col, x=x, y=y, z=z) for row, col, x, y, z in grid["gcps"]
    )


@pytest.fixture(scope="module")
def scene(grid, gcps) -> SceneGeoref:
    return SceneGeoref(width=grid["width"], height=grid["height"], gcps=gcps)


def candidate(row: float = 100.0, col: float = 100.0, heading: float = 0.0) -> Candidate:
    return Candidate(
        row=row,
        col=col,
        length_m=200.0,
        width_m=30.0,
        heading_deg=heading,
        pixel_count=40,
        peak_amplitude=1500.0,
        mean_amplitude=900.0,
    )


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------


def test_control_points_are_reproduced_exactly(scene, gcps):
    """Guards the interpolation method.

    A thin-plate spline passes through every control point. GDAL's default
    polynomial fit, measured on this very grid, missed them by up to 1.2 km.
    Control points use corner coordinates, so each is probed half a pixel
    back from the pixel-centre convention lonlat applies.
    """
    rows = [g.row - 0.5 for g in gcps]
    cols = [g.col - 0.5 for g in gcps]

    lons, lats = scene.lonlat(rows, cols)
    errors = [haversine_m(lon, lat, g.x, g.y) for lon, lat, g in zip(lons, lats, gcps, strict=True)]

    assert max(errors) < 1.0


def test_pixels_round_trip_through_the_ground(scene):
    rows = np.array([2000.0, 8000.0, 14000.0])
    cols = np.array([3000.0, 12000.0, 22000.0])

    lons, lats = scene.lonlat(rows, cols)
    back_rows, back_cols = scene.raster_coords(lons, lats)

    # raster_coords is corner-based, so a pixel centre comes back at +0.5.
    assert np.allclose(back_rows, rows + 0.5, atol=1.0)
    assert np.allclose(back_cols, cols + 0.5, atol=1.0)


def test_pixel_spacing_is_about_ten_metres(scene):
    along_col, along_row = scene.metres_per_pixel(8000.0, 12000.0)

    assert 9.0 < along_col < 11.0
    assert 9.0 < along_row < 11.0


# --------------------------------------------------------------------------
# Orientation: why headings must be computed geographically
# --------------------------------------------------------------------------


def test_columns_run_east_to_west_on_a_descending_pass(scene):
    bearing = scene.bearing(8000.0, 12000.0, drow=0.0, dcol=1.0)

    assert 265.0 < bearing < 295.0


def test_rows_run_roughly_south(scene):
    bearing = scene.bearing(8000.0, 12000.0, drow=1.0, dcol=0.0)

    assert 175.0 < bearing < 205.0


def test_an_image_frame_heading_is_not_a_compass_heading(scene):
    """The mirror, made concrete.

    A diagonal that reads as 45 degrees in the image frame points north-west
    on the ground here, not north-east, because columns run westward.
    """
    located = georeference([candidate(heading=45.0)], scene, window_offset=(8000, 12000))[0]

    assert angular_difference_deg(located.heading_deg, 45.0) > 60.0
    assert 120.0 < located.heading_deg < 160.0


def test_georeference_places_candidates_at_the_scene_location(scene):
    located = georeference([candidate(row=100.0, col=100.0)], scene, window_offset=(8000, 12000))[0]
    lons, lats = scene.lonlat([8100.0], [12100.0])

    assert located.row == 8100.0
    assert located.col == 12100.0
    assert located.lon == pytest.approx(lons[0])
    assert located.lat == pytest.approx(lats[0])
    # Gulf of Oman.
    assert 55.5 < located.lon < 59.0
    assert 23.5 < located.lat < 26.5


def test_georeference_keeps_headings_within_half_a_circle(scene):
    """A SAR return is symmetric: bow and stern cannot be told apart."""
    located = georeference(
        [candidate(heading=h) for h in (0.0, 30.0, 90.0, 150.0)],
        scene,
        window_offset=(8000, 12000),
    )

    assert all(0.0 <= c.heading_deg < 180.0 for c in located)


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------


def test_a_box_inside_the_scene_gives_a_window_around_it(scene):
    lons, lats = scene.lonlat([8000.0], [12000.0])
    box = BBox(lons[0] - 0.05, lats[0] - 0.05, lons[0] + 0.05, lats[0] + 0.05)

    window = scene.window_for_bbox(box)

    assert not window.is_empty()
    assert window.col_off <= 12000 < window.col_off + window.width
    assert window.row_off <= 8000 < window.row_off + window.height
    # 0.1 degree is ~10 km, so roughly a thousand 10 m pixels either way.
    assert 700 < window.width < 1600
    assert 700 < window.height < 1600


def test_a_box_far_from_the_scene_gives_an_empty_window(scene):
    assert scene.window_for_bbox(BBox(40.0, 10.0, 41.0, 11.0)).is_empty()


def test_a_box_enclosing_the_scene_gives_the_whole_raster(scene, grid):
    window = scene.window_for_bbox(BBox(50.0, 20.0, 62.0, 30.0))

    assert (window.col_off, window.row_off) == (0, 0)
    assert (window.width, window.height) == (grid["width"], grid["height"])


def test_windows_need_raster_dimensions(gcps):
    with pytest.raises(ValueError, match="dimensions"):
        SceneGeoref(gcps=gcps).window_for_bbox(BBox(56.0, 25.0, 57.0, 26.0))


# --------------------------------------------------------------------------
# Affine rasters
# --------------------------------------------------------------------------


def test_the_affine_path_is_north_up():
    scene = SceneGeoref.from_affine(from_bounds(56.0, 26.0, 57.0, 27.0, 1000, 1000), 1000, 1000)

    assert scene.bearing(500.0, 500.0, drow=0.0, dcol=1.0) == pytest.approx(90.0, abs=1.0)
    assert scene.bearing(500.0, 500.0, drow=1.0, dcol=0.0) == pytest.approx(180.0, abs=1.0)


def test_the_affine_path_converts_both_ways():
    scene = SceneGeoref.from_affine(from_bounds(56.0, 26.0, 57.0, 27.0, 1000, 1000), 1000, 1000)

    rows, cols = scene.raster_coords([56.5], [26.5])
    lons, lats = scene.lonlat([499.5], [499.5])

    assert rows[0] == pytest.approx(500.0)
    assert cols[0] == pytest.approx(500.0)
    assert lons[0] == pytest.approx(56.5)
    assert lats[0] == pytest.approx(26.5)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_from_dataset_uses_control_points_when_present(gcps, grid):
    dataset = SimpleNamespace(
        gcps=(list(gcps), CRS.from_epsg(4326)),
        width=grid["width"],
        height=grid["height"],
        crs=None,
        transform=Affine.identity(),
    )

    assert SceneGeoref.from_dataset(dataset).uses_gcps


def test_from_dataset_rejects_a_raster_that_cannot_be_located():
    """No transform and no control points: the state a GRD scene appears to
    be in if only ``crs`` and ``transform`` are consulted."""
    dataset = SimpleNamespace(
        gcps=([], None), width=10, height=10, crs=None, transform=Affine.identity()
    )

    with pytest.raises(ValueError, match="cannot be located"):
        SceneGeoref.from_dataset(dataset)


def test_from_dataset_rejects_projected_control_points(gcps):
    dataset = SimpleNamespace(
        gcps=(list(gcps), CRS.from_epsg(32640)),
        width=10,
        height=10,
        crs=None,
        transform=Affine.identity(),
    )

    with pytest.raises(ValueError, match="EPSG:4326"):
        SceneGeoref.from_dataset(dataset)


def test_exactly_one_georeferencing_source_is_required(gcps):
    with pytest.raises(ValueError, match="exactly one"):
        SceneGeoref(width=10, height=10)
    with pytest.raises(ValueError, match="exactly one"):
        SceneGeoref(width=10, height=10, affine=Affine.identity(), gcps=gcps)


def test_a_zero_length_direction_has_no_bearing(scene):
    with pytest.raises(ValueError, match="non-zero"):
        scene.bearing(8000.0, 12000.0, drow=0.0, dcol=0.0)


# --------------------------------------------------------------------------
# Infrastructure masking on a real scene
# --------------------------------------------------------------------------


def test_a_platform_is_masked_at_its_true_pixel_in_a_window(scene):
    """End to end on radar geometry: ground position -> window pixel."""
    lons, lats = scene.lonlat([8000.0], [12000.0])
    platform = Platform(lon=float(lons[0]), lat=float(lats[0]), installed=None, last_observed=None)
    mask = InfrastructureMask([platform], radius_m=250.0)

    raster = mask.rasterise(scene, 200, 200, window_offset=(7900, 11900))

    assert raster[100, 100]
    assert not raster[0, 0]
    assert not raster[199, 199]
    # A 250 m radius at ~10 m per pixel is a disc about 25 pixels in radius.
    expected = np.pi * 25.0**2
    assert 0.7 * expected < raster.sum() < 1.3 * expected
