"""Tests for rendering radar chips that people look at (Phase 7 evidence card).

The chip must read like the map: north up, east right, at a known scale.
Ground-range scenes are not: a descending pass shows the sea mirrored east-west
and turned by the track angle. The geometry tests therefore run on a real
Sentinel-1 control-point grid (tests/chip_fakes.py): the descending S1D pass of
2026-09-10 that holds the console's dark-vessel candidates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import Affine  # noqa: E402

from core.fusion.chips import CHIP_STRETCH_DN  # noqa: E402
from core.geo import EARTH_RADIUS_M, haversine_m  # noqa: E402
from core.sar.display import (  # noqa: E402
    cut_north_up,
    db_stretch,
    encode_gray_png,
    north_up_grid,
    sample_bilinear,
)
from core.sar.georef import SceneGeoref  # noqa: E402
from tests.chip_fakes import (  # noqa: E402
    LAT,
    LON,
    SEA_DN,
    SHIP_DN,
    FakeBand,
    decode_gray_png,
    load_gcps,
    load_grid,
)

SIZE, PIXEL_M = 256, 10.0


@pytest.fixture(scope="module")
def grid() -> dict:
    return load_grid()


@pytest.fixture(scope="module")
def gcps(grid) -> tuple:
    return load_gcps(grid)


@pytest.fixture(scope="module")
def scene(grid, gcps) -> SceneGeoref:
    return SceneGeoref(width=grid["width"], height=grid["height"], gcps=gcps)


def offset(lon: float, lat: float, *, north_m: float = 0.0, east_m: float = 0.0):
    return (
        lon + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))),
        lat + math.degrees(north_m / EARTH_RADIUS_M),
    )


def pixel_of(scene: SceneGeoref, lon: float, lat: float) -> tuple[int, int]:
    """The raster pixel containing a point."""
    rows, cols = scene.raster_coords([lon], [lat])
    return math.floor(rows[0]), math.floor(cols[0])


def brightest(chip: np.ndarray) -> tuple[int, int]:
    row, col = np.unravel_index(int(np.argmax(chip)), chip.shape)
    return int(row), int(col)


def cut(scene, bands, lon=LON, lat=LAT):
    return cut_north_up(bands, scene, lon, lat, size_px=SIZE, pixel_m=PIXEL_M)


# --------------------------------------------------------------------------
# The north-up grid
# --------------------------------------------------------------------------


def test_the_grid_is_north_up_and_centred_on_the_point():
    lons, lats = north_up_grid(LON, LAT, size_px=4, pixel_m=10.0)
    assert lons.shape == lats.shape == (4, 4)
    assert np.all(np.diff(lats, axis=0) < 0)  # row 0 is the northern edge
    assert np.all(np.diff(lons, axis=1) > 0)  # column 0 is the western edge
    assert lons.mean() == pytest.approx(LON, abs=1e-9)
    assert lats.mean() == pytest.approx(LAT, abs=1e-9)


def test_grid_pixels_are_the_requested_size_on_the_ground():
    lons, lats = north_up_grid(LON, LAT, size_px=SIZE, pixel_m=PIXEL_M)
    across = haversine_m(lons[0, 0], lats[0, 0], lons[0, -1], lats[0, -1])
    down = haversine_m(lons[0, 0], lats[0, 0], lons[-1, 0], lats[-1, 0])
    assert across == pytest.approx((SIZE - 1) * PIXEL_M, rel=1e-3)
    assert down == pytest.approx((SIZE - 1) * PIXEL_M, rel=1e-3)


# --------------------------------------------------------------------------
# Bilinear sampling
# --------------------------------------------------------------------------


PIXELS = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]])


def test_pixel_centres_reproduce_the_pixels_including_the_last_row_and_column():
    rows, cols = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    np.testing.assert_allclose(sample_bilinear(PIXELS, rows, cols), PIXELS)


def test_between_centres_the_sample_is_interpolated():
    assert sample_bilinear(PIXELS, np.array([0.5]), np.array([0.5]))[0] == pytest.approx(30.0)
    assert sample_bilinear(PIXELS, np.array([0.0]), np.array([1.25]))[0] == pytest.approx(22.5)


def test_samples_outside_the_array_are_no_data():
    values = sample_bilinear(
        PIXELS, np.array([-0.5, 2.5, 1.0, np.nan]), np.array([1.0, 1.0, 3.0, 1.0])
    )
    np.testing.assert_array_equal(values, [0.0, 0.0, 0.0, 0.0])


def test_no_data_is_never_blended_into_a_sample():
    """Zero is no data in GRD products; half a missing pixel is not a darker pixel."""
    pixels = PIXELS.copy()
    pixels[0, 0] = 0.0
    assert sample_bilinear(pixels, np.array([0.5]), np.array([0.5]))[0] == 0.0
    assert sample_bilinear(pixels, np.array([1.5]), np.array([1.5]))[0] == pytest.approx(70.0)


# --------------------------------------------------------------------------
# Fixed dB stretch
# --------------------------------------------------------------------------


def test_the_stretch_is_fixed_and_logarithmic():
    low, high = 30.0, 1500.0
    grey = db_stretch(
        np.array([0.0, 10.0, low, math.sqrt(low * high), high, 9000.0]), low_dn=low, high_dn=high
    )
    assert grey.dtype == np.uint8
    assert grey.tolist() == [0, 1, 1, 128, 255, 255]


def test_the_same_amplitude_gives_the_same_grey_whatever_else_is_in_the_chip():
    """Unlike a percentile stretch, a faint return stays faint next to a bright ship."""
    alone = db_stretch(np.array([200.0, 60.0]), low_dn=30.0, high_dn=1500.0)
    with_ship = db_stretch(np.array([200.0, 60.0, 5000.0]), low_dn=30.0, high_dn=1500.0)
    assert alone.tolist() == with_ship[:2].tolist()


def test_sea_is_dark_and_a_bright_ship_is_white_in_both_polarisations():
    for pol, sea, ship in (("vv", 65.0, 1200.0), ("vh", 33.0, 600.0)):
        low, high = CHIP_STRETCH_DN[pol]
        grey = db_stretch(np.array([sea, ship]), low_dn=low, high_dn=high)
        assert grey[0] < 70 and grey[1] > 200, pol


# --------------------------------------------------------------------------
# PNG encoding
# --------------------------------------------------------------------------


def test_a_grey_png_round_trips():
    grey = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    pixels, transparent = decode_gray_png(encode_gray_png(grey))
    np.testing.assert_array_equal(pixels, grey)
    assert transparent is None


def test_no_data_can_be_written_transparent():
    """So the console can show where the radar did not look, instead of a black 'calm sea'."""
    grey = np.array([[0, 50], [128, 255]], dtype=np.uint8)
    pixels, transparent = decode_gray_png(encode_gray_png(grey, transparent_zero=True))
    np.testing.assert_array_equal(pixels, grey)
    assert transparent == 0


def test_only_2d_bytes_are_encoded():
    with pytest.raises(ValueError):
        encode_gray_png(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_gray_png(np.zeros((2, 2), dtype=np.uint16))


# --------------------------------------------------------------------------
# Cutting a north-up chip from radar geometry
# --------------------------------------------------------------------------


def bands_with_target(scene, grid, gcps, lon, lat) -> dict:
    target = {pixel_of(scene, lon, lat): SHIP_DN}
    return {
        "vv": FakeBand(grid["width"], grid["height"], targets=target, gcps=gcps),
        "vh": FakeBand(grid["width"], grid["height"], sea=30, targets=target, gcps=gcps),
    }


def test_the_detection_is_at_the_centre_of_the_chip(scene, grid, gcps):
    chips = cut(scene, bands_with_target(scene, grid, gcps, LON, LAT))
    assert set(chips) == {"vv", "vh"}
    assert chips["vv"].shape == (SIZE, SIZE)
    row, col = brightest(chips["vv"])
    assert abs(row - 127.5) <= 1.5 and abs(col - 127.5) <= 1.5


@pytest.mark.parametrize(
    ("north_m", "east_m", "expected"),
    [(500.0, 0.0, (77.5, 127.5)), (0.0, 500.0, (127.5, 177.5)), (-800.0, -300.0, (207.5, 97.5))],
)
def test_a_descending_pass_comes_out_north_up_and_east_right(
    scene, grid, gcps, north_m, east_m, expected
):
    """On this pass raw rows run north to south and columns east to west."""
    lon, lat = offset(LON, LAT, north_m=north_m, east_m=east_m)
    chips = cut(scene, bands_with_target(scene, grid, gcps, lon, lat))
    row, col = brightest(chips["vv"])
    assert abs(row - expected[0]) <= 2.0 and abs(col - expected[1]) <= 2.0


def test_each_polarisation_is_read_from_its_own_band(scene, grid, gcps):
    chips = cut(scene, bands_with_target(scene, grid, gcps, LON, LAT))
    assert np.median(chips["vv"]) == pytest.approx(SEA_DN)
    assert np.median(chips["vh"]) == pytest.approx(30)


def test_only_a_small_window_is_read(scene, grid, gcps):
    """One chip must not pull a large part of the scene from object storage."""
    bands = bands_with_target(scene, grid, gcps, LON, LAT)
    cut(scene, bands)
    for band in bands.values():
        assert len(band.windows) == 1
        window = band.windows[0]
        assert window.width <= 400 and window.height <= 400


def test_beyond_the_scene_edge_the_chip_is_no_data(scene, grid, gcps):
    lons, lats = scene.lonlat([8000.0], [3.0])
    bands = {"vv": FakeBand(grid["width"], grid["height"], gcps=gcps)}
    chip = cut(scene, bands, float(lons[0]), float(lats[0]))["vv"]
    missing = float(np.mean(chip == 0))
    assert 0.3 < missing < 0.7


def test_a_point_the_scene_does_not_cover_is_refused(scene, grid, gcps):
    bands = {"vv": FakeBand(grid["width"], grid["height"], gcps=gcps)}
    with pytest.raises(ValueError, match="does not cover"):
        cut(scene, bands, 50.0, 27.5)


class LinearBand:
    """A raster whose DN is linear in the pixel index, so bilinear samples are exact."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.dtypes = ("float64",)

    def read(self, _band, window=None):
        rows = np.arange(window.row_off, window.row_off + window.height, dtype=np.float64)
        cols = np.arange(window.col_off, window.col_off + window.width, dtype=np.float64)
        return 100.0 + 3.0 * rows[:, None] + 7.0 * cols[None, :]


def test_chip_pixels_sample_the_raster_at_their_exact_centres():
    """Pins the half-pixel convention: raster_coords counts from corners, sampling from centres.

    A north-up geographic raster of 10 m pixels (rows run south, columns east)
    with a linear DN: each chip pixel must read the DN at its own centre to
    1e-6, where a half-pixel slip would be off by 5 DN.
    """
    size, extent = 8, 2000
    dlat = math.degrees(PIXEL_M / EARTH_RADIUS_M)
    dlon = math.degrees(PIXEL_M / (EARTH_RADIUS_M * math.cos(math.radians(LAT))))
    lon_min, lat_max = LON - extent / 2 * dlon, LAT + extent / 2 * dlat
    georef = SceneGeoref.from_affine(Affine(dlon, 0, lon_min, 0, -dlat, lat_max), extent, extent)
    chip = cut_north_up(
        {"vv": LinearBand(extent, extent)}, georef, LON, LAT, size_px=size, pixel_m=PIXEL_M
    )["vv"]
    lons, lats = north_up_grid(LON, LAT, size_px=size, pixel_m=PIXEL_M)
    rows = (lat_max - lats) / dlat - 0.5
    cols = (lons - lon_min) / dlon - 0.5
    np.testing.assert_allclose(chip, 100.0 + 3.0 * rows + 7.0 * cols, atol=1e-6)
