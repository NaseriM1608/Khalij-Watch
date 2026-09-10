"""Windowed reading against a real Cloud-Optimized GeoTIFF.

The guard tests elsewhere are pure arithmetic. These build an actual
georeferenced COG on disk and read through it, which is what proves the
window maths, the CRS handling and the memory ceiling all line up.
"""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_bounds as transform_from_bounds  # noqa: E402

from core.db.spatial import BBox  # noqa: E402
from core.sar.windows import (  # noqa: E402
    PixelWindow,
    WindowTooLargeError,
    bbox_to_window,
    guard_allocation,
    normalise_sar,
    read_window,
    tile_window,
)

# A synthetic scene over the strait, at a coarse resolution so the test is fast.
SCENE_BBOX = BBox(56.0, 26.0, 57.0, 27.0)
SIZE = 1000


@pytest.fixture(scope="module")
def cog_path(tmp_path_factory):
    """A tiled GeoTIFF whose pixel values encode their own column index."""
    path = tmp_path_factory.mktemp("sar") / "scene.tif"
    transform = transform_from_bounds(
        SCENE_BBOX.min_lon,
        SCENE_BBOX.min_lat,
        SCENE_BBOX.max_lon,
        SCENE_BBOX.max_lat,
        SIZE,
        SIZE,
    )
    # Column index in band 1 makes it trivial to prove a window landed where
    # it was asked to.
    data = np.tile(np.arange(SIZE, dtype=np.uint16), (SIZE, 1))

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=2,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    ) as dataset:
        dataset.write(data, 1)
        dataset.write(data // 2, 2)  # stand-in for the second polarisation
    return path


def test_full_extent_maps_to_the_whole_raster(cog_path):
    with rasterio.open(cog_path) as dataset:
        window = bbox_to_window(dataset, SCENE_BBOX)

    assert window.col_off == 0
    assert window.row_off == 0
    assert window.width == SIZE
    assert window.height == SIZE


def test_a_sub_bbox_maps_to_the_right_pixels(cog_path):
    """The eastern half of the scene must start halfway across the raster."""
    eastern_half = BBox(56.5, 26.0, 57.0, 27.0)

    with rasterio.open(cog_path) as dataset:
        window = bbox_to_window(dataset, eastern_half)

    assert window.col_off == pytest.approx(SIZE // 2, abs=1)
    assert window.width == pytest.approx(SIZE // 2, abs=1)


def test_reading_a_window_returns_only_that_window(cog_path):
    eastern_half = BBox(56.5, 26.0, 57.0, 27.0)

    with rasterio.open(cog_path) as dataset:
        window = bbox_to_window(dataset, eastern_half)
        data = read_window(dataset, window)

    assert data.shape == (window.height, window.width)
    # Values encode column index, so the eastern half starts at ~500.
    assert data.min() >= SIZE // 2 - 2
    assert data.max() <= SIZE - 1


def test_both_polarisations_are_readable(cog_path):
    """Sentinel-1 Gulf scenes are dual-pol; VH is the better ship channel."""
    small = BBox(56.1, 26.1, 56.2, 26.2)

    with rasterio.open(cog_path) as dataset:
        window = bbox_to_window(dataset, small)
        vv = read_window(dataset, window, band=1)
        vh = read_window(dataset, window, band=2)

    assert vv.shape == vh.shape
    assert np.all(vh == vv // 2)


def test_an_aoi_overhanging_the_scene_is_clipped(cog_path):
    """A scene clips the AOI edge constantly; that must not read out of bounds."""
    overhanging = BBox(56.5, 26.5, 60.0, 30.0)

    with rasterio.open(cog_path) as dataset:
        window = bbox_to_window(dataset, overhanging)
        data = read_window(dataset, window)

    assert window.col_off + window.width <= SIZE
    assert window.row_off + window.height <= SIZE
    assert data.size > 0


def test_an_aoi_entirely_outside_the_scene_is_empty(cog_path):
    with rasterio.open(cog_path) as dataset:
        window = bbox_to_window(dataset, BBox(0.0, 50.0, 1.0, 51.0))

    assert window.is_empty()
    with pytest.raises(ValueError, match="empty window"), rasterio.open(cog_path) as dataset:
        read_window(dataset, window)


def test_reading_a_window_never_loads_the_whole_raster(cog_path):
    """The governing constraint: window-sized memory, not scene-sized."""
    small = BBox(56.0, 26.0, 56.05, 26.05)

    with rasterio.open(cog_path) as dataset:
        window = bbox_to_window(dataset, small)
        data = read_window(dataset, window)

    assert data.nbytes < 100_000
    assert data.size < SIZE * SIZE // 100


def test_an_oversized_read_is_refused_before_allocating():
    """A real IW GRD scene is ~25000x17000; that read must never be attempted."""
    with pytest.raises(WindowTooLargeError):
        guard_allocation(PixelWindow(0, 0, 25_000, 17_000).nbytes(dtype_size=4))


def test_tiles_of_a_real_window_are_all_readable(cog_path):
    with rasterio.open(cog_path) as dataset:
        window = bbox_to_window(dataset, SCENE_BBOX)
        tiles = tile_window(window, tile_size=256, overlap=64)
        assert len(tiles) > 1
        for tile in tiles[:5]:
            data = read_window(dataset, tile)
            assert data.shape == (tile.height, tile.width)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_normalisation_maps_into_the_unit_interval():
    values = np.array([[1, 300, 5000]], dtype=np.uint16)

    result = normalise_sar(values)

    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_normalisation_is_monotonic():
    """Brighter returns must stay brighter after mapping."""
    values = np.array([[100, 300, 900, 3000]], dtype=np.uint16)

    result = normalise_sar(values)[0]

    assert list(result) == sorted(result)


def test_missing_data_stays_zero():
    """GRD encodes no-data as 0, and it must not become mid-grey."""
    values = np.array([[0, 0, 800]], dtype=np.uint16)

    result = normalise_sar(values)

    assert result[0, 0] == 0.0
    assert result[0, 1] == 0.0
    assert result[0, 2] > 0.5


def test_ships_land_in_the_responsive_part_of_the_curve():
    """Sea saturates low, a bright target sits well above it.

    The point of the sigmoid over linear scaling: contrast is spent where
    ships are, not on the vast dynamic range of open water.
    """
    sea = normalise_sar(np.array([[80]], dtype=np.uint16))[0, 0]
    ship = normalise_sar(np.array([[1200]], dtype=np.uint16))[0, 0]

    assert sea < 0.4
    assert ship > 0.8
    assert ship - sea > 0.5
