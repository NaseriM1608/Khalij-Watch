"""Tests for CDSE catalogue parsing and windowed reading.

Catalogue parsing is tested against a real OData record captured from the live
service, so a schema change upstream shows up here rather than in production.
Windowing is tested against synthetic rasters.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import pytest

from core.db.spatial import HORMUZ_BBOX, BBox
from core.sar.catalogue import (
    PRODUCT_TYPE_GRDH,
    CatalogueError,
    mean_revisit_days,
    product_from_odata,
    search,
)
from core.sar.windows import (
    MAX_WINDOW_BYTES,
    PixelWindow,
    WindowTooLargeError,
    clamp_window,
    estimate_bytes,
    guard_allocation,
    tile_window,
)

# A real record, trimmed, as returned by the live catalogue on 2026-09-10.
REAL_PRODUCT = {
    "Id": "abc12345-0000-1111-2222-333344445555",
    "Name": ("S1D_IW_GRDH_1SDV_20260910T020615_20260910T020640_004509_00860C_D1F0_COG.SAFE"),
    "ContentLength": 1131000000,
    "Online": True,
    "S3Path": "/eodata/Sentinel-1/SAR/IW_GRDH_1S/2026/09/10/S1D_...SAFE",
    "ContentDate": {
        "Start": "2026-09-10T02:06:15.000Z",
        "End": "2026-09-10T02:06:40.000Z",
    },
    "GeoFootprint": {
        "type": "Polygon",
        "coordinates": [
            [
                [58.655457, 25.508358],
                [58.974457, 27.018028],
                [56.419304, 27.435465],
                [56.133514, 25.928946],
                [58.655457, 25.508358],
            ]
        ],
    },
    "Attributes": [
        {"Name": "orbitDirection", "Value": "DESCENDING"},
        {"Name": "timeliness", "Value": "Fast-24h"},
        {"Name": "operationalMode", "Value": "IW"},
    ],
}


# --------------------------------------------------------------------------
# Catalogue parsing
# --------------------------------------------------------------------------


def test_parses_a_real_product_record():
    product = product_from_odata(REAL_PRODUCT)

    assert product.satellite == "S1D"
    assert product.mode == "IW"
    assert product.product_type == "GRDH"
    assert product.polarisation == "DV"
    assert product.is_cog
    assert product.online


def test_dual_polarisation_is_recognised():
    """VV+VH matters: VH is the better ship-detection channel."""
    assert product_from_odata(REAL_PRODUCT).is_dual_pol


def test_single_polarisation_is_not_dual():
    record = dict(REAL_PRODUCT)
    record["Name"] = REAL_PRODUCT["Name"].replace("1SDV", "1SSV")

    assert not product_from_odata(record).is_dual_pol


def test_acquisition_times_are_timezone_aware():
    product = product_from_odata(REAL_PRODUCT)

    assert product.acq_start.tzinfo is not None
    assert product.acq_start == dt.datetime(2026, 9, 10, 2, 6, 15, tzinfo=dt.UTC)


def test_footprint_becomes_a_bounding_box():
    product = product_from_odata(REAL_PRODUCT)

    assert product.bbox.min_lon == pytest.approx(56.133514)
    assert product.bbox.max_lat == pytest.approx(27.435465)
    assert len(product.footprint) == 5  # closed ring


def test_footprint_bbox_overlaps_the_hormuz_aoi():
    """This scene covers the strait, so it must intersect the training AOI."""
    bbox = product_from_odata(REAL_PRODUCT).bbox

    assert bbox.min_lon < HORMUZ_BBOX.max_lon
    assert bbox.max_lon > HORMUZ_BBOX.min_lon


def test_non_cog_variant_is_identified():
    record = dict(REAL_PRODUCT)
    record["Name"] = REAL_PRODUCT["Name"].replace("_COG.SAFE", ".SAFE")

    assert not product_from_odata(record).is_cog


def test_offline_products_are_flagged():
    """An offline product sits in long-term archive and cannot be read."""
    record = dict(REAL_PRODUCT, Online=False)

    assert not product_from_odata(record).online


def test_attributes_are_extracted():
    product = product_from_odata(REAL_PRODUCT)

    assert product.orbit_direction == "DESCENDING"
    assert product.timeliness == "Fast-24h"


def test_record_without_a_name_is_rejected():
    with pytest.raises(CatalogueError, match="no Name"):
        product_from_odata({"Id": "x"})


def test_record_without_a_footprint_is_rejected():
    record = dict(REAL_PRODUCT)
    record.pop("GeoFootprint")

    with pytest.raises(CatalogueError, match="footprint"):
        product_from_odata(record)


def test_malformed_records_are_skipped_not_fatal():
    """One bad record must not lose the rest of the page."""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"value": [{"Id": "broken"}, REAL_PRODUCT]}).encode()

    products = search(
        HORMUZ_BBOX,
        since=dt.datetime(2026, 9, 1, tzinfo=dt.UTC),
        opener=lambda *_a, **_k: FakeResponse(),
    )

    assert len(products) == 1
    assert products[0].satellite == "S1D"


def test_cog_filter_excludes_the_safe_variant():
    safe = dict(REAL_PRODUCT)
    safe["Name"] = REAL_PRODUCT["Name"].replace("_COG.SAFE", ".SAFE")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"value": [safe, REAL_PRODUCT]}).encode()

    opener = lambda *_a, **_k: FakeResponse()  # noqa: E731
    cog_only = search(HORMUZ_BBOX, since=dt.datetime(2026, 9, 1, tzinfo=dt.UTC), opener=opener)
    everything = search(
        HORMUZ_BBOX,
        since=dt.datetime(2026, 9, 1, tzinfo=dt.UTC),
        cog_only=False,
        opener=opener,
    )

    assert len(cog_only) == 1
    assert len(everything) == 2


def test_mean_revisit_collapses_products_from_one_pass():
    def at(hours: float):
        record = dict(REAL_PRODUCT)
        stamp = dt.datetime(2026, 9, 1, tzinfo=dt.UTC) + dt.timedelta(hours=hours)
        record["ContentDate"] = {
            "Start": stamp.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "End": stamp.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }
        return product_from_odata(record)

    # Two products from one pass, then another pass two days later.
    products = [at(0), at(0.005), at(48)]

    assert mean_revisit_days(products) == pytest.approx(2.0, abs=0.01)


# --------------------------------------------------------------------------
# Memory guard
# --------------------------------------------------------------------------


def test_a_full_scene_would_be_refused():
    """The whole point: a 25000x17000 float32 read must never be attempted."""
    full_scene = PixelWindow(0, 0, 25_000, 17_000)

    with pytest.raises(WindowTooLargeError, match="refusing to allocate"):
        guard_allocation(full_scene.nbytes(dtype_size=4))


def test_a_reasonable_window_is_allowed():
    guard_allocation(PixelWindow(0, 0, 2048, 2048).nbytes(dtype_size=4))


def test_the_guard_reports_the_limit():
    with pytest.raises(WindowTooLargeError, match="limit"):
        guard_allocation(MAX_WINDOW_BYTES + 1)


def test_estimate_bytes_scales_with_area():
    small = estimate_bytes(BBox(56.0, 26.0, 56.1, 26.1))
    large = estimate_bytes(BBox(56.0, 26.0, 57.0, 27.0))

    assert large > small * 50


def test_whole_gulf_at_full_resolution_exceeds_the_guard():
    """Sanity: the AOI cannot be read in one go, which is why tiling exists."""
    from core.db.spatial import GULF_BBOX

    assert estimate_bytes(GULF_BBOX) > MAX_WINDOW_BYTES


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_clamp_keeps_a_window_inside_the_raster():
    clamped = clamp_window(PixelWindow(-100, -50, 500, 500), 1000, 1000)

    assert clamped.col_off == 0
    assert clamped.row_off == 0
    assert clamped.width == 400
    assert clamped.height == 450


def test_clamp_truncates_at_the_far_edge():
    clamped = clamp_window(PixelWindow(900, 900, 500, 500), 1000, 1000)

    assert clamped.width == 100
    assert clamped.height == 100


def test_clamp_of_a_fully_outside_window_is_empty():
    assert clamp_window(PixelWindow(5000, 5000, 100, 100), 1000, 1000).is_empty()


def test_tiles_cover_the_whole_window():
    window = PixelWindow(0, 0, 5000, 4000)
    tiles = tile_window(window, tile_size=2048, overlap=512)

    assert tiles
    assert max(t.col_off + t.width for t in tiles) == 5000
    assert max(t.row_off + t.height for t in tiles) == 4000


def test_tiles_overlap_by_the_requested_amount():
    tiles = tile_window(PixelWindow(0, 0, 5000, 2048), tile_size=2048, overlap=512)
    first_row = sorted({t.col_off for t in tiles})

    # stride = tile_size - overlap = 1536
    assert first_row[1] - first_row[0] == 1536


def test_tiling_an_empty_window_returns_nothing():
    assert tile_window(PixelWindow(0, 0, 0, 0)) == []


def test_tile_parameters_are_validated():
    with pytest.raises(ValueError, match="tile_size"):
        tile_window(PixelWindow(0, 0, 100, 100), tile_size=0)
    with pytest.raises(ValueError, match="overlap"):
        tile_window(PixelWindow(0, 0, 100, 100), tile_size=256, overlap=256)


def test_every_tile_is_within_the_memory_guard():
    """Tiling is what makes an unreadable scene readable."""
    tiles = tile_window(PixelWindow(0, 0, 25_000, 17_000), tile_size=2048, overlap=512)

    assert len(tiles) > 100
    for tile in tiles:
        guard_allocation(tile.nbytes(dtype_size=4, bands=2))


# --------------------------------------------------------------------------
# Live catalogue
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_catalogue_returns_cog_products_over_hormuz():
    """Needs no credentials: catalogue search is public."""
    products = search(
        HORMUZ_BBOX,
        since=dt.datetime.now(dt.UTC) - dt.timedelta(days=20),
        product_type=PRODUCT_TYPE_GRDH,
    )

    assert products, "no Sentinel-1 GRDH COG products over Hormuz in 20 days"
    # Post 2026-06-29 the constellation is S1C + S1D only.
    assert {p.satellite for p in products} <= {"S1C", "S1D"}
    assert all(p.is_cog for p in products)
    assert all(p.is_dual_pol for p in products)
    assert all(p.mode == "IW" for p in products)


@pytest.mark.network
def test_live_products_have_usable_footprints_and_paths():
    products = search(HORMUZ_BBOX, since=dt.datetime.now(dt.UTC) - dt.timedelta(days=20))

    product = products[0]
    assert product.s3_path and product.s3_path.startswith("/eodata/")
    assert product.size_bytes > 100_000_000
    assert not math.isnan(product.bbox.min_lon)
