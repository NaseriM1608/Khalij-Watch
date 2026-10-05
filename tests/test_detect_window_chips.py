"""The chip step of scripts/detect_window.py, unstubbed: each band's pixels go to its own chip.

detect_window holds VV and VH as one (2, rows, cols) stack from read_pair; a
swapped band would write VH pixels under VV's stretch, and a failure must
never fail the detection run whose detections are already stored.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from core.fusion.chips import CHIP_STRETCH_DN, chip_relpath  # noqa: E402
from core.sar.display import db_stretch  # noqa: E402
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.reader import S3AccessError  # noqa: E402
from core.sar.windows import PixelWindow  # noqa: E402
from scripts import cut_chips  # noqa: E402
from scripts.detect_window import cut_found_chips  # noqa: E402
from tests.chip_fakes import LAT, LON, SCENE, load_gcps, load_grid  # noqa: E402

VV_DN, VH_DN = 1000, 20


@pytest.fixture
def held():
    grid = load_grid()
    georef = SceneGeoref(width=grid["width"], height=grid["height"], gcps=load_gcps(grid))
    rows, cols = georef.raster_coords([LON], [LAT])
    window = PixelWindow(int(cols[0]) - 400, int(rows[0]) - 400, 800, 800)
    bands = np.stack(
        [np.full((800, 800), VV_DN, dtype=np.uint16), np.full((800, 800), VH_DN, dtype=np.uint16)]
    )
    return georef, window, bands


def grey_of(root, pol):
    from tests.chip_fakes import decode_gray_png

    grey, _ = decode_gray_png((root / chip_relpath(SCENE, LON, LAT, pol)).read_bytes())
    return set(np.unique(grey).tolist())


def test_each_band_of_the_stack_goes_to_its_own_chip(tmp_path, held):
    georef, window, bands = held
    product = SimpleNamespace(name=SCENE, s3_path="/eodata/x")
    cut_found_chips([SimpleNamespace(lon=LON, lat=LAT)], georef, window, bands, product, tmp_path)
    for pol, dn in (("vv", VV_DN), ("vh", VH_DN)):
        low, high = CHIP_STRETCH_DN[pol]
        assert grey_of(tmp_path, pol) == {int(db_stretch([dn], low_dn=low, high_dn=high)[0])}, pol


def test_a_chip_failure_never_fails_the_detection_run(tmp_path, held, monkeypatch, capsys):
    georef, window, bands = held

    @contextlib.contextmanager
    def unreachable(*_args, **_kwargs):
        raise S3AccessError("network", "down")
        yield

    monkeypatch.setattr(cut_chips, "open_scene_bands", unreachable)
    product = SimpleNamespace(name=SCENE, s3_path="/eodata/x")
    beyond = SimpleNamespace(lon=56.75, lat=25.45)  # outside the 800 px held window
    cut_found_chips(
        [SimpleNamespace(lon=LON, lat=LAT), beyond], georef, window, bands, product, tmp_path
    )
    assert "1 from the pixels read, 0 read again" in capsys.readouterr().out
    assert len(list(tmp_path.rglob("*.png"))) == 2
