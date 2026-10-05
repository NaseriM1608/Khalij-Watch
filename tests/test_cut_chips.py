"""Tests for cutting the radar chips of stored dark-vessel candidates.

One scene is opened per pass and one small window read per candidate and
polarisation; each chip is written under the chip folder and recorded on the
event's evidence card, so the API can serve it without touching CDSE. A
candidate whose chip cannot be cut keeps its card as it was: a missing image
is shown as missing, never as an empty sea. Cards are written back only if
they did not change meanwhile (a re-fusion can reuse event ids), and chips
already on disk are reused for re-fused events without a CDSE read.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

rasterio = pytest.importorskip("rasterio")

from core.db.repository import (  # noqa: E402
    DetectionRepository,
    EventRepository,
    NewEvent,
    SceneRepository,
)
from core.db.session import (  # noqa: E402
    create_all,
    create_db_engine,
    make_session_factory,
    session_scope,
)
from core.db.spatial import BBox  # noqa: E402
from core.fusion.chips import CHIP_STRETCH_DN, chip_relpath, image_meta  # noqa: E402
from core.sar.display import db_stretch  # noqa: E402
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.reader import S3AccessError  # noqa: E402
from core.sar.windows import PixelWindow  # noqa: E402
from scripts.cut_chips import (  # noqa: E402
    ChipResult,
    HeldBand,
    cut_detection_chips,
    cut_event_chips,
    cut_stored_detection_chips,
    open_scene_bands,
)
from tests.chip_fakes import (  # noqa: E402
    LAT,
    LON,
    SCENE,
    SEA_DN,
    FakeBand,
    decode_gray_png,
    load_gcps,
    load_grid,
)

ACQ = dt.datetime(2026, 9, 10, 2, 6, 40, tzinfo=dt.UTC)
S3_PATH = f"/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/2026/09/10/{SCENE}"
RUN = "fusion-v1"
NEAR = (56.60, 25.30)
OFF_SCENE = (50.0, 27.5)
VH_SEA_DN = 30


@dataclass(frozen=True)
class Extent:
    min_lon: float = 55.8
    min_lat: float = 24.0
    max_lon: float = 58.7
    max_lat: float = 26.0


@dataclass(frozen=True)
class Product:
    """Shaped like core.sar.catalogue.SarProduct."""

    product_id: str = SCENE
    name: str = SCENE
    acq_start: dt.datetime = ACQ
    satellite: str = "S1D"
    mode: str = "IW"
    polarisation: str = "DV"
    orbit_direction: str = "DESCENDING"
    s3_path: str | None = S3_PATH
    footprint: tuple = ((55.8, 24.0), (58.7, 24.0), (58.7, 26.0), (55.8, 26.0), (55.8, 24.0))
    bbox: Extent = Extent()


def dark(lon: float, lat: float) -> NewEvent:
    return NewEvent(
        event_type="DARK_VESSEL",
        ts=ACQ,
        lon=lon,
        lat=lat,
        severity="low",
        detection_id=None,
        evidence={"status": "dark", "chip": {"scene": SCENE, "lon": lon, "lat": lat}},
    )


def make_factory(positions, *, s3_path: str | None = S3_PATH):
    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        SceneRepository(session).upsert(Product(s3_path=s3_path))
        scene_id = SceneRepository(session).id_for(SCENE)
        repo = EventRepository(session)
        repo.replace_for_scene(scene_id, RUN, [dark(*p) for p in positions])
        repo.replace_for_scene(scene_id, "fusion-v0", [dark(*NEAR)])
    return factory, scene_id


def cards(factory, scene_id, run: str = RUN) -> list[dict]:
    with session_scope(factory) as session:
        return [e.evidence for e in EventRepository(session).for_scene(scene_id, run)]


def refuse(factory, scene_id, positions) -> None:
    """Fuse the pass again: new events (SQLite may reuse the ids) with fresh cards."""
    with session_scope(factory) as session:
        EventRepository(session).replace_for_scene(scene_id, RUN, [dark(*p) for p in positions])


def layout(image: dict) -> dict:
    """The card's chip metadata without the per-cut fields."""
    return {k: v for k, v in image.items() if k not in ("cut_at", "no_data")}


def chip_path(root, card, pol):
    chip = card["chip"]
    return root / chip_relpath(SCENE, chip["lon"], chip["lat"], pol)


class Opener:
    """Stands in for opening a scene's bands on CDSE."""

    def __init__(self, pols=("vv", "vh"), error: Exception | None = None):
        grid = load_grid()
        gcps = load_gcps(grid)
        sea = {"vv": SEA_DN, "vh": VH_SEA_DN}
        self.bands = {
            p: FakeBand(grid["width"], grid["height"], sea=sea[p], gcps=gcps) for p in pols
        }
        self.error = error
        self.calls = []
        self.while_open = None

    @contextlib.contextmanager
    def __call__(self, name: str, s3_path: str):
        self.calls.append((name, s3_path))
        if self.error is not None:
            raise self.error
        if self.while_open is not None:
            self.while_open()
        yield self.bands


def test_chips_are_cut_written_and_recorded_on_the_cards(tmp_path):
    factory, scene_id = make_factory([(LON, LAT), NEAR])
    opener = Opener()
    result = cut_event_chips(factory, scene_id, root=tmp_path, open_scene=opener)
    assert result == ChipResult(cut=2, skipped=0, failed=0)
    assert opener.calls == [(SCENE, S3_PATH)]
    for card in cards(factory, scene_id):
        image = card["chip"]["image"]
        assert layout(image) == layout(image_meta(["vv", "vh"]))
        assert image["no_data"] == 0.0
        mtime = dt.datetime.fromtimestamp(chip_path(tmp_path, card, "vv").stat().st_mtime, dt.UTC)
        assert dt.datetime.fromisoformat(image["cut_at"]) == mtime
        assert card["status"] == "dark"
        # Each polarisation's own sea level through its own stretch.
        for pol, sea in (("vv", SEA_DN), ("vh", VH_SEA_DN)):
            grey, transparent = decode_gray_png(chip_path(tmp_path, card, pol).read_bytes())
            expected = db_stretch(
                [sea], low_dn=CHIP_STRETCH_DN[pol][0], high_dn=CHIP_STRETCH_DN[pol][1]
            )
            assert (grey == expected[0]).all(), pol
            assert transparent == 0
    assert not [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix != ".png"]
    # Another run's events of the same pass are left alone.
    assert "image" not in cards(factory, scene_id, "fusion-v0")[0]["chip"]


def test_a_second_run_cuts_nothing_and_opens_no_scene(tmp_path):
    factory, scene_id = make_factory([(LON, LAT)])
    cut_event_chips(factory, scene_id, root=tmp_path, open_scene=Opener())
    again = Opener()
    assert cut_event_chips(factory, scene_id, root=tmp_path, open_scene=again) == ChipResult(
        cut=0, skipped=1, failed=0
    )
    assert again.calls == []


def test_force_cuts_every_chip_again(tmp_path):
    factory, scene_id = make_factory([(LON, LAT)])
    cut_event_chips(factory, scene_id, root=tmp_path, open_scene=Opener())
    again = Opener()
    result = cut_event_chips(factory, scene_id, root=tmp_path, open_scene=again, force=True)
    assert result == ChipResult(cut=1, skipped=0, failed=0)
    assert len(again.calls) == 1


def test_a_candidate_the_scene_does_not_cover_fails_alone(tmp_path):
    factory, scene_id = make_factory([(LON, LAT), OFF_SCENE])
    result = cut_event_chips(factory, scene_id, root=tmp_path, open_scene=Opener())
    assert result == ChipResult(cut=1, skipped=0, failed=1)
    inside, outside = cards(factory, scene_id)
    assert inside["chip"]["image"]["pols"] == ["vv", "vh"]
    assert "image" not in outside["chip"]
    assert len(list(tmp_path.rglob("*.png"))) == 2


def test_a_scene_without_vh_gets_vv_chips_only(tmp_path):
    factory, scene_id = make_factory([(LON, LAT)])
    cut_event_chips(factory, scene_id, root=tmp_path, open_scene=Opener(pols=("vv",)))
    (card,) = cards(factory, scene_id)
    assert card["chip"]["image"]["pols"] == ["vv"]
    assert [p.name.rsplit("_", 1)[-1] for p in tmp_path.rglob("*.png")] == ["vv.png"]


def test_a_scene_that_cannot_be_opened_raises_and_records_nothing(tmp_path):
    factory, scene_id = make_factory([(LON, LAT)])
    with pytest.raises(S3AccessError):
        cut_event_chips(
            factory, scene_id, root=tmp_path, open_scene=Opener(error=S3AccessError("network"))
        )
    assert "image" not in cards(factory, scene_id)[0]["chip"]
    assert not list(tmp_path.rglob("*"))


def test_re_fused_events_reuse_the_chips_on_disk_without_cdse(tmp_path):
    factory, scene_id = make_factory([(LON, LAT), NEAR])
    cut_event_chips(factory, scene_id, root=tmp_path, open_scene=Opener())
    refuse(factory, scene_id, [(LON, LAT), NEAR])
    offline = Opener(error=S3AccessError("network"))
    result = cut_event_chips(factory, scene_id, root=tmp_path, open_scene=offline)
    assert result == ChipResult(cut=0, skipped=0, failed=0, reused=2)
    assert offline.calls == []
    for card in cards(factory, scene_id):
        assert card["chip"]["image"]["pols"] == ["vv", "vh"]


def test_force_cuts_again_even_when_chips_are_on_disk(tmp_path):
    factory, scene_id = make_factory([(LON, LAT)])
    cut_event_chips(factory, scene_id, root=tmp_path, open_scene=Opener())
    refuse(factory, scene_id, [(LON, LAT)])
    with pytest.raises(S3AccessError):
        cut_event_chips(
            factory,
            scene_id,
            root=tmp_path,
            open_scene=Opener(error=S3AccessError("network")),
            force=True,
        )


def test_a_card_that_changed_while_cutting_is_not_overwritten(tmp_path):
    """A re-fusion during the CDSE reads can give the old event ids to other detections."""
    factory, scene_id = make_factory([(LON, LAT), NEAR])
    opener = Opener()
    opener.while_open = lambda: refuse(factory, scene_id, [(56.59, 25.29), (56.61, 25.31)])
    result = cut_event_chips(factory, scene_id, root=tmp_path, open_scene=opener)
    assert (result.cut, result.failed) == (0, 2)
    for card in cards(factory, scene_id):
        assert "image" not in card["chip"]
        assert (card["chip"]["lon"], card["chip"]["lat"]) in [(56.59, 25.29), (56.61, 25.31)]


def test_a_chip_that_cannot_be_written_fails_alone(tmp_path):
    factory, scene_id = make_factory([(LON, LAT), NEAR])
    blocked = tmp_path / chip_relpath(SCENE, NEAR[0], NEAR[1], "vv")
    blocked.mkdir(parents=True)  # a folder where the file should go
    result = cut_event_chips(factory, scene_id, root=tmp_path, open_scene=Opener())
    assert (result.cut, result.failed) == (1, 1)
    first, second = cards(factory, scene_id)
    assert first["chip"]["image"]["pols"] == ["vv", "vh"]
    assert "image" not in second["chip"]
    assert not [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix != ".png"]


def test_a_scene_without_an_image_path_is_an_error(tmp_path):
    factory, scene_id = make_factory([(LON, LAT)], s3_path=None)
    with pytest.raises(ValueError, match="image path"):
        cut_event_chips(factory, scene_id, root=tmp_path, open_scene=Opener())


# --------------------------------------------------------------------------
# Opening a scene's bands
# --------------------------------------------------------------------------


def fake_opener(missing=(), error: Exception | None = None):
    opened = []

    @contextlib.contextmanager
    def opener(product, polarisation="vv"):
        if polarisation in missing:
            raise S3AccessError("not_found", f"no {polarisation}")
        if error is not None and polarisation == "vh":
            raise error
        opened.append((product.name, product.s3_path, polarisation))
        yield SimpleNamespace(pol=polarisation)

    return opener, opened


def test_both_polarisations_are_opened():
    opener, opened = fake_opener()
    with open_scene_bands(SCENE, S3_PATH, opener=opener) as bands:
        assert {p: b.pol for p, b in bands.items()} == {"vv": "vv", "vh": "vh"}
    assert opened == [(SCENE, S3_PATH, "vv"), (SCENE, S3_PATH, "vh")]


def test_a_scene_without_a_vh_band_is_opened_with_vv_only():
    opener, _ = fake_opener(missing=("vh",))
    with open_scene_bands(SCENE, S3_PATH, opener=opener) as bands:
        assert list(bands) == ["vv"]


def test_a_scene_without_a_vv_band_is_an_error():
    opener, _ = fake_opener(missing=("vv",))
    with pytest.raises(S3AccessError), open_scene_bands(SCENE, S3_PATH, opener=opener):
        pass


def test_other_failures_opening_vh_are_not_hidden():
    opener, _ = fake_opener(error=S3AccessError("wrong_secret"))
    with (
        pytest.raises(S3AccessError, match="wrong_secret"),
        open_scene_bands(SCENE, S3_PATH, opener=opener),
    ):
        pass


# --------------------------------------------------------------------------
# Every model detection's image, cut while it is found
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """Shaped like a stored detection target (core.inference.runner.PointDetection)."""

    lon: float
    lat: float
    confidence: float = 0.9
    length_m: float | None = None
    width_m: float | None = None
    heading_deg: float | None = None
    pixel_count: int | None = None
    peak_amplitude: float | None = None


def held_around(lon: float, lat: float, half_px: int) -> tuple[dict, SceneGeoref]:
    """Both bands as detect_window holds them: one window read around a point."""
    grid = load_grid()
    georef = SceneGeoref(width=grid["width"], height=grid["height"], gcps=load_gcps(grid))
    rows, cols = georef.raster_coords([lon], [lat])
    window = PixelWindow(int(cols[0]) - half_px, int(rows[0]) - half_px, 2 * half_px, 2 * half_px)
    held = {
        pol: HeldBand(
            FakeBand(grid["width"], grid["height"], sea=sea).read(1, window),
            row_off=window.row_off,
            col_off=window.col_off,
        )
        for pol, sea in (("vv", SEA_DN), ("vh", VH_SEA_DN))
    }
    return held, georef


def test_a_detections_image_is_cut_from_the_pixels_already_read(tmp_path):
    held, georef = held_around(LON, LAT, half_px=400)
    opener = Opener()
    done = cut_detection_chips(
        [(LON, LAT)],
        held,
        georef,
        root=tmp_path,
        scene=SCENE,
        open_scene=lambda: opener(SCENE, S3_PATH),
    )
    assert (done.from_memory, done.read, done.failed) == (1, 0, 0)
    assert opener.calls == []
    for pol, sea in (("vv", SEA_DN), ("vh", VH_SEA_DN)):
        grey, _ = decode_gray_png((tmp_path / chip_relpath(SCENE, LON, LAT, pol)).read_bytes())
        low, high = CHIP_STRETCH_DN[pol]
        assert (grey == db_stretch([sea], low_dn=low, high_dn=high)[0]).all(), pol


def test_images_reaching_beyond_the_pixels_read_are_read_from_the_scene_once(tmp_path):
    held, georef = held_around(LON, LAT, half_px=100)  # smaller than an image
    opener = Opener()
    near = (56.585, 25.29)
    done = cut_detection_chips(
        [(LON, LAT), near],
        held,
        georef,
        root=tmp_path,
        scene=SCENE,
        open_scene=lambda: opener(SCENE, S3_PATH),
    )
    assert (done.from_memory, done.read, done.failed) == (0, 2, 0)
    assert len(opener.calls) == 1
    assert len(list(tmp_path.rglob("*.png"))) == 4


def test_an_image_that_cannot_be_cut_does_not_stop_the_others(tmp_path):
    held, georef = held_around(LON, LAT, half_px=400)
    done = cut_detection_chips(
        [OFF_SCENE, (LON, LAT)],
        held,
        georef,
        root=tmp_path,
        scene=SCENE,
        open_scene=lambda: Opener()(SCENE, S3_PATH),
    )
    assert (done.from_memory, done.failed) == (1, 1)


def store_points(factory, scene_id, points, version="kaggle-v2"):
    with session_scope(factory) as session:
        DetectionRepository(session).replace_in_bbox(
            scene_id,
            BBox(55.0, 24.0, 58.0, 27.0),
            points,
            polarisation="DV",
            detector="pointdet",
            detector_version=version,
            confidence=0.0,
        )


def test_stored_product_detections_get_their_images_once(tmp_path):
    factory, scene_id = make_factory([])
    store_points(factory, scene_id, [Point(LON, LAT), Point(*NEAR)])
    opener = Opener()
    result = cut_stored_detection_chips(factory, scene_id, root=tmp_path, open_scene=opener)
    assert result == ChipResult(cut=2, skipped=0, failed=0)
    assert len(opener.calls) == 1
    assert len(list(tmp_path.rglob("*.png"))) == 4
    again = Opener()
    assert cut_stored_detection_chips(
        factory, scene_id, root=tmp_path, open_scene=again
    ) == ChipResult(cut=0, skipped=2, failed=0)
    assert again.calls == []


def test_only_the_product_models_detections_get_images(tmp_path):
    factory, scene_id = make_factory([])
    store_points(factory, scene_id, [Point(LON, LAT)], version="kaggle-v1")
    opener = Opener()
    result = cut_stored_detection_chips(factory, scene_id, root=tmp_path, open_scene=opener)
    assert result == ChipResult(cut=0, skipped=0, failed=0)
    assert opener.calls == []


def test_force_cuts_stored_detection_chips_again(tmp_path):
    factory, scene_id = make_factory([])
    store_points(factory, scene_id, [Point(LON, LAT)])
    cut_stored_detection_chips(factory, scene_id, root=tmp_path, open_scene=Opener())
    again = Opener()
    result = cut_stored_detection_chips(
        factory, scene_id, root=tmp_path, open_scene=again, force=True
    )
    assert result == ChipResult(cut=1, skipped=0, failed=0)
    assert len(again.calls) == 1


def test_a_detection_with_only_its_vv_chip_gets_its_vh_chip(tmp_path):
    factory, scene_id = make_factory([])
    store_points(factory, scene_id, [Point(LON, LAT)])
    cut_stored_detection_chips(factory, scene_id, root=tmp_path, open_scene=Opener(pols=("vv",)))
    assert not (tmp_path / chip_relpath(SCENE, LON, LAT, "vh")).exists()
    result = cut_stored_detection_chips(factory, scene_id, root=tmp_path, open_scene=Opener())
    assert result.cut == 1
    assert (tmp_path / chip_relpath(SCENE, LON, LAT, "vh")).is_file()
