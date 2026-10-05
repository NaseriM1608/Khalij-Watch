"""Satellite-image chips: of dark-vessel candidates, and of the product model's detections.

Each vessel the deep-learning model finds in a Sentinel-1 image is shown with
its image beside it (Phase 7). scripts/detect_window.py cuts these chips while
it detects, from the pixels it has already read (cut_detection_chips); with
--detections this script cuts them for stored product-model detections that
have none (one scene opened per pass).

For each DARK_VESSEL event of a pass without a chip (every one with --force),
cuts a north-up VV and VH image 2.56 km across around the detection from the
pass's COG on CDSE - one scene opened per pass, one small window read per
candidate and polarisation - writes them under <data_dir>/chips/ (pixels the
radar did not image are transparent) and records them on the event's
evidence card, so the API serves them from disk. Chips are named by scene and
position, so the events of a re-fused pass take up the chips already on disk
without a CDSE read (unless --force). A card is written back only if it did
not change meanwhile. scripts/fuse_scene.py runs this after storing a pass's
events; run it alone to retry or to backfill older events. Needs CDSE S3 keys
in .env.

On this host a process that has used GDAL can fail to exit after a failed
request (core/sar/reader.py): if the script hangs after its summary line, its
work is done and stored, and the process can be ended.

    uv run --offline --python 3.12 --no-project --with numpy --with rasterio --with boto3 \
        --with sqlalchemy python -u scripts/cut_chips.py [--scene NAME] [--force] [--detections]
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.areas import PRODUCT_DETECTOR, PRODUCT_VERSION  # noqa: E402
from core.db.repository import (  # noqa: E402
    DetectionRepository,
    EventRepository,
    EventView,
    SceneRepository,
)
from core.db.session import session_scope  # noqa: E402
from core.db.spatial import BBox  # noqa: E402
from core.fusion.chips import (  # noqa: E402
    CHIP_PIXEL_M,
    CHIP_PX,
    CHIP_STRETCH_DN,
    chip_file_at,
    chip_relpath,
    image_meta,
    stored_image,
    with_chip_image,
)
from core.sar.display import cut_north_up, db_stretch, encode_gray_png  # noqa: E402
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.reader import S3AccessError  # noqa: E402
from core.sar.windows import PixelWindow  # noqa: E402

RUN = "fusion-v1"
DARK_VESSEL = "DARK_VESSEL"
# Why one chip may fail while the others go on.
CHIP_ERRORS = (KeyError, TypeError, ValueError, OSError, S3AccessError)


@dataclass(frozen=True)
class ChipResult:
    cut: int
    skipped: int
    failed: int
    reused: int = 0


@dataclass(frozen=True)
class DetectionChips:
    from_memory: int
    read: int
    failed: int


class NotHeldError(LookupError):
    """The pixels a chip needs reach beyond those already read."""


@dataclass(frozen=True)
class HeldBand:
    """One band's pixels already read for detection, at their raster offset."""

    pixels: Any
    row_off: int
    col_off: int


def _read_held(band: HeldBand, window: PixelWindow) -> Any:
    """The window from pixels already read; NotHeldError if it reaches beyond them."""
    height, width = band.pixels.shape
    row0, col0 = window.row_off - band.row_off, window.col_off - band.col_off
    if row0 < 0 or col0 < 0 or row0 + window.height > height or col0 + window.width > width:
        raise NotHeldError(f"window {window} lies beyond the pixels read")
    return band.pixels[row0 : row0 + window.height, col0 : col0 + window.width]


@contextmanager
def open_scene_bands(
    name: str, s3_path: str, *, opener: Callable[..., Any] | None = None
) -> Iterator[dict[str, Any]]:
    """A scene's VV band, and its VH band if it has one, open for windowed reads."""
    if opener is None:
        from core.sar.reader import open_measurement as opener
    # open_measurement needs only the product's name and object-storage path.
    product = SimpleNamespace(name=name, s3_path=s3_path)
    with ExitStack() as stack:
        bands = {"vv": stack.enter_context(opener(product, "vv"))}
        try:
            bands["vh"] = stack.enter_context(opener(product, "vh"))
        except S3AccessError as exc:
            if exc.reason != "not_found":
                raise
        yield bands


def _file_time(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat()


def _write_chips(root: Path, scene: str, lon: float, lat: float, amplitudes: dict) -> None:
    """Each polarisation's chip: fixed dB stretch, transparent where not imaged, written atomically."""
    for pol, dn in amplitudes.items():
        low, high = CHIP_STRETCH_DN[pol]
        grey = db_stretch(dn, low_dn=low, high_dn=high)
        png = encode_gray_png(grey, transparent_zero=True)
        _write_atomic(root / chip_relpath(scene, lon, lat, pol), png)


def _write_atomic(path: Path, data: bytes) -> None:
    """Never leave a half-written chip where the API could serve it; each writer has its own temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        Path(partial).replace(path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise


def _on_disk(event: EventView, root: Path) -> dict | None:
    """The event's card recording the chips already on disk for its scene and position, or None."""
    chip = event.evidence.get("chip") or {}
    try:
        image = stored_image(root, chip["scene"], float(chip["lon"]), float(chip["lat"]))
    except (KeyError, TypeError, ValueError):
        return None
    return None if image is None else with_chip_image(event.evidence, image)


def _cut_one(
    event: EventView, bands: dict[str, Any], georef: SceneGeoref, root: Path
) -> dict | None:
    """The event's card recording its new chip, or None (reported) if it could not be cut."""
    chip = event.evidence.get("chip") or {}
    try:
        scene, lon, lat = chip["scene"], float(chip["lon"]), float(chip["lat"])
        amplitudes = cut_north_up(bands, georef, lon, lat, size_px=CHIP_PX, pixel_m=CHIP_PIXEL_M)
        _write_chips(root, scene, lon, lat, amplitudes)
    except CHIP_ERRORS as exc:
        print(f"  event {event.id}: chip not cut ({exc})", flush=True)
        return None
    image = image_meta(
        list(amplitudes),
        cut_at=_file_time(root / chip_relpath(scene, lon, lat, "vv")),
        no_data=round(float((amplitudes["vv"] == 0).mean()), 4),
    )
    return with_chip_image(event.evidence, image)


def _record(factory, pending: Sequence[tuple[EventView, dict]]) -> int:
    """Write cards back where they did not change meanwhile; the number written."""
    written = 0
    with session_scope(factory) as session:
        repo = EventRepository(session)
        for event, card in pending:
            if repo.update_evidence(event.id, expected=event.evidence, evidence=card):
                written += 1
            else:
                print(
                    f"  event {event.id}: changed while its chip was cut; not recorded", flush=True
                )
    return written


def cut_event_chips(
    factory,
    scene_id: int,
    *,
    root: Path,
    open_scene: Callable[[str, str], Any] = open_scene_bands,
    run: str = RUN,
    force: bool = False,
) -> ChipResult:
    """Record the chips of a pass's dark-vessel candidates that have none (all, with ``force``).

    Chips already on disk for a candidate's scene and position are recorded
    without a CDSE read (not with ``force``); the scene is opened once, and
    only for candidates left. A candidate whose chip cannot be cut, or whose
    card changed meanwhile, keeps its card and counts as failed; a scene that
    cannot be opened raises.
    """
    with session_scope(factory) as session:
        scene = SceneRepository(session).get(scene_id)
        if scene is None:
            raise LookupError(f"no scene {scene_id}")
        name, s3_path = scene.product_name, scene.cog_url
        events = [
            e
            for e in EventRepository(session).for_scene(scene_id, run)
            if e.event_type == DARK_VESSEL
        ]
    todo = [e for e in events if force or "image" not in (e.evidence.get("chip") or {})]
    skipped = len(events) - len(todo)
    on_disk = [] if force else [(e, card) for e in todo if (card := _on_disk(e, root))]
    reused = _record(factory, on_disk) if on_disk else 0
    taken = {e.id for e, _ in on_disk}
    to_cut = [e for e in todo if e.id not in taken]
    not_reused = len(on_disk) - reused
    if not to_cut:
        return ChipResult(cut=0, skipped=skipped, failed=not_reused, reused=reused)
    if not s3_path:
        raise ValueError(f"{name} has no image path (cog_url) to cut chips from")

    cut_cards = []
    with open_scene(name, s3_path) as bands:
        georef = SceneGeoref.from_dataset(bands["vv"])
        for event in to_cut:
            card = _cut_one(event, bands, georef, root)
            if card is not None:
                cut_cards.append((event, card))
    cut = _record(factory, cut_cards)
    return ChipResult(
        cut=cut, skipped=skipped, failed=not_reused + len(to_cut) - cut, reused=reused
    )


def cut_detection_chips(
    points: Sequence[tuple[float, float]],
    held: dict[str, HeldBand],
    georef: SceneGeoref,
    *,
    root: Path,
    scene: str,
    open_scene: Callable[[], Any],
) -> DetectionChips:
    """Chips of detections, cut from the pixels already read for detection.

    A chip reaching beyond those pixels (a detection near the box edge) is
    read from the scene instead, opened once for all of them by
    ``open_scene()``. A chip that cannot be cut is reported and counted; the
    others go on.
    """
    from_memory = read = failed = 0
    beyond: list[tuple[float, float]] = []
    for lon, lat in points:
        try:
            amplitudes = cut_north_up(
                held, georef, lon, lat, size_px=CHIP_PX, pixel_m=CHIP_PIXEL_M, read=_read_held
            )
            _write_chips(root, scene, lon, lat, amplitudes)
            from_memory += 1
        except NotHeldError:
            beyond.append((lon, lat))
        except CHIP_ERRORS as exc:
            print(f"  chip at {lat:.5f}, {lon:.5f} not cut ({exc})", flush=True)
            failed += 1
    if not beyond:
        return DetectionChips(from_memory, read, failed)
    try:
        with open_scene() as bands:
            for lon, lat in beyond:
                try:
                    amplitudes = cut_north_up(
                        bands, georef, lon, lat, size_px=CHIP_PX, pixel_m=CHIP_PIXEL_M
                    )
                    _write_chips(root, scene, lon, lat, amplitudes)
                    read += 1
                except CHIP_ERRORS as exc:
                    print(f"  chip at {lat:.5f}, {lon:.5f} not cut ({exc})", flush=True)
                    failed += 1
    except (S3AccessError, OSError, ValueError) as exc:
        print(f"  scene not opened for {len(beyond)} edge chips ({exc})", flush=True)
        failed = len(points) - from_memory - read
    return DetectionChips(from_memory, read, failed)


def cut_stored_detection_chips(
    factory,
    scene_id: int,
    *,
    root: Path,
    open_scene: Callable[[str, str], Any] = open_scene_bands,
    version: str = PRODUCT_VERSION,
    force: bool = False,
) -> ChipResult:
    """Chips for a pass's stored product-model detections missing one (all, with ``force``).

    The scene is opened once, and only if there is work; other detectors' and
    versions' rows get no chips (the console shows the product model only).
    """
    with session_scope(factory) as session:
        scene = SceneRepository(session).get(scene_id)
        if scene is None:
            raise LookupError(f"no scene {scene_id}")
        name, s3_path = scene.product_name, scene.cog_url
        extent = BBox(scene.min_lon, scene.min_lat, scene.max_lon, scene.max_lat)
        found = [
            d
            for d in DetectionRepository(session).in_bbox(
                extent, detector=PRODUCT_DETECTOR, detector_version=version, limit=100_000
            )
            if d.scene_id == scene_id
        ]
    todo = [
        d
        for d in found
        if force or any(chip_file_at(root, name, d.lon, d.lat, pol) is None for pol in ("vv", "vh"))
    ]
    skipped = len(found) - len(todo)
    if not todo:
        return ChipResult(cut=0, skipped=skipped, failed=0)
    if not s3_path:
        raise ValueError(f"{name} has no image path (cog_url) to cut chips from")
    cut = failed = 0
    with open_scene(name, s3_path) as bands:
        georef = SceneGeoref.from_dataset(bands["vv"])
        for detection in todo:
            try:
                amplitudes = cut_north_up(
                    bands,
                    georef,
                    detection.lon,
                    detection.lat,
                    size_px=CHIP_PX,
                    pixel_m=CHIP_PIXEL_M,
                )
                _write_chips(root, name, detection.lon, detection.lat, amplitudes)
                cut += 1
            except CHIP_ERRORS as exc:
                print(f"  detection {detection.id}: chip not cut ({exc})", flush=True)
                failed += 1
    return ChipResult(cut=cut, skipped=skipped, failed=failed)


def main(argv: Sequence[str] | None = None) -> int:
    from sqlalchemy.exc import SQLAlchemyError

    from core.config import load_settings
    from core.db.session import create_all, create_db_engine, make_session_factory

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", help="product name of one pass (default: every fused pass)")
    parser.add_argument("--force", action="store_true", help="cut chips that exist again")
    parser.add_argument(
        "--detections",
        action="store_true",
        help="cut chips of stored product-model detections instead of dark-vessel events",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    engine = create_db_engine(settings.database_url)
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        scenes = SceneRepository(session)
        if args.detections:
            ids = {
                d.scene_id
                for d in DetectionRepository(session).in_bbox(
                    settings.gulf_bbox,
                    detector=PRODUCT_DETECTOR,
                    detector_version=PRODUCT_VERSION,
                    limit=100_000,
                )
            }
        else:
            ids = set(EventRepository(session).scene_ids(RUN))
        passes = {i: scenes.get(i).product_name for i in sorted(ids)}
    if args.scene is not None:
        passes = {i: name for i, name in passes.items() if name == args.scene}
        if not passes:
            print(f"nothing to cut stored for {args.scene}", flush=True)
            return 1

    root = settings.data_dir / "chips"
    problems = 0
    for scene_id, name in passes.items():
        try:
            if args.detections:
                result = cut_stored_detection_chips(factory, scene_id, root=root, force=args.force)
            else:
                result = cut_event_chips(factory, scene_id, root=root, force=args.force)
        except (S3AccessError, ValueError, LookupError, OSError, SQLAlchemyError) as exc:
            print(f"{name}: no chips cut ({exc})", flush=True)
            problems += 1
            continue
        print(
            f"{name}: {result.cut} cut, {result.reused} reused from disk, "
            f"{result.skipped} already recorded, {result.failed} failed",
            flush=True,
        )
        problems += result.failed
    print("Contains modified Copernicus Sentinel data.", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
