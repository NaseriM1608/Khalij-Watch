"""Radar chips for stored dark-vessel candidates (Phase 7: the evidence card's SAR image).

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
        --with sqlalchemy python -u scripts/cut_chips.py [--scene NAME] [--force]
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

from core.db.repository import EventRepository, EventView, SceneRepository  # noqa: E402
from core.db.session import session_scope  # noqa: E402
from core.fusion.chips import (  # noqa: E402
    CHIP_PIXEL_M,
    CHIP_PX,
    CHIP_STRETCH_DN,
    POLARISATIONS,
    chip_relpath,
    image_meta,
    with_chip_image,
)
from core.sar.display import cut_north_up, db_stretch, encode_gray_png  # noqa: E402
from core.sar.georef import SceneGeoref  # noqa: E402
from core.sar.reader import S3AccessError  # noqa: E402

RUN = "fusion-v1"
DARK_VESSEL = "DARK_VESSEL"


@dataclass(frozen=True)
class ChipResult:
    cut: int
    skipped: int
    failed: int
    reused: int = 0


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
        scene, lon, lat = chip["scene"], float(chip["lon"]), float(chip["lat"])
        paths = {p: root / chip_relpath(scene, lon, lat, p) for p in POLARISATIONS}
    except (KeyError, TypeError, ValueError):
        return None
    if not paths["vv"].is_file():
        return None
    pols = [p for p in POLARISATIONS if paths[p].is_file()]
    image = image_meta(pols, cut_at=_file_time(paths["vv"]), no_data=None)
    return with_chip_image(event.evidence, image)


def _cut_one(
    event: EventView, bands: dict[str, Any], georef: SceneGeoref, root: Path
) -> dict | None:
    """The event's card recording its new chip, or None (reported) if it could not be cut."""
    chip = event.evidence.get("chip") or {}
    try:
        scene, lon, lat = chip["scene"], float(chip["lon"]), float(chip["lat"])
        amplitudes = cut_north_up(bands, georef, lon, lat, size_px=CHIP_PX, pixel_m=CHIP_PIXEL_M)
        for pol, dn in amplitudes.items():
            low, high = CHIP_STRETCH_DN[pol]
            grey = db_stretch(dn, low_dn=low, high_dn=high)
            png = encode_gray_png(grey, transparent_zero=True)
            _write_atomic(root / chip_relpath(scene, lon, lat, pol), png)
    except (KeyError, TypeError, ValueError, OSError, S3AccessError) as exc:
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


def main(argv: Sequence[str] | None = None) -> int:
    from sqlalchemy.exc import SQLAlchemyError

    from core.config import load_settings
    from core.db.session import create_all, create_db_engine, make_session_factory

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", help="product name of one pass (default: every fused pass)")
    parser.add_argument("--force", action="store_true", help="cut chips that exist again")
    args = parser.parse_args(argv)

    settings = load_settings()
    engine = create_db_engine(settings.database_url)
    create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        scenes = SceneRepository(session)
        passes = {i: scenes.get(i).product_name for i in EventRepository(session).scene_ids(RUN)}
    if args.scene is not None:
        passes = {i: name for i, name in passes.items() if name == args.scene}
        if not passes:
            print(f"no {RUN} events stored for {args.scene}", flush=True)
            return 1

    root = settings.data_dir / "chips"
    problems = 0
    for scene_id, name in passes.items():
        try:
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
