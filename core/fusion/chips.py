"""Radar chips of dark-vessel candidates: where each is stored and how its card points to it.

The evidence card shows what the radar saw (PLAN §9: each event stores its SAR
chip). A chip is cut once, in batch, from the pass's image on CDSE
(scripts/cut_chips.py, also run by scripts/fuse_scene.py) and served from
disk, so a browser never causes a CDSE read. Files are named by scene and
position, not by event id, because re-fusing a pass replaces its events. The
API builds a chip's path from the stored card, so a card can never point
outside the chip folder.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

CHIP_PX = 256
# The GRD pixel spacing: a chip is 2.56 km across, enough to show a ship and
# its neighbours in an anchorage.
CHIP_PIXEL_M = 10.0
POLARISATIONS = ("vv", "vh")
# Fixed display stretch (core.sar.display.db_stretch), in DN per polarisation
# on a logarithmic scale. Fixed rather than per image so that chips of
# different passes compare and a faint return stays faint: open sea (median
# VV 56-72, VH 28-39 DN on S1A, S1C and S1D, PLAN §17 2026-10-04) lands near
# the dark end, a bright ship return (about 1200 DN in VV) near white. Kept
# here, free of numpy, because the API imports this module.
CHIP_STRETCH_DN = {"vv": (30.0, 1500.0), "vh": (15.0, 750.0)}
# Sentinel-1 product names: letters, digits and underscores, ".SAFE" at most.
_SCENE_NAME = re.compile(r"[A-Za-z0-9_]+(?:\.SAFE)?")


def chip_relpath(scene: str, lon: float, lat: float, pol: str) -> PurePosixPath:
    """A chip's path under the chip folder: ``<scene stem>/<lat>_<lon>_<pol>.png`` (about 1 m)."""
    if not isinstance(scene, str) or not _SCENE_NAME.fullmatch(scene):
        raise ValueError(f"not a Sentinel-1 product name: {scene!r}")
    if pol not in POLARISATIONS:
        raise ValueError(f"no chips for polarisation {pol!r}")
    return PurePosixPath(scene.removesuffix(".SAFE"), f"{lat:.5f}_{lon:.5f}_{pol}.png")


def image_meta(
    pols: Sequence[str], *, cut_at: str | None = None, no_data: float | None = 0.0
) -> dict:
    """What the card records about a cut chip: how to read it, for the console's overlays.

    ``cut_at`` (the file's time) versions the image for the console, since a
    re-fused pass can give an event id to another detection; ``no_data`` is
    the share of the chip the radar did not image (None if unknown).
    """
    return {
        "pols": list(pols),
        "size_px": CHIP_PX,
        "pixel_m": CHIP_PIXEL_M,
        "north_up": True,
        "stretch_dn": {p: list(CHIP_STRETCH_DN[p]) for p in pols},
        "cut_at": cut_at,
        "no_data": no_data,
    }


def with_chip_image(evidence: Mapping[str, Any], image: Mapping[str, Any]) -> dict:
    """A copy of an evidence card recording its cut chip; the card passed in is unchanged."""
    chip = evidence.get("chip")
    if not isinstance(chip, Mapping):
        raise ValueError("the evidence card has no chip reference (scene and position)")
    return {**evidence, "chip": {**chip, "image": dict(image)}}


def chip_file(root: Path, evidence: Mapping[str, Any], pol: str) -> Path | None:
    """The stored chip of an event in one polarisation, or None if none was cut or the card is malformed."""
    chip = evidence.get("chip")
    if not isinstance(chip, Mapping):
        return None
    image = chip.get("image")
    if not isinstance(image, Mapping) or not isinstance(image.get("pols"), list):
        return None
    if pol not in image["pols"]:
        return None
    lon, lat = chip.get("lon"), chip.get("lat")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (lon, lat)):
        return None
    try:
        relative = chip_relpath(chip.get("scene"), lon, lat, pol)
    except ValueError:
        return None
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        return None
    return path
