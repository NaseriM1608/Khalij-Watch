"""Tests for building persistent spots from archive comparison passes."""

from __future__ import annotations

import json
import sys

from core.persistence import load_spots
from scripts import build_persistence as build

PRODUCT = {"sha256": "0535c82ddaa577f1", "tile": 256, "overlap": 64}
OTHER = {"sha256": "8910aba9e98c1bc2", "tile": 128, "overlap": 32}
SPOT = (50.30, 29.25)


def write(folder, area, scene, *, product_peaks, other_peaks=(), label="run2"):
    folder.mkdir(parents=True, exist_ok=True)
    day = scene.split("_")[4]
    record = {
        "area": area,
        "scene": scene,
        "acquired": f"{day[:4]}-{day[4:6]}-{day[6:8]}T02:38:34+00:00",
        "cfar": [],
        "models": {
            label: {**PRODUCT, "path": "x.onnx", "peaks": [list(p) for p in product_peaks]},
            "v1": {**OTHER, "path": "y.onnx", "peaks": [list(p) for p in other_peaks]},
        },
        "refs": [],
    }
    (folder / f"{area}__{scene.removesuffix('.SAFE')}.json").write_text(json.dumps(record))


def scene(day: str) -> str:
    return f"S1A_IW_GRDH_1SDV_{day}T023834_{day}T023859_000000_000000_ABCD_COG.SAFE"


def test_spots_come_from_the_product_models_confident_detections_before_the_cutoff(
    tmp_path, monkeypatch
):
    archive, later = tmp_path / "compare_a", tmp_path / "compare_b"
    for i, day in enumerate(("20240105", "20240305", "20240504", "20240703")):
        write(archive, "kharg", scene(day), product_peaks=[(*SPOT, 0.9)] if i < 3 else [])
    # Below the product threshold, another model, or after the cutoff: never counted.
    write(archive, "bushehr", scene("20240105"), product_peaks=[(50.72, 28.95, 0.5)])
    write(
        archive, "bushehr", scene("20240305"), product_peaks=[], other_peaks=[(50.72, 28.95, 0.9)]
    )
    write(archive, "bushehr", scene("20240504"), product_peaks=[], label="e05_t256o64")
    for day in ("20260910", "20260922", "20260928"):
        write(later, "bushehr", scene(day), product_peaks=[(50.72, 28.95, 0.9)])
    out = tmp_path / "spots.json"
    argv = ["build", "--results", str(archive), str(later), "--before", "2026-07-01"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out)])

    assert build.main() == 0

    meta, spots = load_spots(out)
    assert [(s.area, s.passes_seen, s.passes_imaged) for s in spots] == [("kharg", 3, 4)]
    assert meta["passes"] == {"bushehr": 3, "kharg": 4}
    assert meta["before"] == "2026-07-01"
    assert meta["model"] == PRODUCT
    assert meta["threshold"] == 0.6
