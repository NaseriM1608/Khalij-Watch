"""Tests for where dark-vessel chips are stored and how evidence cards point to them.

Chips are named by scene and position, not by event id: re-fusing a pass
replaces its events, and their chips must still be found. The API serves a
chip from a path built out of the stored card, so a card must never be able
to point outside the chip folder.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path, PurePosixPath

import pytest

from core.fusion.chips import (
    CHIP_PIXEL_M,
    CHIP_PX,
    chip_file,
    chip_file_at,
    chip_index,
    chip_relpath,
    image_in_index,
    image_meta,
    stored_image,
    with_chip_image,
)

SCENE = "S1D_IW_GRDH_1SDV_20260910T020640_20260910T020705_004509_00860C_1202_COG.SAFE"
STEM = SCENE.removesuffix(".SAFE")
LON, LAT = 56.58012000553768, 25.28648476904467


def evidence(scene: str = SCENE, image: dict | None = None) -> dict:
    chip = {"scene": scene, "lon": LON, "lat": LAT}
    if image is not None:
        chip["image"] = image
    return {"status": "dark", "confidence": "low", "chip": chip}


def store(root: Path, pol: str, scene: str = SCENE) -> Path:
    path = root / chip_relpath(scene, LON, LAT, pol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    return path


def test_a_chip_is_named_by_scene_and_position():
    assert chip_relpath(SCENE, LON, LAT, "vv") == PurePosixPath(STEM, "25.28648_56.58012_vv.png")
    assert chip_relpath(SCENE, -0.5, -1.25, "vh").name == "-1.25000_-0.50000_vh.png"


@pytest.mark.parametrize("scene", ["../etc", "S1D/../../x", "a\\b", "", "S1D_..SAFE", "C:x"])
def test_a_scene_name_that_could_leave_the_chip_folder_is_refused(scene):
    with pytest.raises(ValueError):
        chip_relpath(scene, LON, LAT, "vv")


def test_only_known_polarisations_have_chips():
    with pytest.raises(ValueError):
        chip_relpath(SCENE, LON, LAT, "hh")


def test_the_image_metadata_says_how_to_read_the_chip():
    meta = image_meta(["vv", "vh"])
    assert meta["pols"] == ["vv", "vh"]
    assert (meta["size_px"], meta["pixel_m"], meta["north_up"]) == (CHIP_PX, CHIP_PIXEL_M, True)
    assert set(meta["stretch_dn"]) == {"vv", "vh"}
    assert all(low < high for low, high in meta["stretch_dn"].values())


def test_recording_a_chip_returns_a_new_card_and_keeps_the_reference():
    card = evidence()
    updated = with_chip_image(card, image_meta(["vv"]))
    assert "image" not in card["chip"]
    assert updated["chip"]["image"]["pols"] == ["vv"]
    assert (updated["chip"]["scene"], updated["chip"]["lon"]) == (SCENE, LON)
    assert updated["status"] == "dark"


def test_a_card_without_a_chip_reference_cannot_take_an_image():
    with pytest.raises(ValueError):
        with_chip_image({"status": "dark"}, image_meta(["vv"]))


def test_a_stored_chip_is_found_from_its_card(tmp_path):
    path = store(tmp_path, "vv")
    assert chip_file(tmp_path, evidence(image=image_meta(["vv"])), "vv") == path.resolve()


def test_no_chip_is_found_until_one_was_recorded_and_written(tmp_path):
    store(tmp_path, "vv")
    assert chip_file(tmp_path, evidence(), "vv") is None  # cut but not on the card
    assert chip_file(tmp_path, evidence(image=image_meta(["vv"])), "vh") is None  # pol not cut
    # Recorded but no file on disk.
    assert chip_file(tmp_path / "other", evidence(image=image_meta(["vv"])), "vv") is None


@pytest.mark.parametrize(
    "card",
    [
        {"status": "dark"},
        {"chip": None},
        {"chip": {"scene": SCENE, "lon": "x", "lat": LAT, "image": image_meta(["vv"])}},
        {"chip": {"scene": "../../secrets", "lon": LON, "lat": LAT, "image": image_meta(["vv"])}},
        {"chip": {"scene": SCENE, "lon": LON, "lat": LAT, "image": {"pols": "vv"}}},
    ],
)
def test_a_malformed_card_finds_no_chip(tmp_path, card):
    store(tmp_path, "vv")
    assert chip_file(tmp_path, card, "vv") is None


def test_the_image_metadata_carries_the_cut_version_and_the_no_data_share():
    meta = image_meta(["vv"], cut_at="2026-10-05T08:08:00+00:00", no_data=0.25)
    assert (meta["cut_at"], meta["no_data"]) == ("2026-10-05T08:08:00+00:00", 0.25)


def test_a_chip_on_disk_is_found_by_scene_and_position(tmp_path):
    path = store(tmp_path, "vv")
    assert chip_file_at(tmp_path, SCENE, LON, LAT, "vv") == path.resolve()
    assert chip_file_at(tmp_path, SCENE, LON, LAT, "vh") is None
    assert chip_file_at(tmp_path, "../x", LON, LAT, "vv") is None


def test_the_image_on_disk_says_which_polarisations_exist_and_when_they_were_cut(tmp_path):
    assert stored_image(tmp_path, SCENE, LON, LAT) is None
    store(tmp_path, "vh")
    assert stored_image(tmp_path, SCENE, LON, LAT) is None  # no VV, no chip
    vv = store(tmp_path, "vv")
    image = stored_image(tmp_path, SCENE, LON, LAT)
    assert image["pols"] == ["vv", "vh"]
    cut = dt.datetime.fromisoformat(image["cut_at"])
    assert cut == dt.datetime.fromtimestamp(vv.stat().st_mtime, dt.UTC)
    assert image["no_data"] is None


def test_a_scenes_chips_are_listed_once_and_read_like_stored_image(tmp_path):
    """One folder listing per scene replaces several file checks per detection."""
    store(tmp_path, "vv")
    store(tmp_path, "vh")
    index = chip_index(tmp_path, SCENE)
    assert image_in_index(index, LON, LAT) == stored_image(tmp_path, SCENE, LON, LAT)
    assert image_in_index(index, LON + 0.01, LAT) is None
    assert chip_index(tmp_path, "../x") == {}
    assert chip_index(tmp_path, "S1D_NO_FOLDER.SAFE") == {}
