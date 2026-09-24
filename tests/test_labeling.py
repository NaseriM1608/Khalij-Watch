"""Tests for chip extraction, dataset manifests, QA contact sheets and negatives."""

from __future__ import annotations

import numpy as np
import pytest

from core.labeling.chips import CHIP_SIZE, ChipRequest, extract_chips
from core.labeling.contact_sheet import GAP_PX, TICK_FAR_PX, TICK_NEAR_PX, contact_sheet
from core.labeling.manifest import ChipRecord, records_from_jsonl, records_to_jsonl, split_for
from core.labeling.negatives import sample_sea_negatives

HALF = CHIP_SIZE // 2


def bands(height: int = 400, width: int = 500) -> np.ndarray:
    """A (VV, VH) stack with no zero (nodata) pixels."""
    vv = (np.arange(height * width, dtype=np.int64).reshape(height, width) % 1000 + 1).astype(
        np.uint16
    )
    return np.stack([vv, vv // 2 + 1])


# --------------------------------------------------------------------------
# Chips
# --------------------------------------------------------------------------


def test_a_chip_is_centred_on_the_requested_pixel():
    stack = bands()
    stack[:, 200, 250] = 9999

    chips, kept, skipped = extract_chips(stack, [ChipRequest("a", row=200.0, col=250.0)])

    assert chips.shape == (1, 2, CHIP_SIZE, CHIP_SIZE)
    assert chips.dtype == np.uint16
    assert chips[0, 0, HALF, HALF] == 9999
    assert kept == ["a"]
    assert skipped == []


def test_scene_coordinates_are_converted_with_the_window_offset():
    stack = bands()
    stack[:, 200, 250] = 9999

    chips, _, _ = extract_chips(
        stack, [ChipRequest("a", row=1200.0, col=3250.0)], window_offset=(1000, 3000)
    )

    assert chips[0, 0, HALF, HALF] == 9999


def test_fractional_positions_round_to_the_nearest_pixel():
    stack = bands()
    stack[:, 200, 250] = 9999

    chips, _, _ = extract_chips(stack, [ChipRequest("a", row=200.4, col=249.6)])

    assert chips[0, 0, HALF, HALF] == 9999


def test_a_chip_that_would_leave_the_window_is_skipped_with_a_reason():
    chips, kept, skipped = extract_chips(bands(), [ChipRequest("edge", row=10.0, col=250.0)])

    assert chips.shape == (0, 2, CHIP_SIZE, CHIP_SIZE)
    assert kept == []
    assert skipped == [("edge", "outside_window")]


def test_a_chip_mostly_without_data_is_skipped():
    stack = bands()
    stack[:, :, :260] = 0  # beyond the swath edge

    _, kept, skipped = extract_chips(stack, [ChipRequest("swath_edge", row=200.0, col=250.0)])

    assert kept == []
    assert skipped == [("swath_edge", "nodata")]


def test_raw_amplitude_is_kept_unnormalised():
    """Normalisation is chosen, and frozen, at training time; chips stay lossless."""
    stack = bands()
    stack[:, 200, 250] = 12345

    chips, _, _ = extract_chips(stack, [ChipRequest("a", row=200.0, col=250.0)])

    assert chips.max() == 12345


def test_floating_point_input_is_rounded_into_uint16():
    stack = bands().astype(np.float32)
    stack[:, 200, 250] = 99.6

    chips, _, _ = extract_chips(stack, [ChipRequest("a", row=200.0, col=250.0)])

    assert chips[0, 0, HALF, HALF] == 100


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def record(**overrides) -> ChipRecord:
    values = {
        "chip_id": "S1D_X/0000",
        "scene": "S1D_X",
        "split": "train",
        "shard": "S1D_X.npz",
        "index": 0,
        "lon": 56.5,
        "lat": 25.2,
        "row": 100.0,
        "col": 200.0,
        "kind": "vessel",
        "source": "gfw",
        "mmsi": 422123456,
        "length_m": 212.0,
        "vessel_type": "cargo_tanker",
        "extra": {"matching_score": 0.9},
    }
    values.update(overrides)
    return ChipRecord(**values)


def test_the_split_is_decided_by_scene_and_is_stable():
    """Chips of one scene must never straddle splits: adjacent chips leak."""
    assert split_for("S1D_IW_GRDH_A") == split_for("S1D_IW_GRDH_A")
    assert split_for("S1D_IW_GRDH_A") in {"train", "val", "test"}


def test_split_fractions_are_roughly_respected():
    # 20,000 names: at 3,000 one fixed set of names fell 3.7 sigma low on
    # "val", although 100,000 names give 79.8 / 10.1 / 10.1 percent.
    scenes = [f"S1D_IW_GRDH_1SDV_2026{i:08d}_COG" for i in range(20_000)]
    splits = [split_for(scene) for scene in scenes]

    assert splits.count("train") / len(scenes) == pytest.approx(0.8, abs=0.01)
    assert splits.count("val") / len(scenes) == pytest.approx(0.1, abs=0.01)
    assert splits.count("test") / len(scenes) == pytest.approx(0.1, abs=0.01)


def test_records_round_trip_through_jsonl():
    records = [record(), record(chip_id="S1D_X/0001", index=1, kind="negative", mmsi=None)]

    assert records_from_jsonl(records_to_jsonl(records)) == records


def test_a_record_kind_must_be_known():
    with pytest.raises(ValueError, match="kind"):
        record(kind="boat")


def test_a_record_split_must_be_known():
    with pytest.raises(ValueError, match="split"):
        record(split="holdout")


# --------------------------------------------------------------------------
# Contact sheet
# --------------------------------------------------------------------------


def test_the_contact_sheet_tiles_chips_in_a_grid():
    chips = np.full((10, 2, CHIP_SIZE, CHIP_SIZE), 50, np.uint16)

    sheet = contact_sheet(chips, columns=4)

    rows = 3
    assert sheet.shape == (
        3,
        rows * CHIP_SIZE + (rows + 1) * GAP_PX,
        4 * CHIP_SIZE + 5 * GAP_PX,
    )
    assert sheet.dtype == np.uint8


def test_every_chip_centre_is_marked_without_hiding_the_target():
    """The sheet exists to check that ships sit where the labels put them."""
    chips = np.full((1, 2, CHIP_SIZE, CHIP_SIZE), 50, np.uint16)
    chips[0, :, HALF, HALF] = 5000

    sheet = contact_sheet(chips, columns=1)

    centre = GAP_PX + HALF
    tick = (sheet[0, centre, centre - TICK_NEAR_PX], sheet[1, centre, centre - TICK_NEAR_PX])
    assert tick == (255, 40)  # a red tick beside the centre
    assert sheet[0, centre, centre] == sheet[1, centre, centre]  # the centre itself stays grey
    assert TICK_NEAR_PX < TICK_FAR_PX


def test_each_chip_is_stretched_on_its_own():
    """A faint vessel must stay visible next to a bright one."""
    faint = np.full((2, CHIP_SIZE, CHIP_SIZE), 10, np.uint16)
    faint[:, HALF, HALF] = 100
    bright = np.full((2, CHIP_SIZE, CHIP_SIZE), 10, np.uint16)
    bright[:, HALF, HALF] = 10_000

    sheet = contact_sheet(np.stack([faint, bright]), columns=2)

    centre = GAP_PX + HALF
    assert sheet[1, centre, centre] > 200


def test_an_empty_contact_sheet_is_refused():
    with pytest.raises(ValueError, match="no chips"):
        contact_sheet(np.zeros((0, 2, CHIP_SIZE, CHIP_SIZE), np.uint16))


# --------------------------------------------------------------------------
# Sea negatives
# --------------------------------------------------------------------------


def test_sea_negatives_keep_clear_of_targets_and_masked_ground():
    side = 2000
    mask = np.zeros((side, side), dtype=bool)
    mask[:, :300] = True  # land along the west
    targets = [(1000.0, 1000.0)]

    points = sample_sea_negatives(
        mask, targets, count=50, clearance_px=100, rng=np.random.default_rng(0)
    )

    assert len(points) == 50
    for row, col in points:
        r, c = round(row), round(col)
        assert np.hypot(row - 1000.0, col - 1000.0) >= 100.0
        assert HALF <= r <= side - HALF and HALF <= c <= side - HALF
        # The whole chip, not just its centre, lies on open sea.
        assert not mask[r - HALF : r + HALF, c - HALF : c + HALF].any()


def test_sea_negative_chips_do_not_overlap_each_other():
    """Overlapping negatives would be near-duplicates, inflating the sea class."""
    mask = np.zeros((1000, 1000), dtype=bool)

    points = sample_sea_negatives(mask, [], count=20, clearance_px=0, rng=np.random.default_rng(1))

    for index, (r1, c1) in enumerate(points):
        for r2, c2 in points[index + 1 :]:
            assert abs(r1 - r2) >= CHIP_SIZE or abs(c1 - c2) >= CHIP_SIZE


def test_sea_negatives_are_reproducible():
    mask = np.zeros((600, 600), dtype=bool)

    first = sample_sea_negatives(mask, [], count=5, clearance_px=10, rng=np.random.default_rng(3))
    second = sample_sea_negatives(mask, [], count=5, clearance_px=10, rng=np.random.default_rng(3))

    assert first == second


def test_sampling_gives_up_rather_than_looping_forever():
    mask = np.ones((300, 300), dtype=bool)

    points = sample_sea_negatives(mask, [], count=10, clearance_px=10, rng=np.random.default_rng(0))

    assert points == []
