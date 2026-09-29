"""Tests for the model contract and dense training targets built from index records."""

from __future__ import annotations

import numpy as np
import pytest

from core.inference.contract import (
    BAND_SIGMOIDS,
    OUTPUT_SIZE,
    STRIDE,
    normalise_chip,
)
from core.labeling.targets import (
    CENTRE_RADIUS,
    IGNORE_RADIUS,
    encode_targets,
    output_position,
)


def rec(supervision: str, *, row: float = 1000.0, col: float = 2000.0, objects=None) -> dict:
    return {"supervision": supervision, "row": row, "col": col, "objects": objects or []}


# --- contract ------------------------------------------------------------------


def test_sigmoids_put_sea_near_a_tenth_and_ships_near_nine_tenths():
    """Measured 2026-09-29 on 4,000 chips: sea and ship-peak medians per band."""
    chip = np.zeros((2, 4, 4), dtype=np.uint16)
    chip[0, 0, 0], chip[0, 0, 1] = 60, 1829  # VV sea, ship
    chip[1, 0, 0], chip[1, 0, 1] = 36, 650  # VH sea, ship

    out = normalise_chip(chip)

    assert out[:, 0, 0] == pytest.approx([0.1, 0.1], abs=0.01)
    assert out[:, 0, 1] == pytest.approx([0.9, 0.9], abs=0.01)


def test_nodata_stays_zero_and_output_is_float32():
    chip = np.zeros((2, 3, 3), dtype=np.uint16)

    out = normalise_chip(chip)

    assert out.dtype == np.float32
    assert not out.any()


def test_contract_is_two_bands_at_stride_two():
    assert len(BAND_SIGMOIDS) == 2
    assert STRIDE == 2
    assert OUTPUT_SIZE == 64


# --- geometry --------------------------------------------------------------------


def test_chip_centre_maps_to_the_middle_output_cell():
    """Chip pixel 64 covers output cell 32; the cell centre sits at chip 64.5."""
    cell, offset = output_position(64.0)

    assert cell == 32
    assert offset == pytest.approx(-0.25)


# --- positives ---------------------------------------------------------------------


def test_positive_chip_peaks_at_the_centre_with_its_offset():
    targets = encode_targets(rec("positive", row=1000.4, col=2000.0))

    heat = targets["heatmap"]
    assert heat.shape == (OUTPUT_SIZE, OUTPUT_SIZE)
    assert np.unravel_index(heat.argmax(), heat.shape) == (32, 32)
    assert heat[32, 32] == pytest.approx(1.0)
    assert targets["offset_mask"][32, 32] == 1
    assert targets["offset"][:, 32, 32] == pytest.approx([-0.05, -0.25])
    assert targets["offset_mask"].sum() == 1
    assert targets["weight"].min() == 1.0


def test_vessels_inside_a_chip_get_their_own_peaks():
    targets = encode_targets(rec("full_negative", objects=[[20.0, -40.0, "vessel"]]))

    assert targets["heatmap"][42, 12] == pytest.approx(1.0)
    assert targets["offset_mask"].sum() == 1


# --- ignore and negatives --------------------------------------------------------------


def test_unlabelled_centre_is_left_out_of_the_loss():
    targets = encode_targets(rec("ignore"))

    assert targets["weight"][32, 32] == 0
    assert targets["weight"][32, 32 + IGNORE_RADIUS + 1] == 1
    assert targets["heatmap"].max() == 0


def test_unlabelled_neighbour_is_left_out_but_a_vessel_beside_it_is_kept():
    objects = [[0.0, 30.0, "unlabelled"], [0.0, 40.0, "vessel"]]

    targets = encode_targets(rec("positive", objects=objects))

    assert targets["weight"][32, 47] == 0  # the unlabelled one, 15 cells right
    assert targets["weight"][32, 52] == 1  # the vessel's own cell stays supervised
    assert targets["heatmap"][32, 52] == pytest.approx(1.0)


def test_centre_negative_supervises_only_around_the_centre():
    targets = encode_targets(rec("centre_negative"))

    assert targets["weight"][32, 32] == 1
    assert targets["weight"][32, 32 + CENTRE_RADIUS + 1] == 0
    assert targets["weight"][0, 0] == 0
    assert targets["heatmap"].max() == 0


def test_persistent_structure_inside_a_chip_is_an_explicit_negative():
    targets = encode_targets(rec("centre_negative", objects=[[0.0, 40.0, "persistent"]]))

    assert targets["weight"][32, 52] == 1
    assert targets["heatmap"][32, 52] == 0


def test_full_negative_supervises_everything_as_empty():
    targets = encode_targets(rec("full_negative"))

    assert targets["weight"].min() == 1
    assert targets["heatmap"].max() == 0
    assert targets["offset_mask"].sum() == 0


def test_objects_outside_the_output_grid_are_skipped():
    targets = encode_targets(rec("full_negative", objects=[[70.0, 0.0, "vessel"]]))

    assert targets["heatmap"].max() == 0


def test_unknown_supervision_is_an_error():
    with pytest.raises(ValueError, match="supervision"):
        encode_targets(rec("maybe"))
