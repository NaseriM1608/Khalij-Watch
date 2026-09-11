"""Tests for binary morphology helpers."""

from __future__ import annotations

import numpy as np

from core.sar.morphology import dilate


def test_dilation_grows_a_pixel_into_a_square():
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True

    grown = dilate(mask, 2)

    assert grown.sum() == 25
    assert grown[3:8, 3:8].all()


def test_dilation_is_clipped_at_the_image_edge():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True

    assert dilate(mask, 1).sum() == 4


def test_zero_radius_dilation_changes_nothing():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 3] = True

    assert np.array_equal(dilate(mask, 0), mask)


def test_dilation_returns_a_boolean_mask():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True

    assert dilate(mask, 1).dtype == bool
