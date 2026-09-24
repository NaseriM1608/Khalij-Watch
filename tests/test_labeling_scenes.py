"""Tests for choosing one scene per satellite pass."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from core.labeling.scenes import one_per_pass

T0 = dt.datetime(2025, 9, 17, 14, 16, 51, tzinfo=dt.UTC)


@dataclass(frozen=True)
class Product:
    name: str
    satellite: str
    acq_start: dt.datetime


def at(name: str, satellite: str, seconds: float) -> Product:
    return Product(name, satellite, T0 + dt.timedelta(seconds=seconds))


def test_overlapping_frames_of_one_pass_count_once():
    """Two frames of one pass see the same ships; keeping both would let one
    vessel land in two splits."""
    frames = [at("frame-2", "S1A", 25), at("frame-1", "S1A", 0)]

    assert [p.name for p in one_per_pass(frames)] == ["frame-1"]


def test_separate_passes_are_all_kept_in_time_order():
    passes = [at("evening", "S1A", 12 * 3600), at("morning", "S1A", 0)]

    assert [p.name for p in one_per_pass(passes)] == ["morning", "evening"]


def test_two_satellites_minutes_apart_are_different_passes():
    products = [at("s1a", "S1A", 0), at("s1c", "S1C", 120)]

    assert [p.name for p in one_per_pass(products)] == ["s1a", "s1c"]


def test_nothing_in_nothing_out():
    assert one_per_pass([]) == []
