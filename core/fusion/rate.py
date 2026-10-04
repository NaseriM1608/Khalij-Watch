"""The dark rate pooled over passes, with a 95% interval from resampling whole passes.

Rate = dark / kept summed over passes, where "kept" are detections that
survive the pre-filters and "dark" those of them with no AIS candidate in the
gate (PLAN §17, 2026-10-04 "Phase 6 feasibility"). Detections of one pass
share its weather, traffic and AIS reception, so the interval resamples whole
passes, as the detector comparison does (scripts/compare_cfar_model.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class RateEstimate:
    """A pooled dark rate with its 95% interval (None when no detection was kept)."""

    rate: float | None
    low: float | None
    high: float | None
    dark: int
    kept: int
    passes: int


def dark_rate(
    dark: Sequence[int],
    kept: Sequence[int],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> RateEstimate:
    """Dark share of kept detections over passes, with a 95% interval from resampling passes."""
    d, k = np.asarray(dark, dtype=np.int64), np.asarray(kept, dtype=np.int64)
    if d.shape != k.shape:
        raise ValueError(f"dark and kept differ in length: {len(d)} vs {len(k)}")
    if (d < 0).any() or (k < 0).any():
        raise ValueError("counts must not be negative")
    if (d > k).any():
        raise ValueError("a pass has more dark detections than kept ones")
    total_dark, total_kept = int(d.sum()), int(k.sum())
    if total_kept == 0:
        return RateEstimate(None, None, None, total_dark, total_kept, len(k))
    draws = np.random.default_rng(seed).integers(0, len(k), size=(resamples, len(k)))
    boot_kept = k[draws].sum(axis=1)
    usable = boot_kept > 0  # a resample of passes that kept nothing has no rate
    ratios = d[draws].sum(axis=1)[usable] / boot_kept[usable]
    low, high = (float(v) for v in np.percentile(ratios, [2.5, 97.5]))
    return RateEstimate(total_dark / total_kept, low, high, total_dark, total_kept, len(k))
