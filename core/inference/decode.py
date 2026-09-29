"""Turn the point detector's dense output into points, and score points against truth.

Numpy only, so the training evaluator and the CPU inference runner share it.
Peaks are local maxima under a 3x3 window (the "max-pool NMS" that keeps the
model free of custom operators) above a threshold; each peak's sub-pixel
offset moves it back to input-pixel coordinates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from core.inference.contract import STRIDE


def decode_peaks(
    heatmap: np.ndarray, offset: np.ndarray, *, threshold: float
) -> list[tuple[float, float, float]]:
    """(row, col, score) of every 3x3 local maximum above ``threshold``, best first.

    Coordinates are input pixels: output cell i with offset o sits at
    (i + o) * STRIDE + 0.5, the inverse of core.labeling.targets.
    """
    heat = np.asarray(heatmap, dtype=np.float32)
    padded = np.pad(heat, 1, constant_values=-np.inf)
    neighbourhood = np.max(
        [
            padded[dr : dr + heat.shape[0], dc : dc + heat.shape[1]]
            for dr in range(3)
            for dc in range(3)
        ],
        axis=0,
    )
    rows, cols = np.nonzero((heat >= neighbourhood) & (heat >= threshold))
    peaks = [
        (
            float((r + offset[0, r, c]) * STRIDE + 0.5),
            float((c + offset[1, r, c]) * STRIDE + 0.5),
            float(heat[r, c]),
        )
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True)
    ]
    return sorted(peaks, key=lambda p: -p[2])


def match_points(
    predictions: Sequence[tuple[float, float, float]],
    truth: Sequence[tuple[float, float]],
    *,
    tolerance: float,
    counts: Callable[[float, float], bool] | None = None,
) -> tuple[int, int, int]:
    """True positives, false positives, false negatives, matching best-scored first.

    Each truth point matches at most one prediction within ``tolerance``.
    An unmatched prediction counts as false only where ``counts`` says the
    ground is supervised (unlabelled detections, for instance, are not).
    """
    unmatched = list(truth)
    tp = fp = 0
    for row, col, _ in sorted(predictions, key=lambda p: -p[2]):
        best = None
        best_distance = tolerance
        for i, (t_row, t_col) in enumerate(unmatched):
            distance = float(np.hypot(row - t_row, col - t_col))
            if distance <= best_distance:
                best, best_distance = i, distance
        if best is not None:
            unmatched.pop(best)
            tp += 1
        elif counts is None or counts(row, col):
            fp += 1
    return tp, fp, len(unmatched)
