"""Rendering SAR amplitude for people to look at."""

from __future__ import annotations

import numpy as np


def display_stretch(pixels: np.ndarray) -> np.ndarray:
    """Percentile stretch for viewing only.

    Deliberately not the frozen sigmoid used as model input: this exists to
    make faint vessels visible to a person, and must never feed a model.
    """
    valid = pixels[pixels > 0]
    if valid.size == 0:
        return np.zeros(pixels.shape, dtype=np.uint8)
    low, high = np.percentile(valid, [2.0, 99.8])
    scaled = np.clip((pixels.astype(np.float32) - low) / max(high - low, 1.0), 0.0, 1.0)
    return (np.sqrt(scaled) * 255.0).astype(np.uint8)
