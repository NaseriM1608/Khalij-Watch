"""The model contract shared by training and inference: bands, normalisation, stride.

Everything here is FROZEN once a model is trained on it. Normalisation is a
silent train/serve skew source: a model trained under one mapping quietly
degrades under another, so both sides import these values from here.

Sigmoids were solved on 4,000 chips of the 2024-2025 Gulf dataset
(2026-09-29) so that the median open-sea amplitude maps to 0.1 and the
median ship peak to 0.9 in each band:

    band  sea median  ship peak median  scale   offset
    VV    60 DN       1,829 DN          0.0025  945
    VH    36 DN       650 DN            0.0072  343

The older VV-only curve (0.004, 640) sent half of all ship peaks above 0.99.
"""

from __future__ import annotations

import numpy as np

BAND_ORDER = ("vv", "vh")
# (scale, offset) per band, in BAND_ORDER: sigmoid(scale * (dn - offset)).
BAND_SIGMOIDS = ((0.0025, 945.0), (0.0072, 343.0))
CHIP_SIZE = 128
STRIDE = 2
OUTPUT_SIZE = CHIP_SIZE // STRIDE


def normalise_chip(chip: np.ndarray) -> np.ndarray:
    """Raw (band, row, col) amplitude to float32 in [0, 1]; nodata (0) stays 0."""
    values = np.asarray(chip, dtype=np.float32)
    if values.shape[0] != len(BAND_SIGMOIDS):
        raise ValueError(f"expected {len(BAND_SIGMOIDS)} bands, got shape {values.shape}")
    out = np.empty_like(values)
    for band, (scale, offset) in enumerate(BAND_SIGMOIDS):
        out[band] = 1.0 / (1.0 + np.exp(-scale * (values[band] - offset)))
    out[values == 0] = 0.0
    return out
