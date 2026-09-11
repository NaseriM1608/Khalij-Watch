"""Binary morphology on boolean masks, with numpy only.

Kept free of scipy for the same reason as the rest of the detector: the
dependency is heavy and slow to install on this host, and what is needed here
is small.
"""

from __future__ import annotations

import numpy as np


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a mask by ``radius`` pixels in a square, using a summed-area table.

    A square reaches ``radius`` pixels along each axis and further along the
    diagonals, so as a buffer it errs on the generous side.
    """
    if radius <= 0:
        return mask.copy()
    size = 2 * radius + 1
    rows, cols = mask.shape
    # Counts never exceed the pixel total, so int32 suffices below ~46k x 46k
    # and halves the memory of a large window's integral image.
    dtype = np.int32 if (rows + 2 * radius) * (cols + 2 * radius) < 2**31 - 1 else np.int64
    padded = np.pad(mask.astype(dtype), radius)
    integral = np.zeros((rows + 2 * radius + 1, cols + 2 * radius + 1), dtype=dtype)
    integral[1:, 1:] = padded.cumsum(axis=0, dtype=dtype).cumsum(axis=1, dtype=dtype)
    counts = (
        integral[size : size + rows, size : size + cols]
        - integral[:rows, size : size + cols]
        - integral[size : size + rows, :cols]
        + integral[:rows, :cols]
    )
    return counts > 0
