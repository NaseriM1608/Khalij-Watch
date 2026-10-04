"""Run the exported point detector (ONNX, fp32) over a SAR window on the CPU.

EfficientNet's squeeze-excitation blocks average over the whole input, so
the output depends on the tile size. On large tiles of calm, dark sea that
average shifts every feature and the model goes blind (2024-05-16, Fujairah,
kaggle-v1: 0 detections on 2048 px tiles, 128 on 128 px tiles), so it runs on
small tiles. For the product model (kaggle-v2) 256 px tiles with 64 px overlap
scored best against Global Fishing Watch on the validation passes, and they
halve the duplicate peaks that tile seams make at 128 px. INT8 dynamic
quantisation is not used: it breaks the trained model (logits off by up to
8.8) and is slower on this CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.inference.contract import normalise_chip
from core.inference.decode import decode_peaks

# Chosen against GFW on the validation passes for kaggle-v2 epoch 5 (2026-10-04).
DEFAULT_THRESHOLD = 0.60
DEFAULT_TILE = 256
DEFAULT_OVERLAP = 64
DETECTOR = "pointdet"


@dataclass(frozen=True)
class PointDetection:
    """A model detection: a position and a score, no measured hull."""

    lon: float
    lat: float
    confidence: float
    length_m: None = None
    width_m: None = None
    heading_deg: None = None
    pixel_count: None = None
    peak_amplitude: None = None


def _starts(size: int, tile: int, step: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, step))
    if starts[-1] + tile < size:
        starts.append((size - tile + 1) // 2 * 2)  # even, so output cells align
    return starts


def check_tiling(tile: int, overlap: int) -> None:
    """Tile starts must fall on output cells (even pixels): ``tile`` and ``tile - overlap`` even."""
    if tile % 2 or (tile - overlap) % 2 or not 0 <= overlap < tile:
        raise ValueError(f"tile {tile}, overlap {overlap}: need both even and 0 <= overlap < tile")


def run_tiled(
    predict: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    image: np.ndarray,
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Objectness probabilities and offsets at half resolution over a whole window.

    Tiles overlap and are padded to a multiple of 32; where tiles overlap, the
    more confident tile wins. See check_tiling for the tile sizes allowed.
    """
    check_tiling(tile, overlap)
    _, height, width = image.shape
    out_h, out_w = (height + 1) // 2, (width + 1) // 2
    heat = np.zeros((out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    step = tile - overlap
    for r0 in _starts(height, tile, step):
        for c0 in _starts(width, tile, step):
            crop = image[:, r0 : r0 + tile, c0 : c0 + tile]
            pad = ((0, 0), (0, -crop.shape[1] % 32), (0, -crop.shape[2] % 32))
            logits, off = predict(np.pad(crop, pad)[None].astype(np.float32))
            ch, cw = (crop.shape[1] + 1) // 2, (crop.shape[2] + 1) // 2
            prob = 1.0 / (1.0 + np.exp(-logits[0, 0, :ch, :cw]))
            region = heat[r0 // 2 : r0 // 2 + ch, c0 // 2 : c0 // 2 + cw]
            better = prob > region
            region[better] = prob[better]
            off_region = offset[:, r0 // 2 : r0 // 2 + ch, c0 // 2 : c0 // 2 + cw]
            off_region[:, better] = off[0][:, :ch, :cw][:, better]
    return heat, offset


def detections_from_peaks(
    peaks: Sequence[tuple[float, float, float]], mask: np.ndarray, *, threshold: float
) -> list[tuple[float, float, float]]:
    """Window (row, col, score) peaks at or above ``threshold`` on unmasked pixels."""
    height, width = mask.shape
    return [
        (r, c, s)
        for r, c, s in peaks
        if s >= threshold
        and 0 <= round(r) < height
        and 0 <= round(c) < width
        and not mask[round(r), round(c)]
    ]


class PointDetectorRunner:
    """The exported model, ready to run over (VV, VH) windows."""

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        tile: int = DEFAULT_TILE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.tile = tile
        self.overlap = overlap

    def predict(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, offset = self._session.run(["objectness", "offset"], {"image": batch})
        return logits, offset

    def detect(self, bands: np.ndarray, mask: np.ndarray) -> list[tuple[float, float, float]]:
        """Window (row, col, score) of vessels in raw (VV, VH) amplitude, mask honoured."""
        heat, offset = run_tiled(
            self.predict, normalise_chip(bands), tile=self.tile, overlap=self.overlap
        )
        peaks = decode_peaks(heat, offset, threshold=self.threshold)
        return detections_from_peaks(peaks, mask, threshold=self.threshold)
