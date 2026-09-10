"""Turn CFAR pixel hits into vessel candidates.

Pipeline: mask -> CFAR -> morphological cleanup -> connected components ->
geometry measurement -> plausibility filter -> georeferenced candidates.

The geometry filter is what separates a detector from a noise counter. Sea
clutter spikes are single pixels or blobs; ships are elongated objects of a
plausible size. Filtering on measured length, width and aspect ratio removes
most surviving false alarms at essentially no cost, and yields the length
estimate that later becomes a first-class model output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.sar.cfar import CfarConfig, cfar_detect

# Sentinel-1 IW GRDH ground sample distance.
PIXEL_SPACING_M = 10.0

# Plausible vessel geometry. The lower bound is set by resolution rather than
# by naval architecture: below ~2 pixels there is nothing to measure. The
# upper bound is a little beyond the largest ULCC afloat.
MIN_LENGTH_M = 15.0
MAX_LENGTH_M = 450.0
MIN_ASPECT_RATIO = 1.3
MAX_ASPECT_RATIO = 14.0

# Discard components smaller than this: a lone bright pixel is speckle.
MIN_PIXELS = 3


@dataclass(frozen=True)
class Candidate:
    """A vessel-like object measured in a SAR scene."""

    row: float
    col: float
    length_m: float
    width_m: float
    heading_deg: float
    pixel_count: int
    peak_amplitude: float
    mean_amplitude: float
    lon: float | None = None
    lat: float | None = None

    @property
    def aspect_ratio(self) -> float:
        return self.length_m / max(self.width_m, PIXEL_SPACING_M)


def _binary_opening(mask: np.ndarray) -> np.ndarray:
    """Erode then dilate with a 3x3 cross, removing isolated pixels.

    Written directly rather than via scipy.ndimage so the detector has no
    heavy dependency; for a 3x3 cross the shifted-array form is both short and
    fast.
    """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    # Erosion: a pixel survives only if it and its 4-neighbours are set.
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    padded = np.pad(eroded, 1, mode="constant", constant_values=False)
    # Dilation restores the shape of whatever survived.
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected components with a two-pass union-find.

    Returns the label image (0 = background) and the number of components.
    """
    rows, cols = mask.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # First pass: provisional labels plus equivalences.
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            neighbours = []
            if r > 0:
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if 0 <= cc < cols and labels[r - 1, cc]:
                        neighbours.append(labels[r - 1, cc])
            if c > 0 and labels[r, c - 1]:
                neighbours.append(labels[r, c - 1])

            if not neighbours:
                parent.append(len(parent))
                labels[r, c] = len(parent) - 1
            else:
                smallest = min(neighbours)
                labels[r, c] = smallest
                for other in neighbours:
                    union(smallest, other)

    # Second pass: collapse equivalences to dense consecutive ids.
    remap: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            if not labels[r, c]:
                continue
            root = find(labels[r, c])
            if root not in remap:
                remap[root] = len(remap) + 1
            labels[r, c] = remap[root]
    return labels, len(remap)


def measure_component(rows_idx: np.ndarray, cols_idx: np.ndarray) -> tuple[float, float, float]:
    """Measure a component's length, width and orientation by PCA.

    The principal axis of the pixel cloud gives the vessel's heading, and the
    extent of the pixels projected onto each axis gives length and width.
    Projected extent is used rather than an eigenvalue-derived radius because
    it is robust for the very small pixel counts typical here.

    Heading is returned modulo 180 degrees: a SAR return is symmetric, so the
    bow cannot be distinguished from the stern.
    """
    if len(rows_idx) < 2:
        return PIXEL_SPACING_M, PIXEL_SPACING_M, 0.0

    points = np.column_stack([cols_idx.astype(np.float64), rows_idx.astype(np.float64)])
    centred = points - points.mean(axis=0)

    covariance = np.cov(centred, rowvar=False)
    if not np.all(np.isfinite(covariance)):
        return PIXEL_SPACING_M, PIXEL_SPACING_M, 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    minor = eigenvectors[:, int(np.argmin(eigenvalues))]

    along = centred @ major
    across = centred @ minor
    length = (along.max() - along.min() + 1.0) * PIXEL_SPACING_M
    width = (across.max() - across.min() + 1.0) * PIXEL_SPACING_M

    # Image y grows downward, so negate to get a compass-style bearing.
    heading = math.degrees(math.atan2(major[0], -major[1])) % 180.0
    return length, width, heading


def is_plausible_vessel(length_m: float, width_m: float, pixel_count: int) -> bool:
    """Reject components that cannot be a ship."""
    if pixel_count < MIN_PIXELS:
        return False
    if not MIN_LENGTH_M <= length_m <= MAX_LENGTH_M:
        return False
    aspect = length_m / max(width_m, PIXEL_SPACING_M)
    return MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO


def detect_candidates(
    amplitude: np.ndarray,
    *,
    config: CfarConfig | None = None,
    mask: np.ndarray | None = None,
    apply_geometry_filter: bool = True,
) -> list[Candidate]:
    """Run the full pixel-to-candidate pipeline on one window."""
    config = config or CfarConfig()
    hits = cfar_detect(amplitude, config, mask=mask)
    cleaned = _binary_opening(hits)
    labels, count = label_components(cleaned)
    if count == 0:
        return []

    candidates: list[Candidate] = []
    for label in range(1, count + 1):
        rows_idx, cols_idx = np.nonzero(labels == label)
        if rows_idx.size == 0:
            continue
        length_m, width_m, heading = measure_component(rows_idx, cols_idx)
        if apply_geometry_filter and not is_plausible_vessel(length_m, width_m, rows_idx.size):
            continue
        values = amplitude[rows_idx, cols_idx]
        candidates.append(
            Candidate(
                row=float(rows_idx.mean()),
                col=float(cols_idx.mean()),
                length_m=length_m,
                width_m=width_m,
                heading_deg=heading,
                pixel_count=int(rows_idx.size),
                peak_amplitude=float(values.max()),
                mean_amplitude=float(values.mean()),
            )
        )
    return candidates


def georeference(
    candidates: list[Candidate], transform, window_offset: tuple[int, int] = (0, 0)
) -> list[Candidate]:
    """Attach lon/lat to candidates using a raster affine transform.

    ``window_offset`` is the (row, col) origin of the window within the scene,
    so candidates found in a tile map back to true scene coordinates.
    """
    from rasterio.transform import xy

    row_off, col_off = window_offset
    located: list[Candidate] = []
    for candidate in candidates:
        # rasterio.transform.xy targets the pixel centre by default.
        x, y = xy(transform, candidate.row + row_off, candidate.col + col_off)
        located.append(
            Candidate(
                row=candidate.row + row_off,
                col=candidate.col + col_off,
                length_m=candidate.length_m,
                width_m=candidate.width_m,
                heading_deg=candidate.heading_deg,
                pixel_count=candidate.pixel_count,
                peak_amplitude=candidate.peak_amplitude,
                mean_amplitude=candidate.mean_amplitude,
                lon=float(x),
                lat=float(y),
            )
        )
    return located
