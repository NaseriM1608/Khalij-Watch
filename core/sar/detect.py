"""Turn CFAR pixel hits into vessel candidates.

Pipeline: mask -> CFAR -> morphological cleanup -> connected components ->
geometry measurement -> plausibility filter -> georeferenced candidates.

The geometry filter is what separates a detector from a noise counter. Sea
clutter spikes are single pixels or blobs; ships are elongated objects of a
plausible size. Filtering on measured length, width and aspect ratio removes
most surviving false alarms at essentially no cost, and yields the length
estimate that later becomes a first-class model output.

A bright vessel's CFAR mask includes its sidelobes
--------------------------------------------------
A strong target throws a cross of sidelobes along azimuth and range, and
against quiet sea those sidelobes clear the CFAR threshold as well. Measuring
the whole mask therefore describes the cross rather than the ship. On a real
VH window over the Fujairah anchorage (Sentinel-1D, 2026-09-10, 192 vessels)
the mask gave a median extent of 289 m x 135 m, with 89% of widths above
80 m - broader than any ship afloat. Measuring only the pixels within 12 dB
of each target's peak amplitude gave 208 m x 62 m, with 16% above 80 m,
which is consistent with an anchorage of tankers and bulk carriers. Heading
alignment with the azimuth axis was the same either way, so headings follow
the hulls, not the sidelobes.

The 12 dB threshold is a heuristic chosen from that imagery, not a
calibration. Phase 4 supplies AIS lengths for matched vessels, which is the
ground truth needed to calibrate it or replace it with learned regression.
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

# Only pixels at or above this fraction of a target's peak amplitude count
# towards its extent and position. 0.25 in amplitude is -12 dB. See the
# module docstring for how it was chosen and why it is provisional.
EXTENT_PEAK_FRACTION = 0.25


@dataclass(frozen=True)
class Candidate:
    """A vessel-like object measured in a SAR scene.

    ``row``, ``col``, ``length_m``, ``width_m`` and ``heading_deg`` describe
    the target's hull, excluding sidelobes. ``pixel_count`` and the amplitude
    statistics cover the whole CFAR component. ``heading_deg`` is in the image
    frame until :func:`georeference` converts it to a compass bearing.
    """

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

    Only foreground pixels are visited. After CFAR and morphological opening a
    real scene window is overwhelmingly background, so iterating every pixel
    in Python - as an earlier version did - spent minutes on a multi-megapixel
    window doing nothing. ``np.nonzero`` yields foreground in raster order,
    which is exactly the order the first pass needs: every neighbour it
    consults (the row above, and the pixel to the left) has already been
    labelled.
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

    fg_rows, fg_cols = np.nonzero(mask)
    foreground = list(zip(fg_rows.tolist(), fg_cols.tolist(), strict=True))

    # First pass: provisional labels plus equivalences.
    for r, c in foreground:
        neighbours: list[int] = []
        if r > 0:
            for cc in (c - 1, c, c + 1):
                if 0 <= cc < cols:
                    above = int(labels[r - 1, cc])
                    if above:
                        neighbours.append(above)
        if c > 0:
            left = int(labels[r, c - 1])
            if left:
                neighbours.append(left)

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
    for r, c in foreground:
        root = find(int(labels[r, c]))
        if root not in remap:
            remap[root] = len(remap) + 1
        labels[r, c] = remap[root]
    return labels, len(remap)


def hull_pixels(
    rows_idx: np.ndarray,
    cols_idx: np.ndarray,
    values: np.ndarray,
    fraction: float = EXTENT_PEAK_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """The component pixels that belong to the target rather than its sidelobes.

    Keeps pixels at or above ``fraction`` of the component's peak amplitude.
    A fraction of zero, or a result too small to measure, returns the whole
    component unchanged.
    """
    if fraction <= 0.0 or values.size == 0:
        return rows_idx, cols_idx
    keep = values >= fraction * float(values.max())
    if int(keep.sum()) < 2:
        return rows_idx, cols_idx
    return rows_idx[keep], cols_idx[keep]


def measure_component(rows_idx: np.ndarray, cols_idx: np.ndarray) -> tuple[float, float, float]:
    """Measure a set of pixels' length, width and orientation by PCA.

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
    extent_peak_fraction: float = EXTENT_PEAK_FRACTION,
) -> list[Candidate]:
    """Run the full pixel-to-candidate pipeline on one window.

    ``extent_peak_fraction`` controls how much of each component counts as
    hull when measuring geometry; zero measures the raw CFAR mask.
    """
    config = config or CfarConfig()
    hits = cfar_detect(amplitude, config, mask=mask)
    cleaned = _binary_opening(hits)
    labels, count = label_components(cleaned)
    if count == 0:
        return []

    # Group foreground pixels by component once. Re-scanning the whole label
    # image for every component - as an earlier version did - is quadratic,
    # and a real coastal window easily holds hundreds of components.
    fg_rows, fg_cols = np.nonzero(labels)
    fg_labels = labels[fg_rows, fg_cols]
    order = np.argsort(fg_labels, kind="stable")
    fg_rows, fg_cols, fg_labels = fg_rows[order], fg_cols[order], fg_labels[order]
    boundaries = np.flatnonzero(np.diff(fg_labels)) + 1

    candidates: list[Candidate] = []
    for rows_idx, cols_idx in zip(
        np.split(fg_rows, boundaries), np.split(fg_cols, boundaries), strict=True
    ):
        if rows_idx.size == 0:
            continue
        values = amplitude[rows_idx, cols_idx]
        hull_rows, hull_cols = hull_pixels(rows_idx, cols_idx, values, extent_peak_fraction)
        length_m, width_m, heading = measure_component(hull_rows, hull_cols)
        if apply_geometry_filter and not is_plausible_vessel(length_m, width_m, rows_idx.size):
            continue
        candidates.append(
            Candidate(
                row=float(hull_rows.mean()),
                col=float(hull_cols.mean()),
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
    candidates: list[Candidate],
    georef: object,
    window_offset: tuple[int, int] = (0, 0),
) -> list[Candidate]:
    """Attach longitude, latitude and a true compass heading to candidates.

    ``georef`` is a :class:`core.sar.georef.SceneGeoref` - required for
    Sentinel-1 GRD scenes, which are located by ground control points - or an
    affine transform for a north-up lon/lat raster. ``window_offset`` is the
    (row, col) origin of the window within the scene, so candidates found in a
    tile map back to true scene coordinates.

    Heading is recomputed, not carried over. :func:`measure_component` reports
    it in the image frame, which is a compass bearing only if the image is
    north-up. GRD scenes are not: on a descending pass columns run east to
    west, which mirrors every heading.
    """
    from dataclasses import replace

    from core.sar.georef import SceneGeoref

    if not candidates:
        return []
    scene = georef if isinstance(georef, SceneGeoref) else SceneGeoref.from_affine(georef)

    row_off, col_off = window_offset
    rows = np.array([c.row + row_off for c in candidates], dtype=np.float64)
    cols = np.array([c.col + col_off for c in candidates], dtype=np.float64)
    lons, lats = scene.lonlat(rows, cols)

    # Inverse of measure_component's image-frame heading, h = atan2(dcol, -drow).
    radians = np.radians([c.heading_deg for c in candidates])
    headings = scene.bearings(rows, cols, -np.cos(radians), np.sin(radians)) % 180.0

    return [
        replace(
            candidate,
            row=float(row),
            col=float(col),
            lon=float(lon),
            lat=float(lat),
            heading_deg=float(heading),
        )
        for candidate, row, col, lon, lat, heading in zip(
            candidates, rows, cols, lons, lats, headings, strict=True
        )
    ]
