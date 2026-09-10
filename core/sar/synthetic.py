"""Synthetic SAR scenes with known ground truth.

Detector correctness cannot be established on real imagery, because nobody
knows exactly which pixels are vessels. Here the truth is constructed, so the
false-alarm rate and detection rate are both exactly measurable - which is the
only way to check that a CFAR's Pfa dial means what it claims.

The speckle model matters. SAR speckle is multiplicative, not additive: a
homogeneous surface imaged with ``L`` looks has intensity following a Gamma
distribution with shape ``L``. Sentinel-1 IW GRDH is roughly 4.4 looks.
Substituting Gaussian noise would make the sea far better behaved than it is
and flatter any detector tested on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Equivalent number of looks for Sentinel-1 IW GRDH.
IW_GRDH_LOOKS = 4.4

# Representative digital numbers for Sentinel-1 VV over the Gulf.
SEA_AMPLITUDE = 80.0
SHIP_AMPLITUDE = 1400.0
LAND_AMPLITUDE = 420.0


@dataclass(frozen=True)
class Target:
    """A vessel placed in a synthetic scene."""

    row: int
    col: int
    length_px: int
    width_px: int
    amplitude: float = SHIP_AMPLITUDE
    heading_deg: float = 0.0


@dataclass
class SyntheticScene:
    """A generated scene and the truth that produced it."""

    amplitude: np.ndarray
    truth: np.ndarray
    targets: list[Target] = field(default_factory=list)
    land: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.amplitude.shape


def speckle(mean_intensity: np.ndarray, looks: float, rng: np.random.Generator) -> np.ndarray:
    """Apply multiplicative Gamma speckle to a mean-intensity field.

    Normalised so the speckle has unit mean: the returned field has the same
    expected intensity as the input, with realistic variance around it.
    """
    noise = rng.gamma(shape=looks, scale=1.0 / looks, size=mean_intensity.shape)
    return mean_intensity * noise


def make_scene(
    shape: tuple[int, int] = (512, 512),
    *,
    targets: list[Target] | None = None,
    sea_amplitude: float = SEA_AMPLITUDE,
    looks: float = IW_GRDH_LOOKS,
    land_rows: int = 0,
    seed: int = 0,
) -> SyntheticScene:
    """Build a speckled sea scene with optional vessels and a coastal strip."""
    rng = np.random.default_rng(seed)
    rows, cols = shape

    mean_intensity = np.full(shape, sea_amplitude**2, dtype=np.float64)
    truth = np.zeros(shape, dtype=bool)
    land = np.zeros(shape, dtype=bool)

    if land_rows > 0:
        # Land along the top edge. Crucially it is HETEROGENEOUS: uniformly
        # bright land does not trigger a CFAR at all, because CFAR responds to
        # local contrast rather than absolute brightness. Real coastline sets
        # a detector off because of discrete bright structures - quays, tanks,
        # buildings - scattered through a darker background.
        mean_intensity[:land_rows, :] = LAND_AMPLITUDE**2
        structure_rng = np.random.default_rng(seed + 1)
        for _ in range(40):
            r = int(structure_rng.integers(0, land_rows))
            c = int(structure_rng.integers(0, cols))
            size = int(structure_rng.integers(2, 6))
            mean_intensity[r : r + size, c : c + size] = (LAND_AMPLITUDE * 6.0) ** 2
        land[:land_rows, :] = True

    placed: list[Target] = []
    for target in targets or []:
        half_len = max(1, target.length_px // 2)
        half_wid = max(1, target.width_px // 2)
        r0, r1 = max(0, target.row - half_wid), min(rows, target.row + half_wid + 1)
        c0, c1 = max(0, target.col - half_len), min(cols, target.col + half_len + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        mean_intensity[r0:r1, c0:c1] = target.amplitude**2
        truth[r0:r1, c0:c1] = True
        placed.append(target)

    intensity = speckle(mean_intensity, looks, rng)
    amplitude = np.sqrt(np.maximum(intensity, 0.0))

    return SyntheticScene(
        amplitude=amplitude.astype(np.float32),
        truth=truth,
        targets=placed,
        land=land if land_rows > 0 else None,
    )


def ship_grid(
    shape: tuple[int, int],
    *,
    rows: int = 3,
    cols: int = 3,
    length_px: int = 20,
    width_px: int = 5,
    amplitude: float = SHIP_AMPLITUDE,
    margin: int = 60,
) -> list[Target]:
    """Evenly spaced vessels, kept clear of the scene edge."""
    height, width = shape
    row_positions = np.linspace(margin, height - margin, rows).astype(int)
    col_positions = np.linspace(margin, width - margin, cols).astype(int)
    return [
        Target(int(r), int(c), length_px, width_px, amplitude)
        for r in row_positions
        for c in col_positions
    ]
