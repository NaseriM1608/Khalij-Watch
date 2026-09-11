"""Stored detections must record the settings that produced them.

Pfa, guard and training sizes, the looks model and the hull threshold all
change what CFAR reports. Without a version string a stored row cannot be
compared with another run, or reproduced.
"""

from __future__ import annotations

from core.sar.cfar import CfarConfig
from core.sar.detect import detector_version

# The width of detections.detector_version.
COLUMN_WIDTH = 32


def test_detector_version_fits_the_database_column():
    awkward = CfarConfig(looks=4.35, pfa=2.5e-7, guard_half=124, train_half=240)

    assert len(detector_version(awkward, extent_peak_fraction=0.125)) <= COLUMN_WIDTH


def test_detector_version_changes_with_every_setting_that_changes_results():
    base = detector_version(CfarConfig())

    assert detector_version(CfarConfig(pfa=1e-5)) != base
    assert detector_version(CfarConfig(pfa=2.5e-6)) != detector_version(CfarConfig(pfa=2e-6))
    assert detector_version(CfarConfig(guard_half=20)) != base
    assert detector_version(CfarConfig(train_half=50)) != base
    assert detector_version(CfarConfig(looks=1.0)) != base
    assert detector_version(CfarConfig(), extent_peak_fraction=0.5) != base


def test_detector_version_is_stable_for_the_same_settings():
    assert detector_version(CfarConfig()) == detector_version(CfarConfig())
