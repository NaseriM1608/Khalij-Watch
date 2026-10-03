"""Tests for the combined index's metadata: its version and the rules it states."""

from __future__ import annotations

from scripts.build_index import INDEX_VERSION, meta_for


def test_meta_reports_the_index_version_with_gfw_unmatched_positives():
    """Since run 2, detections GFW saw without AIS are positives: a new index version."""
    meta = meta_for([], ["khor_fakkan"], {"khor_fakkan": 0})

    assert INDEX_VERSION == "khalij-watch-index-v2"
    assert meta["version"] == INDEX_VERSION


def test_meta_rules_state_that_gfw_detections_without_ais_are_positive():
    rules = meta_for([], [], {})["rules"]["supervision"]

    assert "gfw_unmatched" in rules["positive"]
    assert "AIS" in rules["positive"]
    assert "gfw_unmatched" not in rules["ignore"]
    assert "gfw_absent" in rules["ignore"]
    assert "ambiguous" in rules["ignore"]
