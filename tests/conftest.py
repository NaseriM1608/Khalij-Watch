"""Unit tests never start a real upstream AIS connection."""

import pytest


@pytest.fixture(autouse=True)
def disable_live_ais_autostart(monkeypatch):
    monkeypatch.setenv("AIS_AUTOSTART_ENABLED", "0")
