"""Tests for the point-detector path of scripts/detect_window.py, with I/O stubbed."""

from __future__ import annotations

import argparse
import datetime as dt
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from core.db.spatial import BBox
from core.sar.windows import PixelWindow
from scripts import build_dataset
from scripts import detect_window as dw

BOX = BBox(56.0, 25.0, 57.0, 26.0)
PRODUCT = SimpleNamespace(acq_start=dt.datetime(2024, 5, 16, 2, 0, tzinfo=dt.UTC))


class FakeScene:
    def __init__(self):
        self.lonlat_calls = []

    def lonlat(self, rows, cols):
        self.lonlat_calls.append((list(rows), list(cols)))
        return np.full(len(rows), 56.5), np.full(len(rows), 25.5)


class FakeRunner:
    created: ClassVar[list[FakeRunner]] = []

    def __init__(self, _model_path):
        self.threshold, self.tile = 0.45, 128
        self.masks = []
        self.peaks = [(10.25, 20.75, 0.9), (30.0, 5.5, 0.6)]
        FakeRunner.created.append(self)

    def detect(self, _bands, mask):
        self.masks.append(mask.copy())
        return self.peaks


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    """read_pair, the masks, the model and the store step, recorded instead of run."""
    model = tmp_path / "point_detector_test.onnx"
    model.write_bytes(b"")
    state = SimpleNamespace(
        scene=FakeScene(),
        window=PixelWindow(300, 200, 40, 30),  # col_off, row_off, width, height
        coast_calls=0,
        stored=[],
    )
    FakeRunner.created = []

    def read_pair(*_args):
        if state.window.is_empty():  # as core.sar.windows.read_window does
            raise ValueError("cannot read an empty window")
        shape = (2, state.window.height, state.window.width)
        return state.scene, state.window, np.ones(shape, dtype=np.uint16)

    def coastline_mask(_box, _scene, window, _buffer_m):
        state.coast_calls += 1
        coast = np.zeros((window.height, window.width), dtype=bool)
        coast[:, :5] = True
        return coast

    def store_detections(_url, _product, _box, found, *_args, **_kwargs):
        state.stored.append(found)
        return len(found)

    monkeypatch.setattr(build_dataset, "read_pair", read_pair)
    monkeypatch.setattr(dw, "platform_mask", lambda *_args: (None, 0))
    monkeypatch.setattr(dw, "coastline_mask", coastline_mask)
    monkeypatch.setattr("core.inference.runner.PointDetectorRunner", FakeRunner)
    monkeypatch.setattr(dw, "store_detections", store_detections)
    state.args = argparse.Namespace(
        model=model, no_coastline=False, coast_buffer_m=500.0, no_store=False
    )
    return state


def run(stubs):
    return dw.run_model(stubs.args, SimpleNamespace(database_url=None), BOX, PRODUCT)


def test_the_coastline_is_masked_by_default(stubs):
    assert run(stubs) == 0

    assert stubs.coast_calls == 1
    assert FakeRunner.created[0].masks[0][:, :5].all()


def test_no_coastline_skips_the_coastline_mask(stubs):
    stubs.args.no_coastline = True

    assert run(stubs) == 0

    assert stubs.coast_calls == 0
    assert not FakeRunner.created[0].masks[0].any()


def test_an_empty_window_stops_before_the_model(stubs, capsys):
    stubs.window = PixelWindow(300, 200, 0, 30)

    assert run(stubs) == 1

    assert "does not intersect" in capsys.readouterr().out
    assert FakeRunner.created == []
    assert stubs.stored == []


def test_peaks_are_located_at_window_offset_plus_peak_position(stubs):
    assert run(stubs) == 0

    # Same convention as CFAR's georeference: scene pixel = window pixel + offset.
    assert stubs.scene.lonlat_calls == [([210.25, 230.0], [320.75, 305.5])]
    assert [p.confidence for p in stubs.stored[0]] == [0.9, 0.6]
