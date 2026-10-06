"""CPU benchmark of the product model on one 20 km window (Phase 8, efficiency).

Runs the exported model (fp32 ONNX, 256 px tiles, 64 px overlap) over a
synthetic two-band window of the size a 20 km box has at 10 m pixels (2000 x
2000), a few times, and records the seconds and the process's peak memory.
Synthetic sea clutter is enough: inference time does not depend on content.
Detection quality is measured elsewhere, on real passes.

Run:
    .venv-eval/Scripts/python scripts/bench_inference.py --out models/runs/kaggle-v2/bench.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Median sea amplitude (DN) measured on Sentinel-1 (PLAN §17, S1C/S1D check).
SEA_DN = (60.0, 30.0)
# Speckle of a 4.4-look GRD product: intensity is Gamma(L, mean / L).
LOOKS = 4.4
TARGETS = 20
TARGET_DN = (3000.0, 1200.0)


def synthetic_window(size: int, seed: int = 0) -> np.ndarray:
    """(VV, VH) amplitudes of sea clutter with a few bright, ship-like targets."""
    rng = np.random.default_rng(seed)
    bands = np.empty((2, size, size), dtype=np.float32)
    for i, level in enumerate(SEA_DN):
        intensity = rng.gamma(LOOKS, level**2 / LOOKS, size=(size, size))
        bands[i] = np.sqrt(intensity)
    for row, col in rng.integers(4, size - 4, size=(TARGETS, 2)):
        for i, level in enumerate(TARGET_DN):
            bands[i, row - 2 : row + 3, col - 1 : col + 2] = level
    return bands


def peak_memory_mb() -> float:
    """The process's peak memory so far: peak working set on Windows, max RSS elsewhere."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def main() -> int:
    from core.inference.runner import PointDetectorRunner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models" / "point_detector_kaggle-v2.onnx"
    )
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bands = synthetic_window(args.size)
    mask = np.zeros(bands.shape[1:], dtype=bool)
    runner = PointDetectorRunner(args.model)
    seconds = []
    for _ in range(args.repeats):
        began = time.perf_counter()
        found = runner.detect(bands, mask)
        seconds.append(round(time.perf_counter() - began, 2))
    result = {
        "model": args.model.as_posix(),
        "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()[:16],
        "tile": runner.tile,
        "overlap": runner.overlap,
        "window_px": args.size,
        "seconds": seconds,
        "median_s": statistics.median(seconds),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "detections": len(found),
        "machine": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
