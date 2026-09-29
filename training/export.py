"""Export a trained point detector to ONNX, quantise it, and time it on this CPU.

ONNX opset 17 with dynamic height and width, so one file serves 128 px chips
and 2048 px scene tiles (PLAN.md section 7.1b). The INT8 file uses dynamic
quantisation as the plan asks; for a convolutional network it may gain little,
which the benchmark shows rather than assumes.

Run:
    python -m training.export --checkpoint models/runs/x/best.pt --out models/point_detector
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.model import PointDetector

OPSET = 17
OUTPUT_NAMES = ("objectness", "offset")


def load_model(checkpoint: Path) -> PointDetector:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PointDetector(state["config"]["encoder"], pretrained=False)
    model.load_state_dict(state["model"])
    return model.eval()


def export_onnx(model: PointDetector, path: Path, *, size: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 2, size, size)
    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=OPSET,
        input_names=["image"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "objectness": {0: "batch", 2: "out_height", 3: "out_width"},
            "offset": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        dynamo=False,
    )
    return path


def quantise(path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = path.with_name(path.stem + ".int8.onnx")
    quantize_dynamic(str(path), str(out), weight_type=QuantType.QInt8)
    return out


def max_difference(model: PointDetector, path: Path, *, size: int = 256, seed: int = 0) -> float:
    """Largest absolute objectness-logit difference between PyTorch and ONNX Runtime."""
    import onnxruntime as ort

    image = np.random.default_rng(seed).random((1, 2, size, size), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(image))[0].numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(["objectness"], {"image": image})[0]
    return float(np.abs(got - expected).max())


def benchmark(path: Path, *, size: int = 2048, runs: int = 3) -> float:
    """Median seconds per size x size tile on this CPU."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    image = np.random.default_rng(0).random((1, 2, size, size), dtype=np.float32)
    session.run(None, {"image": image})  # warm-up
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {"image": image})
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output path without extension")
    parser.add_argument("--tile", type=int, default=2048)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    fp32 = export_onnx(model, args.out.with_suffix(".onnx"))
    int8 = quantise(fp32)
    report = {
        "fp32": str(fp32),
        "int8": str(int8),
        "max_logit_difference_fp32": max_difference(model, fp32),
        "max_logit_difference_int8": max_difference(model, int8),
        f"seconds_per_{args.tile}px_tile_fp32": benchmark(fp32, size=args.tile),
        f"seconds_per_{args.tile}px_tile_int8": benchmark(int8, size=args.tile),
    }
    args.out.with_suffix(".export.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
