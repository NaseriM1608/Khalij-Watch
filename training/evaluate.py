"""Score a trained checkpoint on one split: point P/R/F1 per threshold, overall and per area.

Uses the same matching as training (core.inference.decode within
``TOLERANCE_PX``; predictions on ignored ground count neither way) so numbers
are comparable with the training history, over the same threshold grid, and
also reports each area separately.

Run:
    python -m training.evaluate --checkpoint models/runs/kaggle-v1/run/best.pt \
        --index data/datasets/index.jsonl --root data/datasets --split test \
        --out models/runs/kaggle-v1/eval_test_best.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.data import ChipDataset, load_index
from training.export import load_model
from training.train import THRESHOLDS, chip_counts


def _summary(counts: dict[float, list[int]]) -> dict[str, Any]:
    by_threshold = {}
    for t, (tp, fp, fn) in counts.items():
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        by_threshold[str(t)] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    best_t = max(by_threshold, key=lambda t: by_threshold[t]["f1"])
    first = next(iter(counts.values()))
    return {
        "truth": first[0] + first[2],
        "best": {"threshold": float(best_t), **by_threshold[best_t]},
        "by_threshold": by_threshold,
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    index: Path,
    root: Path,
    *,
    split: str,
    thresholds: tuple[float, ...] = THRESHOLDS,
    batch_size: int = 64,
    workers: int = 0,
    max_chips: int = 0,
) -> dict[str, dict[str, Any]]:
    """{"all": summary, <area>: summary, ...} for one split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(checkpoint)).to(device)
    records = load_index(Path(index), split)
    if max_chips:
        # A seeded random subset, as train.py takes max_val_chips: the index is ordered by area.
        rng = np.random.default_rng(0)
        keep = rng.choice(len(records), size=min(max_chips, len(records)), replace=False)
        records = [records[i] for i in sorted(keep)]
    loader = DataLoader(
        ChipDataset(records, Path(root), augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    counts: dict[str, dict[float, list[int]]] = defaultdict(
        lambda: {t: [0, 0, 0] for t in thresholds}
    )
    position = 0
    for batch in loader:
        logits, offsets = model(batch["image"].to(device))
        heat = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        offsets = offsets.float().cpu().numpy()
        for i in range(heat.shape[0]):
            area = records[position]["area"]
            position += 1
            targets = {
                "weight": batch["weight"][i, 0].numpy(),
                "heatmap": batch["heatmap"][i, 0].numpy(),
                "offset": batch["offset"][i].numpy(),
            }
            for t, (tp, fp, fn) in chip_counts(heat[i], offsets[i], targets, thresholds).items():
                for key in ("all", area):
                    counts[key][t][0] += tp
                    counts[key][t][1] += fp
                    counts[key][t][2] += fn
    return {
        key: _summary(value) for key, value in sorted(counts.items(), key=lambda kv: kv[0] != "all")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-chips", type=int, default=0)
    args = parser.parse_args()

    report = evaluate_checkpoint(
        args.checkpoint,
        args.index,
        args.root,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        max_chips=args.max_chips,
    )
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for area, summary in report.items():
        b = summary["best"]
        print(
            f"{area:16s} truth {summary['truth']:5d}  best F1 {b['f1']:.3f} "
            f"(P {b['precision']:.3f} R {b['recall']:.3f} at {b['threshold']:.2f})"
        )


if __name__ == "__main__":
    main()
