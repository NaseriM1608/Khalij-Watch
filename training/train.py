"""Train the point detector on the chip index, checkpointing every epoch.

Resumes from ``<out>/last.pt`` when present, because Kaggle sessions stop at
12 hours (PLAN.md section 7.4). Validation decodes peaks on the val split and
scores them against every known vessel in each chip within ``TOLERANCE_PX``;
predictions on ignored ground (unidentified detections, unsupervised parts of
centre negatives) are neither right nor wrong.

Run (CPU smoke test):
    python -m training.train --index data/datasets/index.jsonl --root data/datasets \
        --out models/runs/smoke --epochs 1 --samples-per-epoch 256 --batch-size 16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.inference.decode import decode_peaks, match_points
from core.labeling.targets import output_position
from training.data import ChipDataset, load_index, sampling_weight
from training.losses import OBJECTNESS_LOSSES, offset_loss
from training.model import PointDetector

# 100 m at 10 m pixels: labels are CFAR centroids of hulls up to ~300 m long.
TOLERANCE_PX = 10.0
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.1, 0.95, 0.05))


@dataclass
class Config:
    index: str
    root: str
    out: str
    encoder: str = "efficientnet_b0"
    pretrained: bool = False
    objectness_loss: str = "reduced_focal"
    offset_weight: float = 1.0
    epochs: int = 20
    samples_per_epoch: int = 40_000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 2
    max_val_chips: int = 0
    seed: int = 0


def make_loaders(config: Config) -> tuple[DataLoader, DataLoader, list[dict]]:
    train_records = load_index(Path(config.index), "train")
    val_records = load_index(Path(config.index), "val")
    if config.max_val_chips:
        rng = np.random.default_rng(config.seed)
        keep = rng.choice(
            len(val_records), size=min(config.max_val_chips, len(val_records)), replace=False
        )
        val_records = [val_records[i] for i in sorted(keep)]
    sampler = WeightedRandomSampler(
        [sampling_weight(r) for r in train_records],
        num_samples=config.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train = DataLoader(
        ChipDataset(train_records, Path(config.root), augment=True, seed=config.seed),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
        drop_last=True,
    )
    val = DataLoader(
        ChipDataset(val_records, Path(config.root), augment=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
    )
    return train, val, val_records


def step_loss(
    model: PointDetector, batch: dict, config: Config, device: torch.device
) -> torch.Tensor:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    logits, offsets = model(batch["image"])
    objectness = OBJECTNESS_LOSSES[config.objectness_loss](
        logits, batch["heatmap"], batch["weight"]
    )
    return objectness + config.offset_weight * offset_loss(
        offsets, batch["offset"], batch["offset_mask"]
    )


def chip_counts(
    heat: np.ndarray, offsets: np.ndarray, targets: dict[str, np.ndarray]
) -> dict[float, tuple[int, int, int]]:
    """(tp, fp, fn) per threshold for one chip, ignoring unsupervised ground."""
    weight = targets["weight"]
    truth = [
        (r, c) for r, c, _ in decode_peaks(targets["heatmap"], targets["offset"], threshold=0.99)
    ]

    def supervised(row: float, col: float) -> bool:
        cell_r, cell_c = output_position(row)[0], output_position(col)[0]
        size = weight.shape[0]
        return 0 <= cell_r < size and 0 <= cell_c < size and weight[cell_r, cell_c] > 0

    peaks = decode_peaks(heat, offsets, threshold=THRESHOLDS[0])
    return {
        t: match_points(
            [p for p in peaks if p[2] >= t], truth, tolerance=TOLERANCE_PX, counts=supervised
        )
        for t in THRESHOLDS
    }


@torch.no_grad()
def evaluate(model: PointDetector, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Point precision, recall and F1 at each threshold; best F1 and its threshold."""
    model.eval()
    counts = {t: [0, 0, 0] for t in THRESHOLDS}
    for batch in loader:
        logits, offsets = model(batch["image"].to(device))
        heat = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        offsets = offsets.float().cpu().numpy()
        for i in range(heat.shape[0]):
            targets = {
                "weight": batch["weight"][i, 0].numpy(),
                "heatmap": batch["heatmap"][i, 0].numpy(),
                "offset": batch["offset"][i].numpy(),
            }
            for t, (tp, fp, fn) in chip_counts(heat[i], offsets[i], targets).items():
                counts[t][0] += tp
                counts[t][1] += fp
                counts[t][2] += fn
    scores = {}
    for t, (tp, fp, fn) in counts.items():
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores[t] = (2 * precision * recall / max(precision + recall, 1e-9), precision, recall)
    best = max(scores, key=lambda t: scores[t][0])
    return {
        "f1": scores[best][0],
        "precision": scores[best][1],
        "recall": scores[best][2],
        "threshold": best,
        "f1_at_0.5": scores[0.5][0],
    }


def train(config: Config) -> dict[str, float]:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, _ = make_loaders(config)

    model = PointDetector(config.encoder, pretrained=config.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start, best_f1, history = 0, -1.0, []
    if (out / "last.pt").is_file():
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best_f1, history = state["epoch"] + 1, state["best_f1"], state["history"]
        print(f"resumed after epoch {state['epoch']}", flush=True)

    for epoch in range(start, config.epochs):
        model.train()
        started, total = time.perf_counter(), 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = step_loss(model, batch, config, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss)
        metrics = {
            "epoch": epoch,
            "loss": total / len(train_loader),
            **evaluate(model, val_loader, device),
        }
        metrics["seconds"] = time.perf_counter() - started
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        best_f1 = max(best_f1, metrics["f1"])
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "history": history,
            "config": asdict(config),
        }
        torch.save(state, out / "last.pt")
        if metrics["f1"] >= best_f1:
            torch.save(
                {"model": model.state_dict(), "config": asdict(config), "metrics": metrics},
                out / "best.pt",
            )
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return history[-1] if history else {}


_ARG_TYPES = {"str": str, "int": int, "float": float}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Config):
        flag = f"--{field.name.replace('_', '-')}"
        if field.type == "bool":
            parser.add_argument(flag, action="store_true")
        elif field.default is dataclasses.MISSING:
            parser.add_argument(flag, required=True)
        else:
            parser.add_argument(flag, type=_ARG_TYPES[field.type], default=field.default)
    train(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
