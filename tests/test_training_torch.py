"""Tests for the PyTorch side of training: model shapes, losses, a learning step, export.

Skipped unless torch and timm are installed (training machines and smoke runs).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from core.inference.contract import CHIP_SIZE, OUTPUT_SIZE  # noqa: E402
from core.labeling.targets import encode_targets  # noqa: E402
from training.losses import offset_loss, reduced_focal_loss, weighted_bce_loss  # noqa: E402
from training.model import PointDetector  # noqa: E402
from training.train import chip_counts  # noqa: E402


def small_model() -> PointDetector:
    torch.manual_seed(0)
    return PointDetector("efficientnet_b0", pretrained=False)


def batch_of(record: dict, n: int = 2) -> dict:
    t = encode_targets(record)
    return {
        "heatmap": torch.from_numpy(np.stack([t["heatmap"][None]] * n)),
        "weight": torch.from_numpy(np.stack([t["weight"][None]] * n)),
        "offset": torch.from_numpy(np.stack([t["offset"]] * n)),
        "offset_mask": torch.from_numpy(np.stack([t["offset_mask"][None]] * n)),
    }


POSITIVE = {"supervision": "positive", "row": 1000.0, "col": 2000.0, "objects": []}


def test_output_is_half_resolution_for_chips_and_tiles():
    model = small_model().eval()
    with torch.no_grad():
        logits, offsets = model(torch.zeros(1, 2, CHIP_SIZE, CHIP_SIZE))
        big, _ = model(torch.zeros(1, 2, 256, 320))

    assert logits.shape == (1, 1, OUTPUT_SIZE, OUTPUT_SIZE)
    assert offsets.shape == (1, 2, OUTPUT_SIZE, OUTPUT_SIZE)
    assert big.shape == (1, 1, 128, 160)


def test_initial_objectness_is_near_the_prior():
    model = small_model().eval()
    with torch.no_grad():
        logits, _ = model(torch.rand(1, 2, CHIP_SIZE, CHIP_SIZE))

    assert float(torch.sigmoid(logits).mean()) == pytest.approx(0.01, abs=0.02)


def test_losses_are_zero_where_weight_is_zero():
    batch = batch_of(POSITIVE)
    logits = torch.full_like(batch["heatmap"], 5.0)
    zero = torch.zeros_like(batch["weight"])

    assert float(reduced_focal_loss(logits, batch["heatmap"], zero)) == 0.0
    assert float(weighted_bce_loss(logits, batch["heatmap"], zero)) == 0.0


def test_focal_loss_prefers_the_right_heatmap():
    batch = batch_of(POSITIVE)
    good = torch.logit(batch["heatmap"].clamp(1e-4, 1 - 1e-4))
    bad = torch.zeros_like(good)

    assert float(reduced_focal_loss(good, batch["heatmap"], batch["weight"])) < float(
        reduced_focal_loss(bad, batch["heatmap"], batch["weight"])
    )


def test_offset_loss_only_counts_peak_cells():
    batch = batch_of(POSITIVE)
    pred = batch["offset"].clone()
    pred[:, :, 0, 0] = 9.0  # far from the peak: must not count

    assert float(offset_loss(pred, batch["offset"], batch["offset_mask"])) == pytest.approx(0.0)


def test_a_few_steps_on_one_batch_reduce_the_loss():
    model = small_model().train()
    batch = batch_of(POSITIVE, n=4)
    image = torch.rand(4, 2, CHIP_SIZE, CHIP_SIZE)
    image[:, :, 60:68, 62:66] = 0.9
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        logits, offsets = model(image)
        loss = reduced_focal_loss(logits, batch["heatmap"], batch["weight"]) + offset_loss(
            offsets, batch["offset"], batch["offset_mask"]
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0]


def test_chip_counts_score_a_perfect_prediction():
    targets = encode_targets({**POSITIVE, "objects": [[0.0, 30.0, "unlabelled"]]})
    heat = targets["heatmap"].copy()
    heat[32, 47] = 0.95  # a confident peak on the unlabelled neighbour: not an error

    counts = chip_counts(heat, targets["offset"], targets)

    assert counts[0.5] == (1, 0, 0)


def test_threshold_grid_reaches_below_a_tenth():
    """Late epochs of kaggle-v1 all peaked at 0.10, the old grid's floor."""
    from training.train import THRESHOLDS

    assert min(THRESHOLDS) < 0.1
    assert list(THRESHOLDS) == sorted(THRESHOLDS)


def test_evaluate_checkpoint_reports_overall_and_per_area(tmp_path):
    import json

    from training.evaluate import evaluate_checkpoint

    chips = np.full((2, 2, CHIP_SIZE, CHIP_SIZE), 40, dtype=np.uint16)
    chips[0, :, 60:68, 62:66] = 1500
    (tmp_path / "a" / "shards").mkdir(parents=True)
    np.savez(tmp_path / "a" / "shards" / "S.npz", chips=chips, chip_ids=np.array(["S/d0", "S/n0"]))
    base = {"scene": "S.SAFE", "split": "test", "shard": "a/shards/S.npz", "row": 10.0, "col": 20.0}
    records = [
        {
            **base,
            "chip_id": "a/S/d0",
            "area": "a",
            "index": 0,
            "supervision": "positive",
            "objects": [],
        },
        {
            **base,
            "chip_id": "a/S/n0",
            "area": "a",
            "index": 1,
            "supervision": "full_negative",
            "objects": [],
            "source": "sea",
        },
    ]
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    model = small_model()
    torch.save(
        {"model": model.state_dict(), "config": {"encoder": "efficientnet_b0"}}, tmp_path / "m.pt"
    )

    report = evaluate_checkpoint(
        tmp_path / "m.pt", tmp_path / "index.jsonl", tmp_path, split="test", batch_size=2, workers=0
    )

    assert set(report) == {"all", "a"}
    assert report["all"]["truth"] == 1
    assert {"f1", "precision", "recall", "threshold"} <= set(report["all"]["best"])
    assert min(float(t) for t in report["all"]["by_threshold"]) < 0.1


def test_resume_refuses_a_changed_learning_rate_schedule():
    """OneCycleLR fixes its total steps; a checkpoint cannot follow a different plan."""
    from training.train import Config, check_resumable

    config = Config(index="i", root="r", out="o", epochs=2)
    saved = {**{f: getattr(config, f) for f in ("epochs", "samples_per_epoch", "batch_size", "lr")}}

    check_resumable({**saved}, config)
    with pytest.raises(ValueError, match="epochs 1 -> 2"):
        check_resumable({**saved, "epochs": 1}, config)


def test_resume_refuses_a_changed_model_loss_or_crop():
    """What is trained must match the checkpoint; housekeeping fields may change."""
    from dataclasses import asdict, replace

    from training.train import Config, check_resumable

    config = Config(index="i", root="r", out="o", crop=96)
    saved = asdict(config)

    check_resumable(
        {**saved, "keep_epochs": True, "workers": 0, "max_val_chips": 5, "out": "x"}, config
    )
    run_one = {k: v for k, v in saved.items() if k != "crop"}  # saved before crop existed
    check_resumable(run_one, replace(config, crop=0))
    for field, value in (
        ("crop", 0),
        ("objectness_loss", "bce"),
        ("offset_weight", 2.0),
        ("weight_decay", 0.0),
        ("encoder", "resnet18"),
    ):
        with pytest.raises(ValueError, match=field):
            check_resumable(asdict(replace(config, **{field: value})), config)


def tiny_index(tmp_path, splits: dict[str, int]) -> None:
    """index.jsonl and one shard: per split, chips numbered in order, area a<n>."""
    import json

    total = sum(splits.values())
    chips = np.full((total, 2, CHIP_SIZE, CHIP_SIZE), 40, dtype=np.uint16)
    chips[:, :, 62:66, 62:66] = 1500
    (tmp_path / "shards").mkdir()
    np.savez(tmp_path / "shards" / "S.npz", chips=chips, chip_ids=np.arange(total).astype(str))
    records, n = [], 0
    for split, count in splits.items():
        for _ in range(count):
            records.append(
                {
                    "chip_id": f"S/{n}",
                    "area": f"a{n}",
                    "scene": "S.SAFE",
                    "split": split,
                    "shard": "shards/S.npz",
                    "index": n,
                    "row": 1000.0,
                    "col": 2000.0,
                    "supervision": "positive",
                    "source": "gfw",
                    "objects": [],
                }
            )
            n += 1
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def tiny_config(tmp_path, **changes):
    from training.train import Config

    small = {"epochs": 2, "samples_per_epoch": 2, "batch_size": 2, "workers": 0}
    return Config(
        index=str(tmp_path / "index.jsonl"),
        root=str(tmp_path),
        out=str(tmp_path / "run"),
        **{**small, **changes},
    )


def test_a_resumed_epoch_draws_a_new_sample_stream(tmp_path):
    """Neither the sampler nor the augmentation may replay epoch 0 after a resume."""
    from training.train import make_loaders

    tiny_index(tmp_path, {"train": 40, "val": 1})
    config = tiny_config(tmp_path, samples_per_epoch=64, epochs=4)

    fresh, _, _ = make_loaders(config)
    resumed, _, _ = make_loaders(config, start_epoch=3)

    assert list(resumed.sampler) != list(fresh.sampler)
    assert resumed.dataset.seed != fresh.dataset.seed


def test_a_fresh_run_keeps_its_sample_stream(tmp_path):
    from torch.utils.data import WeightedRandomSampler

    from training.data import load_index, sampling_weight
    from training.train import make_loaders

    tiny_index(tmp_path, {"train": 40, "val": 1})
    config = tiny_config(tmp_path, samples_per_epoch=64, seed=5)
    records = load_index(tmp_path / "index.jsonl", "train")
    before = WeightedRandomSampler(
        [sampling_weight(r) for r in records],
        num_samples=64,
        replacement=True,
        generator=torch.Generator().manual_seed(5),
    )

    fresh, _, _ = make_loaders(config)

    assert list(fresh.sampler) == list(before)
    assert fresh.dataset.seed == 5


def test_each_loader_worker_gets_its_own_augmentation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import training.train as train_module

    tiny_index(tmp_path, {"train": 2, "val": 1})
    loader, _, _ = train_module.make_loaders(tiny_config(tmp_path))
    dataset = loader.dataset
    monkeypatch.setattr(
        train_module, "get_worker_info", lambda: SimpleNamespace(id=1, dataset=dataset)
    )

    loader.worker_init_fn(1)

    assert dataset.worker == 1


class Unpicklable:
    def __reduce__(self):
        raise RuntimeError("stopped mid-write")


def test_atomic_save_leaves_the_old_file_when_a_write_fails(tmp_path):
    from training.train import save_atomic

    path = tmp_path / "last.pt"
    save_atomic({"epoch": 1}, path)

    with pytest.raises(RuntimeError, match="mid-write"):
        save_atomic({"epoch": 2, "bad": Unpicklable()}, path)

    assert torch.load(path, weights_only=False) == {"epoch": 1}


def test_training_keeps_every_epoch_and_writes_last_after_the_others(monkeypatch, tmp_path):
    """last.pt records the epoch as done, so it must be the final write of each epoch."""
    from pathlib import Path

    from training.export import load_model
    from training.train import train

    tiny_index(tmp_path, {"train": 2, "val": 1})
    written = []
    real_save = torch.save

    def recording_save(obj, path, *args, **kwargs):
        written.append(Path(path).name.removesuffix(".tmp"))
        return real_save(obj, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", recording_save)

    train(tiny_config(tmp_path, keep_epochs=True))

    run = tmp_path / "run"
    assert written[0] == "epoch_00.pt" and written[-1] == "last.pt"
    assert [i for i, name in enumerate(written) if name == "last.pt"] == [
        written.index("epoch_01.pt") - 1,
        len(written) - 1,
    ]
    for name in ("epoch_00.pt", "epoch_01.pt", "best.pt", "last.pt"):
        load_model(run / name)
    assert not list(run.glob("*.tmp"))
    assert "scaler" in torch.load(run / "last.pt", weights_only=False)


def test_resume_restores_the_scaler_and_accepts_checkpoints_without_it(monkeypatch, tmp_path):
    """CPU runs keep the scaler disabled; old checkpoints have no scaler state at all."""
    import shutil

    import training.train as train_module

    tiny_index(tmp_path, {"train": 2, "val": 1})
    config = tiny_config(tmp_path)
    last, after_first = tmp_path / "run" / "last.pt", tmp_path / "after_epoch_0.pt"
    real_save = train_module.save_atomic

    def keep_first_epoch(obj, path):
        real_save(obj, path)
        if path.name == "last.pt" and obj["epoch"] == 0:
            shutil.copy(path, after_first)

    monkeypatch.setattr(train_module, "save_atomic", keep_first_epoch)
    train_module.train(config)

    for drop_scaler in (True, False):
        state = torch.load(after_first, weights_only=False)
        if drop_scaler:
            del state["scaler"]
        torch.save(state, last)

        metrics = train_module.train(config)

        assert metrics["epoch"] == 1
        assert "scaler" in torch.load(last, weights_only=False)


def test_evaluate_max_chips_samples_like_training_validation(tmp_path):
    """The index is ordered by area, so the first chips would all come from one area."""
    from training.evaluate import evaluate_checkpoint

    tiny_index(tmp_path, {"test": 10})
    torch.save(
        {"model": small_model().state_dict(), "config": {"encoder": "efficientnet_b0"}},
        tmp_path / "m.pt",
    )

    report = evaluate_checkpoint(
        tmp_path / "m.pt", tmp_path / "index.jsonl", tmp_path, split="test", max_chips=3
    )

    keep = sorted(np.random.default_rng(0).choice(10, size=3, replace=False))
    assert set(report) - {"all"} == {f"a{i}" for i in keep}


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnxruntime")
    from training.export import export_onnx, max_difference

    model = small_model().eval()
    path = export_onnx(model, tmp_path / "m.onnx")

    assert max_difference(model, path, size=128) < 1e-3
