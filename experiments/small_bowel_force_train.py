"""Five-fold 3D-ResNet force benchmark on all 50 recordings."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.small_bowel_force.loader import SmallBowelForceDataset  # noqa: E402
from dataset.small_bowel_force.splits import split_recordings  # noqa: E402
from evaluation.force_metrics import (  # noqa: E402
    bootstrap_recording_ci,
    recording_metrics,
    regression_metrics,
)
from models.force_regressor import ForceRegressor  # noqa: E402


RESULTS = ROOT / "results" / "small_bowel_force"
CHECKPOINTS = RESULTS / "checkpoints"
SEED = 2026


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(recordings: list[int], window: int, training: bool, batch_size: int, workers: int):
    dataset = SmallBowelForceDataset(recordings, window=window, augment=training)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


@torch.no_grad()
def predict(model: nn.Module, data: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    targets, predictions, recordings, frames = [], [], [], []
    elapsed = 0.0
    count = 0
    for batch in data:
        video = batch["video"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(video)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        count += len(video)
        targets.extend(batch["force"].numpy().tolist())
        predictions.extend(output.float().cpu().numpy().tolist())
        recordings.extend(batch["recording"].numpy().tolist())
        frames.extend(batch["frame"].numpy().tolist())
    return {
        "target": np.asarray(targets),
        "prediction": np.asarray(predictions),
        "recording": np.asarray(recordings, dtype=int),
        "frame": np.asarray(frames, dtype=int),
        "fps": np.asarray([count / max(elapsed, 1e-9)]),
    }


def evaluate_predictions(payload: dict[str, np.ndarray]) -> dict:
    aggregate = regression_metrics(payload["target"], payload["prediction"])
    per_recording = recording_metrics(
        payload["target"], payload["prediction"], payload["recording"]
    )
    cis = {
        metric: bootstrap_recording_ci(per_recording, metric)
        for metric in ("mae_n", "rmse_n", "nrmse", "r2", "pearson_r")
    }
    return {
        **aggregate,
        "fps": float(payload["fps"][0]),
        "recording_median": {
            metric: float(np.nanmedian([row[metric] for row in per_recording]))
            for metric in ("mae_n", "rmse_n", "nrmse", "r2", "pearson_r")
        },
        "recording_bootstrap_95_ci": cis,
        "per_recording": per_recording,
    }


def run_one(args, protocol: str, fold: int, window: int) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    run_name = f"{protocol}_fold{fold}_w{window}"
    result_path = RESULTS / f"{run_name}.json"
    if result_path.exists() and not args.overwrite:
        print(f"Skipping completed {run_name}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    seed_everything(SEED + fold + window)
    split = split_recordings(protocol, fold)
    effective_batch = max(2, args.batch_size * 10 // window)
    train_loader = loader(split["train"], window, True, effective_batch, args.workers)
    val_loader = loader(split["val"], window, False, effective_batch, args.workers)
    test_loader = loader(split["test"], window, False, effective_batch, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForceRegressor(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = CHECKPOINTS / f"{run_name}.pt"
    best_rmse = math.inf
    stale = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            video = batch["video"].to(device, non_blocking=True)
            target = batch["force"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(video)
                loss = loss_fn(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_payload = predict(model, val_loader, device)
        val_metrics = regression_metrics(val_payload["target"], val_payload["prediction"])
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "val_rmse_n": val_metrics["rmse_n"],
            }
        )
        print(
            f"{run_name} epoch {epoch:02d}: "
            f"train MSE={np.mean(losses):.4f}, val RMSE={val_metrics['rmse_n']:.4f} N",
            flush=True,
        )
        if val_metrics["rmse_n"] < best_rmse - 1e-4:
            best_rmse = val_metrics["rmse_n"]
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    test_payload = predict(model, test_loader, device)
    test_metrics = evaluate_predictions(test_payload)
    train_targets = np.asarray(
        [sample[2] for sample in train_loader.dataset.samples], dtype=float
    )
    mean_prediction = np.full_like(test_payload["target"], train_targets.mean())
    zero_prediction = np.zeros_like(test_payload["target"])
    result = {
        "run": run_name,
        "protocol": protocol,
        "fold": fold,
        "window": window,
        "split": split,
        "seed": SEED + fold + window,
        "best_epoch": int(state["epoch"]),
        "history": history,
        "model": test_metrics,
        "baselines": {
            "train_mean": regression_metrics(test_payload["target"], mean_prediction),
            "zero": regression_metrics(test_payload["target"], zero_prediction),
        },
        "predictions": [
            {
                "recording": int(recording),
                "frame": int(frame),
                "target_n": float(target),
                "prediction_n": float(prediction),
            }
            for recording, frame, target, prediction in zip(
                test_payload["recording"],
                test_payload["frame"],
                test_payload["target"],
                test_payload["prediction"],
            )
        ],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate(results: list[dict]) -> None:
    summary = []
    for protocol in ("geometry", "camera"):
        for window in (10, 20, 30):
            rows = [
                result
                for result in results
                if result["protocol"] == protocol and result["window"] == window
            ]
            if not rows:
                continue
            entry = {"protocol": protocol, "window": window}
            for metric in ("mae_n", "rmse_n", "nrmse", "r2", "pearson_r", "fps"):
                entry[metric] = {
                        "mean": float(np.mean([row["model"][metric] for row in rows])),
                        "sd": float(np.std([row["model"][metric] for row in rows], ddof=1)),
                }
            summary.append(entry)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Small-bowel video force estimation",
        "",
        "| Protocol | Window | MAE (N) | RMSE (N) | NRMSE | Pearson r | R² | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(metric: str) -> str:
            return f"{row[metric]['mean']:.3f}±{row[metric]['sd']:.3f}"

        lines.append(
            f"| {row['protocol']} | {row['window']} | {fmt('mae_n')} | "
            f"{fmt('rmse_n')} | {fmt('nrmse')} | {fmt('pearson_r')} | "
            f"{fmt('r2')} | {fmt('fps')} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "paper_tbme" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for protocol, color, reference in (
        ("geometry", "#4c78a8", [0.323, 0.342, 0.374]),
        ("camera", "#f58518", [0.318, 0.313, 0.328]),
    ):
        protocol_rows = [row for row in summary if row["protocol"] == protocol]
        ax.errorbar(
            [row["window"] for row in protocol_rows],
            [row["rmse_n"]["mean"] for row in protocol_rows],
            yerr=[row["rmse_n"]["sd"] for row in protocol_rows],
            marker="o",
            capsize=3,
            color=color,
            label=protocol,
        )
        ax.plot(
            [10, 20, 30],
            reference,
            color=color,
            linestyle="--",
            alpha=0.5,
        )
    ax.set_xlabel("Temporal window (frames)")
    ax.set_ylabel("Force RMSE (N)")
    ax.set_xticks([10, 20, 30])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_force_benchmark.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["geometry", "camera", "all"], default="all")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--window", type=int, choices=[10, 20, 30], default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocols = ("geometry", "camera") if args.protocol == "all" else (args.protocol,)
    folds = range(5) if args.fold < 0 else (args.fold,)
    windows = (10, 20, 30) if args.protocol == "all" and args.fold < 0 else (args.window,)
    results = [
        run_one(args, protocol, fold, window)
        for protocol in protocols
        for window in windows
        for fold in folds
    ]
    aggregate(results)


if __name__ == "__main__":
    main()
