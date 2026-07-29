"""Train the visual material initializer on the frozen sim_v3 split.

Clinical ION frames are intentionally not loaded here: they have no confirmed
mechanical labels and remain pending visual de-identification QC.  This script
uses only synthetic ground truth to learn physically bounded initialization
parameters for the subsequent optimizer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hierarchical_material_initializer import (  # noqa: E402
    HierarchicalMaterialInitializer,
    physics_ready_predictions,
)


DATASET = ROOT / "dataset" / "sim_v3"
RESULTS = ROOT / "results" / "hierarchical_material_prior"
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class SimV3PriorDataset(Dataset):
    def __init__(self, split: str, frames: tuple[int, ...] = (2, 5, 8)) -> None:
        manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        self.rows = [row for row in manifest["scenes"] if row["split"] == split]
        self.frames = frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.rows[index]
        scene_dir = DATASET / f"scene_{row['id']:04d}"
        ground_truth = torch.load(scene_dir / "gt.pt", weights_only=False)
        images = []
        grayscale_images = []
        for frame in self.frames:
            image = np.asarray(
                Image.open(scene_dir / "images" / f"frame_{frame:02d}.png").convert("RGB"),
                dtype=np.float32,
            )
            tensor = torch.from_numpy(image).permute(2, 0, 1) / 255.0
            images.append((tensor - MEAN) / STD)
            grayscale_images.append(
                cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            )
        flow_magnitudes = []
        for first, second in zip(grayscale_images[:-1], grayscale_images[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_magnitudes.append(np.linalg.norm(flow, axis=2))
        magnitudes = np.concatenate([item.reshape(-1) for item in flow_magnitudes])
        force_features = []
        for frame in self.frames:
            force = ground_truth["forces"][frame].reshape(-1, 3)
            # Geometry-independent scalar load descriptor available at inference
            # from the force sensor/front-end; no stiffness label is used here.
            force_features.append(torch.log1p(force.norm(dim=1).sum()).float())
        return {
            "images": torch.stack(images),
            "force_features": torch.stack(force_features).unsqueeze(1),
            # Measured from rendered frames, not from FEM displacement labels.
            # These descriptors encode the observed response to the known load.
            "motion_features": torch.tensor(
                [np.log1p(magnitudes.mean()), np.log1p(magnitudes.std())],
                dtype=torch.float32,
            ),
            "hypothesis": torch.tensor(
                1 if row["stiffness_mode"] == "inclusion" else 0, dtype=torch.long
            ),
            "log_E_background": torch.tensor(math.log(row["E_bg"]), dtype=torch.float32),
            "log_ratio": torch.tensor(math.log(row["E_inc"] / row["E_bg"]), dtype=torch.float32),
            "radius": torch.tensor(row["inclusion_radius"], dtype=torch.float32),
            "scene_id": row["id"],
            "stiffness_mode": row["stiffness_mode"],
        }


def batch_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | int | str],
) -> tuple[torch.Tensor, dict[str, float]]:
    hypothesis = batch["hypothesis"]
    hypothesis_loss = F.cross_entropy(output["hypothesis_logits"], hypothesis)
    background_loss = F.smooth_l1_loss(
        output["log_E_background"].squeeze(1), batch["log_E_background"]
    )
    inclusion = hypothesis == 1
    if inclusion.any():
        ratio_loss = F.smooth_l1_loss(
            output["log_inclusion_ratio"].squeeze(1)[inclusion],
            batch["log_ratio"][inclusion],
        )
        radius_loss = F.smooth_l1_loss(
            output["inclusion_radius"].squeeze(1)[inclusion],
            batch["radius"][inclusion],
        )
    else:
        ratio_loss = background_loss.new_zeros(())
        radius_loss = background_loss.new_zeros(())
    loss = hypothesis_loss + background_loss + ratio_loss + radius_loss
    return loss, {
        "loss": float(loss.detach()),
        "hypothesis": float(hypothesis_loss.detach()),
        "background": float(background_loss.detach()),
        "ratio": float(ratio_loss.detach()),
        "radius": float(radius_loss.detach()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    correct, count = 0, 0
    background_error, ratio_error, radius_error = [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        output = model(
            images,
            batch["force_features"].to(device),
            batch["motion_features"].to(device),
        )
        prediction = physics_ready_predictions(output)
        correct += int(
            (prediction["hypothesis_probability"].argmax(dim=1).cpu() == batch["hypothesis"]).sum()
        )
        count += len(images)
        predicted_bg = prediction["E_background"].squeeze(1).cpu()
        target_bg = batch["log_E_background"].exp()
        background_error.extend((predicted_bg - target_bg).abs().div(target_bg).tolist())
        inclusion = batch["hypothesis"] == 1
        if inclusion.any():
            predicted_ratio = (
                prediction["E_inclusion"].squeeze(1).cpu()
                / prediction["E_background"].squeeze(1).cpu()
            )
            target_ratio = batch["log_ratio"].exp()
            ratio_error.extend(
                (predicted_ratio[inclusion] - target_ratio[inclusion])
                .abs()
                .div(target_ratio[inclusion])
                .tolist()
            )
            radius_error.extend(
                (prediction["inclusion_radius"].squeeze(1).cpu()[inclusion] - batch["radius"][inclusion])
                .abs()
                .div(batch["radius"][inclusion])
                .tolist()
            )
    return {
        "hypothesis_accuracy": correct / max(count, 1),
        "background_E_median_relative_error": float(np.median(background_error)),
        "inclusion_ratio_median_relative_error": float(np.median(ratio_error))
        if ratio_error
        else float("nan"),
        "radius_median_relative_error": float(np.median(radius_error))
        if radius_error
        else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(
        SimV3PriorDataset("train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation = DataLoader(
        SimV3PriorDataset("val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test = DataLoader(
        SimV3PriorDataset("test"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = HierarchicalMaterialInitializer(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    RESULTS.mkdir(parents=True, exist_ok=True)
    checkpoint = RESULTS / "best.pt"
    best_score = -float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            images = batch["images"].to(device)
            targets = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    images, targets["force_features"], targets["motion_features"]
                )
                loss, _ = batch_loss(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_metrics = evaluate(model, validation, device)
        # This module initializes FEM material parameters; a wrong E scale is
        # more damaging than a hypothesis tie-break, so select it first.
        score = -validation_metrics["background_E_median_relative_error"] + 0.05 * (
            validation_metrics["hypothesis_accuracy"]
        )
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **validation_metrics})
        print(
            f"epoch {epoch:03d}: loss={np.mean(losses):.4f}, "
            f"val hypothesis={validation_metrics['hypothesis_accuracy']:.3f}, "
            f"val E={validation_metrics['background_E_median_relative_error']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    result = {
        "dataset": "sim_v3 only; no clinical ION frames used",
        "selected_epoch": state["epoch"],
        "validation": evaluate(model, validation, device),
        "test": evaluate(model, test, device),
        "history": history,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
