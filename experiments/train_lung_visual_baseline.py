"""Retrain the legacy ResNet18 material initializer as an image baseline."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hierarchical_material_initializer import (  # noqa: E402
    HierarchicalMaterialInitializer,
)

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class LungVisualDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int = 64) -> None:
        self.root = root
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.rows = [row for row in manifest["patients"] if row["split"] == split]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        images, force_features, flows = [], [], []
        for experiment in row["experiments"]:
            gt_path = self.root / experiment["relative_path"]
            gt = torch.load(gt_path, map_location="cpu", weights_only=False)
            image_path = gt_path.parent / "images" / "frame_02.png"
            if not image_path.exists():
                raise FileNotFoundError(
                    f"{image_path}; visual baseline requires data generated with images"
                )
            image = Image.open(image_path).convert("RGB").resize(
                (self.image_size, self.image_size)
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
            images.append(
                (torch.from_numpy(array).permute(2, 0, 1) - MEAN) / STD
            )
            force = gt["forces"][2].view(-1, 3).sum(dim=0).norm()
            force_features.append(torch.log1p(force).float())
            flow = (
                gt["image_uv_deformed_seq"][2] - gt["image_uv_rest_seq"][2]
            )
            visible = gt["image_visibility_seq"][2]
            magnitude = flow[visible].norm(dim=1)
            flows.append(magnitude)
        magnitude = torch.cat(flows)
        return {
            "patient_id": row["patient_id"],
            "images": torch.stack(images),
            "force_features": torch.stack(force_features)[:, None],
            "motion_features": torch.stack(
                (torch.log1p(magnitude.mean()), torch.log1p(magnitude.std()))
            ).float(),
            "heterogeneous": torch.tensor(
                float(row["inclusion_ratio"] > 1.05), dtype=torch.float32
            ),
            "log_E_background": torch.tensor(
                math.log(row["E_background"]), dtype=torch.float32
            ),
            "log_ratio": torch.tensor(
                math.log(row["inclusion_ratio"]), dtype=torch.float32
            ),
            "center_xy": torch.tensor(
                row["inclusion_center_fraction"][:2], dtype=torch.float32
            ),
            "radius": torch.tensor(
                row["inclusion_radius_fraction"], dtype=torch.float32
            ),
        }


def visual_loss(output: dict, batch: dict) -> torch.Tensor:
    heterogeneous = batch["heterogeneous"] > 0.5
    hypothesis_target = heterogeneous.to(torch.long)
    loss = F.cross_entropy(output["hypothesis_logits"], hypothesis_target)
    loss = loss + F.smooth_l1_loss(
        output["log_E_background"].squeeze(1), batch["log_E_background"]
    )
    predicted_log_ratio = output["log_inclusion_ratio"].squeeze(1).clamp(
        min=0.0, max=math.log(5.0)
    )
    loss = loss + F.smooth_l1_loss(predicted_log_ratio, batch["log_ratio"])
    if heterogeneous.any():
        loss = loss + F.smooth_l1_loss(
            output["inclusion_center_xy"][heterogeneous],
            batch["center_xy"][heterogeneous],
        )
        loss = loss + F.smooth_l1_loss(
            output["inclusion_radius"].squeeze(1)[heterogeneous],
            batch["radius"][heterogeneous],
        )
    return loss


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    records = []
    for batch in loader:
        images = batch["images"].to(device)
        output = model(
            images,
            batch["force_features"].to(device),
            batch["motion_features"].to(device),
        )
        E = output["log_E_background"].exp().squeeze(1).cpu()
        ratio = (
            output["log_inclusion_ratio"]
            .squeeze(1)
            .clamp(0.0, math.log(5.0))
            .exp()
            .cpu()
        )
        for index, patient_id in enumerate(batch["patient_id"]):
            target_E = float(batch["log_E_background"][index].exp())
            target_ratio = float(batch["log_ratio"][index].exp())
            records.append(
                {
                    "patient_id": patient_id,
                    "E_background_true": target_E,
                    "E_background_estimated": float(E[index]),
                    "E_background_relative_error": abs(float(E[index]) - target_E)
                    / target_E,
                    "inclusion_ratio_true": target_ratio,
                    "inclusion_ratio_estimated": float(ratio[index]),
                    "inclusion_ratio_relative_error": abs(
                        float(ratio[index]) - target_ratio
                    )
                    / target_ratio,
                }
            )
    return {
        "patient_count": len(records),
        "E_background_median_relative_error": float(
            np.median([row["E_background_relative_error"] for row in records])
        ),
        "inclusion_ratio_median_relative_error": float(
            np.median([row["inclusion_ratio_relative_error"] for row in records])
        ),
    }, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: LungVisualDataset(args.dataset, split)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
        )
        for split, dataset in datasets.items()
    }
    model = HierarchicalMaterialInitializer(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.results.mkdir(parents=True, exist_ok=True)
    checkpoint = args.results / "best_visual_resnet.pt"
    best, stale, history = float("inf"), 0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["images"].to(device),
                batch["force_features"].to(device),
                batch["motion_features"].to(device),
            )
            targets = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss = visual_loss(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _ = evaluate(model, loaders["val"], device)
        score = (
            validation["E_background_median_relative_error"]
            + validation["inclusion_ratio_median_relative_error"]
        )
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **validation})
        if score < best:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    validation, _ = evaluate(model, loaders["val"], device)
    test, records = evaluate(model, loaders["test"], device)
    result = {
        "model": "legacy_visual_resnet18",
        "selected_epoch": state["epoch"],
        "validation": validation,
        "test": test,
        "history": history,
        "records": records,
    }
    (args.results / "metrics_visual_resnet.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"validation": validation, "test": test}, indent=2))


if __name__ == "__main__":
    main()
