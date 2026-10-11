#!/usr/bin/env python3
"""Freeze the MeshGNN backbone and train a residual ratio head on relative error."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.finetune_ct_ratio import ratio_selection_score  # noqa: E402
from experiments.train_lung_mesh_gnn import (  # noqa: E402
    build_model,
    evaluate,
    make_loaders,
    move_graph,
)


class ResidualRatioRefinement(nn.Module):
    def __init__(self, base: nn.Linear, residual_dim: int = 96) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.residual = nn.Sequential(
            nn.LayerNorm(base.in_features),
            nn.Linear(base.in_features, residual_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(residual_dim, base.out_features),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.base(features) + 0.75 * torch.tanh(self.residual(features))


def relative_ratio_loss(
    output: dict[str, torch.Tensor],
    graphs: list[dict],
    size_upweight: float,
) -> torch.Tensor:
    pred = output["log_ratio_mean"].reshape(-1).exp()
    true = torch.stack([graph["labels"]["log_ratio"] for graph in graphs]).reshape(-1).exp()
    radii = torch.stack([graph["labels"]["radius_fraction"] for graph in graphs]).reshape(-1)
    hetero = torch.stack([graph["labels"]["heterogeneous"] for graph in graphs]).reshape(-1)
    size_boost = (0.15 / radii.clamp(min=0.05)).clamp(1.0, 3.0)
    weak_boost = (2.2 / true.clamp(min=1.05)).clamp(1.0, 2.5)
    weights = torch.where(
        hetero > 0.5,
        1.0 + size_upweight * (size_boost - 1.0) + 0.75 * (weak_boost - 1.0),
        0.35 * torch.ones_like(size_boost),
    )
    weights = weights / weights.mean().clamp_min(1e-6)
    relative = F.smooth_l1_loss(pred / true.clamp_min(1e-6), torch.ones_like(pred), reduction="none")
    log_term = F.smooth_l1_loss(pred.log(), true.clamp_min(1e-6).log(), reduction="none")
    return (weights * (relative + 0.15 * log_term)).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=ROOT / "results" / "ion_ct_expanded_training" / "seed_2028" / "best_gnn.pt",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "ion_ct_ratio_optimization" / "relerr_residual_2028",
    )
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--ratio-size-upweight", type=float, default=1.0)
    parser.add_argument("--ratio-e-guard", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2035)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    train_loader, val_loader, test_loader, input_dim = make_loaders(
        ROOT / "dataset" / "ion_ct_synthetic_mechanics540",
        8,
        0,
        config.get("experiments_limit"),
        config.get("observation", "image_tracks"),
        float(config.get("track_noise_px", 0.0)),
        False,
        False,
        False,
        False,
    )
    model = build_model(
        config["model"],
        input_dim,
        config["hidden_dim"],
        config["layers"],
        config["dropout"],
        config.get("dynamic_dim"),
    ).to(device)
    model.load_state_dict(state["model"])
    model.ratio_refinement = ResidualRatioRefinement(model.ratio_refinement).to(device)
    backbone = []
    heads = []
    for name, param in model.named_parameters():
        trainable = (
            "ratio_refinement.residual" in name
            or name.startswith("layers.4")
            or name.startswith("pool_projection")
        )
        param.requires_grad = trainable
        if not trainable:
            continue
        if "ratio_refinement" in name:
            heads.append(param)
        else:
            backbone.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": heads, "lr": args.lr},
            {"params": backbone, "lr": args.backbone_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.results.mkdir(parents=True, exist_ok=True)
    checkpoint = args.results / "best_gnn.pt"
    best_score, stale = float("inf"), 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            graphs = [move_graph(graph, device) for graph in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(graphs)
                loss = relative_ratio_loss(output, graphs, args.ratio_size_upweight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                5.0,
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation, _ = evaluate(model, val_loader, device)
        score = ratio_selection_score(validation, args.ratio_e_guard)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_E": validation["E_background_median_relative_error"],
                "val_ratio": validation["inclusion_ratio_median_relative_error"],
            }
        )
        print(
            f"epoch {epoch:03d} loss={np.mean(losses):.4f} "
            f"val_E={validation['E_background_median_relative_error']:.3f} "
            f"val_ratio={validation['inclusion_ratio_median_relative_error']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score, stale = score, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "config": {**config, "ratio_head": "residual_relerr"},
                },
                checkpoint,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    # Load with the wrapped module still in place.
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    validation, val_records = evaluate(model, val_loader, device)
    test, test_records = evaluate(model, test_loader, device)
    result = {
        "selection": "validation ratio median plus E guard; test unused for selection",
        "selected_epoch": state["epoch"],
        "validation": validation,
        "test": test,
        "history": history,
    }
    (args.results / "metrics_gnn.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    for split, metrics, records in (
        ("val", validation, val_records),
        ("test", test, test_records),
    ):
        (args.results / f"predictions_gnn_{split}.json").write_text(
            json.dumps({"split": split, "metrics": metrics, "records": records}, indent=2),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "selected_epoch": state["epoch"],
                "val_ratio": validation["inclusion_ratio_median_relative_error"],
                "val_E": validation["E_background_median_relative_error"],
                "test_ratio": test["inclusion_ratio_median_relative_error"],
                "test_E": test["E_background_median_relative_error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
