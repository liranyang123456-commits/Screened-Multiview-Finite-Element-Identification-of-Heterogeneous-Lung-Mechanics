#!/usr/bin/env python3
"""Ratio-focused fine-tune on the frozen CT MeshGNN, selected on validation only."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_lung_mesh_gnn import (  # noqa: E402
    build_model,
    evaluate,
    make_loaders,
    material_loss,
    move_graph,
    random_yaw_augmentation,
)


def ratio_selection_score(metrics: dict, e_guard: float) -> float:
    e_penalty = max(0.0, float(metrics["E_background_median_relative_error"]) - e_guard)
    return float(metrics["inclusion_ratio_median_relative_error"]) + 0.25 * e_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset" / "ion_ct_synthetic_mechanics540",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=ROOT / "results" / "ion_ct_expanded_training" / "seed_2028" / "best_gnn.pt",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "ion_ct_ratio_optimization" / "finetune_2028",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ratio-loss-weight", type=float, default=6.0)
    parser.add_argument("--ratio-size-upweight", type=float, default=1.0)
    parser.add_argument("--ratio-e-guard", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    train_loader, val_loader, test_loader, input_dim = make_loaders(
        args.dataset,
        args.batch_size,
        0,
        config.get("experiments_limit"),
        config.get("observation", "image_tracks"),
        float(config.get("track_noise_px", 0.0)),
        bool(config.get("single_view", False)),
        bool(config.get("no_depth", False)),
        bool(config.get("no_confidence", False)),
        bool(config.get("peak_only", False)),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            graphs = [random_yaw_augmentation(move_graph(graph, device)) for graph in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(graphs)
                loss, _ = material_loss(
                    output,
                    graphs,
                    ratio_loss_weight=args.ratio_loss_weight,
                    sdf_weight=float(config.get("sdf_weight", 1.0)),
                    partition_weight=float(config.get("partition_weight", 1.5)),
                    ratio_size_upweight=args.ratio_size_upweight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
                "selection_score": score,
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
                    "config": {
                        **config,
                        "ratio_loss_weight": args.ratio_loss_weight,
                        "ratio_size_upweight": args.ratio_size_upweight,
                        "selection_mode": "ratio",
                        "init_checkpoint": str(args.init_checkpoint),
                    },
                },
                checkpoint,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    validation, val_records = evaluate(model, val_loader, device)
    test, test_records = evaluate(model, test_loader, device)
    result = {
        "selection": "validation ratio median plus E guard; test unused for selection",
        "selected_epoch": state["epoch"],
        "init_checkpoint": str(args.init_checkpoint),
        "validation": validation,
        "test": test,
        "history": history,
    }
    (args.results / "metrics_gnn.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for split, metrics, records in (
        ("val", validation, val_records),
        ("test", test, test_records),
    ):
        (args.results / f"predictions_gnn_{split}.json").write_text(
            json.dumps({"split": split, "metrics": metrics, "records": records}, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(
        {
            "selected_epoch": state["epoch"],
            "val_E": validation["E_background_median_relative_error"],
            "val_ratio": validation["inclusion_ratio_median_relative_error"],
            "test_E": test["E_background_median_relative_error"],
            "test_ratio": test["inclusion_ratio_median_relative_error"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
