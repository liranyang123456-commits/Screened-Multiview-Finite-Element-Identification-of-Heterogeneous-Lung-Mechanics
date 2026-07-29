"""Evaluate a frozen lung graph model under controlled input perturbations."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import SimLungGraphDataset, collate_lung_graphs  # noqa: E402
from experiments.train_lung_mesh_gnn import (  # noqa: E402
    build_model,
    evaluate,
)
from evaluation.material_uncertainty_metrics import (  # noqa: E402
    gaussian_interval_metrics,
)


class PerturbedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base: SimLungGraphDataset,
        *,
        force_scale: float,
        motion_noise_std: float,
        seed: int,
        active_loads: int | None,
    ) -> None:
        self.base = base
        self.force_scale = force_scale
        self.motion_noise_std = motion_noise_std
        self.seed = seed
        self.active_loads = active_loads

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        graph = self.base[index]
        if graph.get("dynamic_seq") is not None:
            graph["dynamic_seq"] = graph["dynamic_seq"].clone()
            dynamic = graph["dynamic_seq"]
            generator = torch.Generator().manual_seed(self.seed + index)
            if self.active_loads is not None:
                dynamic[self.active_loads :] = 0.0
            active = dynamic.shape[0] if self.active_loads is None else min(
                self.active_loads, dynamic.shape[0]
            )
            encoded_force = dynamic[:active, ..., 5:8]
            raw_force = torch.sign(encoded_force) * torch.expm1(encoded_force.abs())
            raw_force *= self.force_scale
            dynamic[:active, ..., 5:8] = (
                torch.sign(raw_force) * torch.log1p(raw_force.abs())
            )
            if self.motion_noise_std:
                flow = dynamic[:active, ..., :2]
                visibility = dynamic[:active, ..., 4:5]
                flow += (
                    self.motion_noise_std
                    * torch.randn(flow.shape, generator=generator, dtype=flow.dtype)
                    * visibility
                )
            return graph
        graph["x"] = graph["x"].clone()
        generator = torch.Generator().manual_seed(self.seed + index)
        for load_index, start in enumerate(range(5, graph["x"].shape[1], 6)):
            if self.active_loads is not None and load_index >= self.active_loads:
                graph["x"][:, start : start + 6] = 0.0
                continue
            encoded_force = graph["x"][:, start : start + 3]
            raw_force = torch.sign(encoded_force) * torch.expm1(
                encoded_force.abs()
            )
            raw_force *= self.force_scale
            graph["x"][:, start : start + 3] = torch.sign(
                raw_force
            ) * torch.log1p(raw_force.abs())
            if self.motion_noise_std:
                motion = graph["x"][:, start + 3 : start + 6]
                graph["x"][:, start + 3 : start + 6] = motion + (
                    self.motion_noise_std
                    * torch.randn(
                        motion.shape, generator=generator, dtype=motion.dtype
                    )
                    * graph["x"][:, 4:5]
                )
        return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--force-scale", type=float, default=1.0)
    parser.add_argument("--motion-noise-std", type=float, default=0.0)
    parser.add_argument("--experiments-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--active-loads", type=int)
    parser.add_argument("--single-view", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-confidence", action="store_true")
    parser.add_argument("--peak-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    experiments_limit = (
        args.experiments_limit
        if args.experiments_limit is not None
        else config.get("experiments_limit")
    )
    base = SimLungGraphDataset(
        args.dataset,
        split=args.split,
        observation_key=config.get("observation", "noisy"),
        experiments_limit=experiments_limit,
        track_noise_px=float(config.get("track_noise_px", 0.0)),
        single_view=args.single_view or bool(config.get("single_view", False)),
        no_depth=args.no_depth or bool(config.get("no_depth", False)),
        no_confidence=args.no_confidence or bool(config.get("no_confidence", False)),
        peak_only=args.peak_only or bool(config.get("peak_only", False)),
    )
    dataset = PerturbedDataset(
        base,
        force_scale=args.force_scale,
        motion_noise_std=args.motion_noise_std,
        seed=args.seed,
        active_loads=args.active_loads,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_lung_graphs,
    )
    model = build_model(
        config["model"],
        config["input_dim"],
        config["hidden_dim"],
        config["layers"],
        config["dropout"],
        config.get("dynamic_dim"),
    ).to(device)
    model.load_state_dict(state["model"])
    metrics, records = evaluate(model, loader, device)
    calibration_scales = state.get("uncertainty_calibration_scales", {})
    for name, true_key, mean_key, std_key in (
        ("log_E", "E_background_true", "log_E_mean", "log_E_std"),
        ("log_ratio", "inclusion_ratio_true", "log_ratio_mean", "log_ratio_std"),
    ):
        if name not in calibration_scales:
            continue
        for row in records:
            row[std_key] *= float(calibration_scales[name])
        metrics[f"{name}_uncertainty"] = gaussian_interval_metrics(
            np.asarray([math.log(row[true_key]) for row in records]),
            np.asarray([row[mean_key] for row in records]),
            np.asarray([row[std_key] for row in records]),
        )
    result = {
        "model": config["model"],
        "split": args.split,
        "force_scale": args.force_scale,
        "motion_noise_std": args.motion_noise_std,
        "experiments_limit": experiments_limit,
        "active_loads": args.active_loads,
        "metrics": metrics,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
