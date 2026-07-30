"""Train patient-level mesh material predictors on sim_lung_ai data."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import (  # noqa: E402
    SimLungGraphDataset,
    collate_lung_graphs,
)
from evaluation.material_uncertainty_metrics import (  # noqa: E402
    binary_calibration,
    bootstrap_median_ci,
    gaussian_interval_metrics,
    gaussian_std_calibration_scale,
)
from models.lung_mesh_material_gnn import (  # noqa: E402
    GlobalFeatureMLP,
    MeshMaterialGNN,
)


def move_graph(graph: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            {name: value.to(device) for name, value in item.items()}
            if key in ("labels", "y")
            else item.to(device)
            if torch.is_tensor(item)
            else item
        )
        for key, item in graph.items()
    }


def random_yaw_augmentation(graph: dict[str, Any]) -> dict[str, Any]:
    # Pixel flow is camera-frame 2D data. Applying a 3D yaw to it (or to the
    # mesh without also re-rendering every camera) is physically inconsistent.
    if graph.get("dynamic_seq") is not None or graph.get("observation_mode") == "image_tracks":
        return graph
    labels = {name: value.clone() for name, value in graph["labels"].items()}
    graph = {
        **graph,
        "x": graph["x"].clone(),
        "static_x": graph["static_x"].clone(),
        "pos": graph["pos"].clone(),
        "labels": labels,
        "y": labels,
    }
    angle = 2.0 * math.pi * torch.rand(
        (), device=graph["x"].device, dtype=graph["x"].dtype
    )
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine, cosine.new_zeros(()))),
            torch.stack((sine, cosine, cosine.new_zeros(()))),
            cosine.new_tensor([0.0, 0.0, 1.0]),
        )
    )
    coordinates = graph["x"][:, :3] - 0.5
    graph["x"][:, :3] = coordinates @ rotation.T + 0.5
    graph["static_x"][:, :3] = graph["x"][:, :3]
    graph["pos"] = graph["x"][:, :3]
    for start in range(5, graph["x"].shape[1], 6):
        graph["x"][:, start : start + 3] = (
            graph["x"][:, start : start + 3] @ rotation.T
        )
        graph["x"][:, start + 3 : start + 6] = (
            graph["x"][:, start + 3 : start + 6] @ rotation.T
        )
    center = graph["labels"]["center_fraction"] - 0.5
    graph["labels"]["center_fraction"] = center @ rotation.T + 0.5
    return graph


def heteroscedastic_nll(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (torch.exp(-logvar) * (mean - target).square() + logvar)


def material_loss(
    output: dict[str, torch.Tensor | list[torch.Tensor]],
    graphs: list[dict[str, Any]],
    ratio_loss_weight: float = 1.0,
    heterogeneity_pos_weight: float = 1.0,
    sdf_weight: float = 0.75,
    partition_weight: float = 1.0,
    eikonal_weight: float = 0.05,
    smoothness_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = [graph["labels"] for graph in graphs]
    heterogeneous = torch.stack([item["heterogeneous"] for item in labels]).view(-1, 1)
    terms: dict[str, torch.Tensor] = {}
    for name in ("log_E_background", "log_ratio"):
        target = torch.stack([item[name] for item in labels]).view(-1, 1)
        terms[name] = heteroscedastic_nll(
            output[f"{name}_mean"], output[f"{name}_logvar"], target  # type: ignore[arg-type]
        ).mean()
        terms[f"{name}_mean"] = F.smooth_l1_loss(
            output[f"{name}_mean"], target  # type: ignore[arg-type]
        )
    region_mask = heterogeneous[:, 0] > 0.5
    if region_mask.any():
        center_target = torch.stack(
            [item["center_fraction"] for item in labels]
        )[region_mask]
        radius_target = torch.stack(
            [item["radius_fraction"] for item in labels]
        ).view(-1, 1)[region_mask]
        terms["center"] = heteroscedastic_nll(
            output["center_fraction_mean"][region_mask],  # type: ignore[index]
            output["center_fraction_logvar"][region_mask],  # type: ignore[index]
            center_target,
        ).mean()
        terms["radius"] = heteroscedastic_nll(
            output["radius_fraction_mean"][region_mask],  # type: ignore[index]
            output["radius_fraction_logvar"][region_mask],  # type: ignore[index]
            radius_target,
        ).mean()
        terms["center_mean"] = F.smooth_l1_loss(
            output["center_fraction_mean"][region_mask], center_target  # type: ignore[index]
        )
        terms["radius_mean"] = F.smooth_l1_loss(
            output["radius_fraction_mean"][region_mask], radius_target  # type: ignore[index]
        )
    else:
        terms["center"] = terms["log_E_background"].new_zeros(())
        terms["radius"] = terms["log_E_background"].new_zeros(())
        terms["center_mean"] = terms["log_E_background"].new_zeros(())
        terms["radius_mean"] = terms["log_E_background"].new_zeros(())
    terms["heterogeneity"] = F.binary_cross_entropy_with_logits(
        output["heterogeneity_logit"],  # type: ignore[arg-type]
        heterogeneous,
        pos_weight=heterogeneous.new_tensor(heterogeneity_pos_weight),
    )
    node_predictions = output["node_log_E"]
    assert isinstance(node_predictions, list)
    node_logvars = output["node_log_E_logvar"]
    assert isinstance(node_logvars, list)
    terms["node"] = torch.stack(
        [
            heteroscedastic_nll(
                prediction,
                logvar,
                label["node_log_E"],
            ).mean()
            for prediction, logvar, label in zip(
                node_predictions, node_logvars, labels
            )
        ]
    ).mean()
    sdf_predictions = output.get("node_sdf_mean")
    sdf_logvars = output.get("node_sdf_logvar")
    partition_logits = output.get("partition_logits")
    assert isinstance(sdf_predictions, list)
    assert isinstance(sdf_logvars, list)
    assert isinstance(partition_logits, list)
    geometry_rows = []
    partition_rows = []
    eikonal_rows = []
    smoothness_rows = []
    node_fields = output["node_log_E"]
    assert isinstance(node_fields, list)
    for prediction, logvar, logits, node_field, label, graph in zip(
        sdf_predictions,
        sdf_logvars,
        partition_logits,
        node_fields,
        labels,
        graphs,
    ):
        source, target = graph["edge_index"].to(torch.long)
        smoothness_rows.append(
            ((node_field[source] - node_field[target]).square()).mean()
        )
        if float(label["geometry_mask"]) <= 0.5:
            continue
        geometry_rows.append(
            heteroscedastic_nll(prediction, logvar, label["node_sdf"]).mean()
        )
        target_partition = label["partition"]
        positives = target_partition.sum()
        negatives = target_partition.numel() - positives
        positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
        bce = F.binary_cross_entropy_with_logits(
            logits, target_partition, pos_weight=positive_weight
        )
        probability = logits.sigmoid()
        dice = 1.0 - (
            2.0 * (probability * target_partition).sum() + 1.0
        ) / (probability.sum() + target_partition.sum() + 1.0)
        partition_rows.append(bce + dice)
        edge_length = torch.linalg.vector_norm(
            graph["pos"][source] - graph["pos"][target], dim=1
        ).clamp_min(1e-4)
        sdf_gradient = (prediction[source] - prediction[target]).abs() / edge_length
        eikonal_rows.append((sdf_gradient - 1.0).abs().mean())
    zero = terms["node"].new_zeros(())
    terms["sdf"] = torch.stack(geometry_rows).mean() if geometry_rows else zero
    terms["partition"] = (
        torch.stack(partition_rows).mean() if partition_rows else zero
    )
    terms["eikonal"] = torch.stack(eikonal_rows).mean() if eikonal_rows else zero
    terms["graph_smoothness"] = (
        torch.stack(smoothness_rows).mean() if smoothness_rows else zero
    )
    total = (
        terms["log_E_background"]
        + 2.0 * terms["log_E_background_mean"]
        + ratio_loss_weight * terms["log_ratio"]
        + 2.0 * ratio_loss_weight * terms["log_ratio_mean"]
        + 0.5 * terms["center"]
        + 3.0 * terms["center_mean"]
        + 0.5 * terms["radius"]
        + 3.0 * terms["radius_mean"]
        + 0.5 * terms["heterogeneity"]
        + 1.00 * terms["node"]
        + sdf_weight * terms["sdf"]
        + partition_weight * terms["partition"]
        + eikonal_weight * terms["eikonal"]
        + smoothness_weight * terms["graph_smoothness"]
    )
    return total, {key: float(value.detach()) for key, value in terms.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    geometry_fusion_weight: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 0.0 <= geometry_fusion_weight <= 1.0:
        raise ValueError("geometry_fusion_weight must lie in [0, 1]")
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        graphs = [move_graph(graph, device) for graph in batch]
        output = model(graphs)
        log_E_mean = output["log_E_background_mean"].squeeze(1).cpu().numpy()
        log_E_std = (
            0.5 * output["log_E_background_logvar"].squeeze(1)  # type: ignore[union-attr]
        ).exp().cpu().numpy()
        log_ratio_mean = output["log_ratio_mean"].squeeze(1).cpu().numpy()
        log_ratio_std = (
            0.5 * output["log_ratio_logvar"].squeeze(1)  # type: ignore[union-attr]
        ).exp().cpu().numpy()
        heterogeneity_probability = (
            output["heterogeneity_logit"].squeeze(1).sigmoid().cpu().numpy()  # type: ignore[union-attr]
        )
        center_mean = output["center_fraction_mean"].cpu().numpy()  # type: ignore[union-attr]
        center_std = (
            0.5 * output["center_fraction_logvar"]  # type: ignore[operator]
        ).exp().cpu().numpy()
        radius_mean = output["radius_fraction_mean"].squeeze(1).cpu().numpy()  # type: ignore[union-attr]
        radius_std = (
            0.5 * output["radius_fraction_logvar"].squeeze(1)  # type: ignore[union-attr]
        ).exp().cpu().numpy()
        node_sdf_means = output.get("node_sdf_mean")
        node_sdf_logvars = output.get("node_sdf_logvar")
        partition_logits = output.get("partition_logits")
        for index, graph in enumerate(graphs):
            label = graph["labels"]
            true_E = float(label["log_E_background"].exp().cpu())
            true_ratio = float(label["log_ratio"].exp().cpu())
            fused_log_E = float(log_E_mean[index])
            fused_log_ratio = float(log_ratio_mean[index])
            fused_center = torch.as_tensor(
                center_mean[index],
                dtype=torch.float32,
                device=device,
            )
            fused_radius = float(radius_mean[index])
            probability = None
            if (
                isinstance(partition_logits, list)
                and isinstance(node_sdf_means, list)
            ):
                probability = partition_logits[index].sigmoid()
                partition_center = (
                    (probability[:, None] * graph["pos"]).sum(dim=0)
                    / probability.sum().clamp_min(1e-6)
                )
                fused_center = (
                    (1.0 - geometry_fusion_weight) * fused_center
                    + geometry_fusion_weight * partition_center
                )
                distance = torch.linalg.vector_norm(
                    graph["pos"] - partition_center,
                    dim=1,
                )
                sdf_radius = torch.median(
                    distance - node_sdf_means[index]
                ).clamp(0.05, 0.40)
                fused_radius = (
                    (1.0 - geometry_fusion_weight) * fused_radius
                    + geometry_fusion_weight * float(sdf_radius)
                )
            estimated_E = math.exp(fused_log_E)
            estimated_ratio = math.exp(fused_log_ratio)
            heterogeneous = bool(float(label["heterogeneous"].cpu()) > 0.5)
            center_error = (
                float(
                    torch.linalg.vector_norm(
                        torch.as_tensor(center_mean[index])
                        - label["center_fraction"].cpu()
                    )
                )
                if heterogeneous
                else None
            )
            if heterogeneous:
                center_error = float(
                    torch.linalg.vector_norm(
                        fused_center - label["center_fraction"]
                    ).cpu()
                )
            radius_error = (
                abs(fused_radius - float(label["radius_fraction"].cpu()))
                / float(label["radius_fraction"].cpu())
                if heterogeneous
                else None
            )
            sdf_mae = None
            sdf_nll = None
            partition_dice = None
            if heterogeneous and isinstance(node_sdf_means, list):
                sdf_mean = node_sdf_means[index]
                sdf_logvar = node_sdf_logvars[index]  # type: ignore[index]
                sdf_target = label["node_sdf"]
                sdf_mae = float((sdf_mean - sdf_target).abs().mean().cpu())
                sdf_nll = float(
                    heteroscedastic_nll(sdf_mean, sdf_logvar, sdf_target)
                    .mean()
                    .cpu()
                )
                if probability is None:
                    probability = partition_logits[index].sigmoid()  # type: ignore[index]
                partition_target = label["partition"]
                partition_dice = float(
                    (
                        (2.0 * (probability * partition_target).sum() + 1.0)
                        / (probability.sum() + partition_target.sum() + 1.0)
                    ).cpu()
                )
            records.append(
                {
                    "patient_id": graph["patient_id"],
                    "E_background_true": true_E,
                    "E_background_estimated": estimated_E,
                    "E_background_relative_error": abs(estimated_E - true_E) / true_E,
                    "inclusion_ratio_true": true_ratio,
                    "inclusion_ratio_estimated": estimated_ratio,
                    "inclusion_ratio_relative_error": abs(estimated_ratio - true_ratio)
                    / true_ratio,
                    "log_E_mean": fused_log_E,
                    "log_E_std": float(log_E_std[index]),
                    "log_ratio_mean": fused_log_ratio,
                    "log_ratio_std": float(log_ratio_std[index]),
                    "heterogeneous_true": heterogeneous,
                    "heterogeneity_probability": float(
                        heterogeneity_probability[index]
                    ),
                    "center_fraction_true": label["center_fraction"].cpu().tolist(),
                    "center_fraction_estimated": fused_center.cpu().tolist(),
                    "center_fraction_std": center_std[index].tolist(),
                    "radius_fraction_true": float(label["radius_fraction"].cpu()),
                    "radius_fraction_estimated": fused_radius,
                    "radius_fraction_std": float(radius_std[index]),
                    "center_error_normalized": center_error,
                    "radius_relative_error": radius_error,
                    "node_sdf_mae": sdf_mae,
                    "node_sdf_nll": sdf_nll,
                    "partition_soft_dice": partition_dice,
                }
            )
    E_errors = [row["E_background_relative_error"] for row in records]
    ratio_errors = [row["inclusion_ratio_relative_error"] for row in records]
    heterogeneous_rows = [row for row in records if row["heterogeneous_true"]]
    metrics = {
        "patient_count": len(records),
        "E_background_median_relative_error": float(np.median(E_errors)),
        "E_background_median_bootstrap_95_ci": bootstrap_median_ci(E_errors),
        "inclusion_ratio_median_relative_error": float(np.median(ratio_errors)),
        "inclusion_ratio_median_bootstrap_95_ci": bootstrap_median_ci(
            ratio_errors, seed=2027
        ),
        "center_error_normalized_median": float(
            np.median(
                [row["center_error_normalized"] for row in heterogeneous_rows]
            )
        )
        if heterogeneous_rows
        else None,
        "radius_relative_error_median": float(
            np.median([row["radius_relative_error"] for row in heterogeneous_rows])
        )
        if heterogeneous_rows
        else None,
        "node_sdf_mae_mean": float(
            np.mean([row["node_sdf_mae"] for row in heterogeneous_rows])
        )
        if heterogeneous_rows
        else None,
        "node_sdf_nll_mean": float(
            np.mean([row["node_sdf_nll"] for row in heterogeneous_rows])
        )
        if heterogeneous_rows
        else None,
        "partition_soft_dice_mean": float(
            np.mean([row["partition_soft_dice"] for row in heterogeneous_rows])
        )
        if heterogeneous_rows
        else None,
        "heterogeneity": binary_calibration(
            np.asarray([row["heterogeneous_true"] for row in records], dtype=float),
            np.asarray(
                [row["heterogeneity_probability"] for row in records], dtype=float
            ),
        ),
        "log_E_uncertainty": gaussian_interval_metrics(
            np.asarray([math.log(row["E_background_true"]) for row in records]),
            np.asarray([row["log_E_mean"] for row in records]),
            np.asarray([row["log_E_std"] for row in records]),
        ),
        "log_ratio_uncertainty": gaussian_interval_metrics(
            np.asarray([math.log(row["inclusion_ratio_true"]) for row in records]),
            np.asarray([row["log_ratio_mean"] for row in records]),
            np.asarray([row["log_ratio_std"] for row in records]),
        ),
    }
    return metrics, records


def make_loaders(
    dataset: Path,
    batch_size: int,
    workers: int,
    experiments_limit: int | None,
    observation: str = "noisy",
    track_noise_px: float = 0.0,
    single_view: bool = False,
    no_depth: bool = False,
    no_confidence: bool = False,
    peak_only: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    datasets = {
        split: SimLungGraphDataset(
            dataset,
            split=split,
            observation_key=observation,
            experiments_limit=experiments_limit,
            track_noise_px=track_noise_px,
            single_view=single_view,
            no_depth=no_depth,
            no_confidence=no_confidence,
            peak_only=peak_only,
            cache_graphs=True,
        )
        for split in ("train", "val", "test")
    }
    if any(len(value) == 0 for value in datasets.values()):
        raise ValueError("Train, val, and test splits must all be non-empty")
    input_dim = int(datasets["train"][0]["x"].shape[1])
    loaders = {
        split: DataLoader(
            value,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_lung_graphs,
        )
        for split, value in datasets.items()
    }
    return loaders["train"], loaders["val"], loaders["test"], input_dim


def build_model(
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    dynamic_dim: int | None = None,
) -> nn.Module:
    model_class = MeshMaterialGNN if model_name == "gnn" else GlobalFeatureMLP
    options = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=layers,
        dropout=dropout,
    )
    if model_class is MeshMaterialGNN:
        options["dynamic_dim"] = (
            dynamic_dim if dynamic_dim is not None else (14 if input_dim == 5 else None)
        )
    return model_class(**options)


def validation_uncertainty_scales(
    validation_records: list[dict[str, Any]],
) -> dict[str, float]:
    scales = {}
    for name, true_key, mean_key, std_key in (
        ("log_E", "E_background_true", "log_E_mean", "log_E_std"),
        ("log_ratio", "inclusion_ratio_true", "log_ratio_mean", "log_ratio_std"),
    ):
        scales[name] = gaussian_std_calibration_scale(
            np.log([row[true_key] for row in validation_records]),
            np.asarray([row[mean_key] for row in validation_records]),
            np.asarray([row[std_key] for row in validation_records]),
        )
    return scales


def calibrate_test_uncertainty(
    test_records: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    scales: dict[str, float],
) -> None:
    for name, true_key, mean_key, std_key in (
        ("log_E", "E_background_true", "log_E_mean", "log_E_std"),
        ("log_ratio", "inclusion_ratio_true", "log_ratio_mean", "log_ratio_std"),
    ):
        for row in test_records:
            row[std_key] *= scales[name]
        test_metrics[f"{name}_uncertainty"] = gaussian_interval_metrics(
            np.log([row[true_key] for row in test_records]),
            np.asarray([row[mean_key] for row in test_records]),
            np.asarray([row[std_key] for row in test_records]),
        )


def validation_selection_score(metrics: dict[str, Any]) -> float:
    """Select checkpoints using global and node-level material recovery."""
    score = (
        float(metrics["E_background_median_relative_error"])
        + float(metrics["inclusion_ratio_median_relative_error"])
        + 0.25 * float(metrics["heterogeneity"]["brier_score"])
    )
    for key, weight in (
        ("center_error_normalized_median", 0.5),
        ("radius_relative_error_median", 0.5),
    ):
        value = metrics.get(key)
        if value is not None:
            score += weight * float(value)
    partition_dice = metrics.get("partition_soft_dice_mean")
    if partition_dice is not None:
        score += 0.5 * (1.0 - float(partition_dice))
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", choices=("gnn", "mlp"), default="gnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--experiments-limit", type=int)
    parser.add_argument(
        "--observation", choices=("oracle", "noisy", "image_tracks"), default="noisy"
    )
    parser.add_argument("--track-noise-px", type=float, default=0.0)
    parser.add_argument("--single-view", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-confidence", action="store_true")
    parser.add_argument("--peak-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-rotation-augmentation", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--ratio-loss-weight", type=float, default=2.0)
    parser.add_argument("--heterogeneity-pos-weight", type=float, default=1.0)
    parser.add_argument("--sdf-weight", type=float, default=1.0)
    parser.add_argument("--partition-weight", type=float, default=1.5)
    parser.add_argument("--eikonal-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, input_dim = make_loaders(
        args.dataset,
        args.batch_size,
        args.workers,
        args.experiments_limit,
        args.observation,
        args.track_noise_px,
        args.single_view,
        args.no_depth,
        args.no_confidence,
        args.peak_only,
    )
    model = build_model(
        args.model, input_dim, args.hidden_dim, args.layers, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.results.mkdir(parents=True, exist_ok=True)
    checkpoint = args.results / f"best_{args.model}.pt"
    last_checkpoint = args.results / f"last_{args.model}.pt"
    best_score, stale, start_epoch = float("inf"), 0, 0
    history = []
    if args.resume:
        resume_state = torch.load(
            args.resume, map_location=device, weights_only=True
        )
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer"])
            except ValueError:
                # Legacy checkpoints have no temporal/SDF-head optimizer slots.
                pass
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        best_score = float(resume_state.get("best_score", best_score))
        stale = int(resume_state.get("stale", 0))
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state["epoch"]) + 1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            graphs = [move_graph(graph, device) for graph in batch]
            if not args.no_rotation_augmentation:
                graphs = [random_yaw_augmentation(graph) for graph in graphs]
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
                    heterogeneity_pos_weight=args.heterogeneity_pos_weight,
                    sdf_weight=args.sdf_weight,
                    partition_weight=args.partition_weight,
                    eikonal_weight=args.eikonal_weight,
                    smoothness_weight=args.smoothness_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation, _ = evaluate(model, val_loader, device)
        score = validation_selection_score(validation)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_score": score,
                **validation,
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
                        "model": args.model,
                        "input_dim": input_dim,
                        "hidden_dim": args.hidden_dim,
                        "layers": args.layers,
                        "dropout": args.dropout,
                        "experiments_limit": args.experiments_limit,
                        "observation": args.observation,
                        "track_noise_px": args.track_noise_px,
                        "ratio_loss_weight": args.ratio_loss_weight,
                        "heterogeneity_pos_weight": args.heterogeneity_pos_weight,
                        "sdf_weight": args.sdf_weight,
                        "partition_weight": args.partition_weight,
                        "eikonal_weight": args.eikonal_weight,
                        "smoothness_weight": args.smoothness_weight,
                        "dynamic_dim": 14 if input_dim == 5 else None,
                        "single_view": args.single_view,
                        "no_depth": args.no_depth,
                        "no_confidence": args.no_confidence,
                        "peak_only": args.peak_only,
                    },
                },
                checkpoint,
            )
        else:
            stale += 1
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "stale": stale,
                "history": history,
                "config": {
                    "model": args.model,
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "dynamic_dim": 14 if input_dim == 5 else None,
                },
            },
            last_checkpoint,
        )
        if stale >= args.patience:
            break
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    validation, validation_records = evaluate(model, val_loader, device)
    test, test_records = (
        (None, [])
        if args.skip_test
        else evaluate(model, test_loader, device)
    )
    uncertainty_scales = validation_uncertainty_scales(validation_records)
    if test is not None:
        calibrate_test_uncertainty(test_records, test, uncertainty_scales)
    state["uncertainty_calibration_scales"] = uncertainty_scales
    torch.save(state, checkpoint)
    result = {
        "model": args.model,
        "dataset": str(args.dataset),
        "selected_epoch": state["epoch"],
        "config": state["config"],
        "validation": validation,
        "test": test,
        "history": history,
        "uncertainty_calibration_scales": uncertainty_scales,
    }
    (args.results / f"metrics_{args.model}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    prediction_sets = [("val", validation, validation_records)]
    if test is not None:
        prediction_sets.append(("test", test, test_records))
    for split, metrics, records in prediction_sets:
        (args.results / f"predictions_{args.model}_{split}.json").write_text(
            json.dumps(
                {"model": args.model, "split": split, "metrics": metrics, "records": records},
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({"validation": validation, "test": test}, indent=2))


if __name__ == "__main__":
    main()
