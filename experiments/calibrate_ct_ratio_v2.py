#!/usr/bin/env python3
"""Train-fit, validation-select CT ratio calibrators and stacking heads.

Fitting never uses validation or test labels except as a frozen selection
score after the train-only parameters are locked. Test is scored once after
the validation winner is chosen.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import SimLungGraphDataset, collate_lung_graphs  # noqa: E402
from experiments.train_lung_mesh_gnn import (  # noqa: E402
    build_model,
    evaluate,
    move_graph,
)

DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
CKPT = ROOT / "results" / "ion_ct_expanded_training" / "seed_2028" / "best_gnn.pt"
VAL_JSON = ROOT / "results" / "ion_ct_expanded_training" / "seed_2028" / "predictions_gnn_val.json"
TEST_JSON = ROOT / "results" / "ion_ct_expanded_training" / "seed_2028" / "predictions_gnn_test.json"
OUT = ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2"
GATE = 0.25


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def row_key(row: dict[str, Any]) -> str:
    if "geometry_id" in row and row.get("scenario_id"):
        return f"{row['geometry_id']}|{row['scenario_id']}"
    patient = str(row.get("patient_id", row.get("entry_id", ""))).replace("\\", "/")
    if "/" in patient:
        geom, scen = patient.split("/", 1)
        return f"{geom}|{scen}"
    if "_scenario_" in patient:
        geom, scen = patient.split("_scenario_", 1)
        return f"{geom}|scenario_{scen}"
    return patient


def geometry_id(row: dict[str, Any]) -> str:
    if "geometry_id" in row:
        return str(row["geometry_id"])
    patient = str(row.get("patient_id", ""))
    return patient.split("_scenario_", 1)[0] if "_scenario_" in patient else patient


def ratio_error(true: float, pred: float) -> float:
    return abs(pred / max(true, 1e-8) - 1.0)


def clip_ratio(value: float) -> float:
    return float(np.clip(value, 1.0, 5.0))


def apply_pred(rows: list[dict], pred: dict[str, float]) -> list[dict]:
    out = []
    for row in rows:
        new = dict(row)
        value = clip_ratio(float(pred[row_key(row)]))
        new["inclusion_ratio_estimated"] = value
        new["inclusion_ratio_relative_error"] = ratio_error(
            float(row["inclusion_ratio_true"]), value
        )
        out.append(new)
    return out


def summarize(rows: list[dict]) -> dict[str, Any]:
    errors = [float(r["inclusion_ratio_relative_error"]) for r in rows]
    by_geom: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_geom[geometry_id(row)].append(float(row["inclusion_ratio_relative_error"]))
    geom_medians = {geom: float(np.median(vals)) for geom, vals in sorted(by_geom.items())}
    return {
        "n": len(errors),
        "median": float(np.median(errors)),
        "pass_rate": float(np.mean(np.asarray(errors) <= GATE)),
        "geometry_medians": geom_medians,
        "geometries_passing": int(sum(m <= GATE for m in geom_medians.values())),
        "geometry_count": len(geom_medians),
    }


def selection_key(summary: dict[str, Any]) -> tuple:
    return (
        float(summary["median"]),
        -int(summary["geometries_passing"]),
        -float(summary["pass_rate"]),
    )


def load_gnn_split(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def make_split_loader(split: str, config: dict[str, Any]) -> tuple[DataLoader, SimLungGraphDataset]:
    dataset = SimLungGraphDataset(
        DATASET,
        split=split,
        observation_key=config.get("observation", "image_tracks"),
        experiments_limit=config.get("experiments_limit"),
        track_noise_px=float(config.get("track_noise_px", 0.0)),
        single_view=bool(config.get("single_view", False)),
        no_depth=bool(config.get("no_depth", False)),
        no_confidence=bool(config.get("no_confidence", False)),
        peak_only=bool(config.get("peak_only", False)),
        cache_graphs=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_lung_graphs,
    )
    return loader, dataset


def flow_contrast(graph: dict[str, Any], center: np.ndarray, radius: float) -> float:
    pos = graph["pos"]
    center_t = torch.as_tensor(center, dtype=pos.dtype, device=pos.device)
    dist = torch.linalg.vector_norm(pos - center_t, dim=-1)
    inside = dist <= max(float(radius), 0.05)
    seq = graph.get("dynamic_seq")
    if seq is None or not bool(inside.any()) or not bool((~inside).any()):
        return 1.0
    flow = seq[..., :2]
    conf = seq[..., 3] if seq.shape[-1] > 3 else torch.ones(seq.shape[:-1], dtype=seq.dtype)
    mag = torch.linalg.vector_norm(flow, dim=-1) * conf
    while mag.ndim > 1:
        mag = mag.mean(dim=0)
    if mag.numel() != pos.shape[0]:
        return 1.0
    inner = mag[inside].mean().clamp_min(1e-8)
    outer = mag[~inside].mean().clamp_min(1e-8)
    return float((inner / outer).cpu())


def compact_flow_stats(graph: dict[str, Any]) -> tuple[float, float]:
    seq = graph.get("dynamic_seq")
    if seq is None:
        return 1.0, 1.0
    mag = torch.linalg.vector_norm(seq[..., :2], dim=-1)
    return float(mag.mean().cpu()), float(mag.amax().cpu())


def attach_aux(records: list[dict], dataset: SimLungGraphDataset) -> None:
    by_id = {str(row["patient_id"]): row for row in records}
    for graph in dataset:
        row = by_id.get(str(graph["patient_id"]))
        if row is None:
            continue
        row["flow_contrast"] = flow_contrast(
            graph,
            np.asarray(row["center_fraction_estimated"], dtype=np.float64),
            float(row["radius_fraction_estimated"]),
        )
        mean_flow, max_flow = compact_flow_stats(graph)
        row["mean_flow"] = mean_flow
        row["max_flow"] = max_flow


def dump_train_gnn(model, device, config: dict[str, Any]) -> list[dict]:
    loader, dataset = make_split_loader("train", config)
    _, records = evaluate(model, loader, device)
    attach_aux(records, dataset)
    return records


def attach_aux_existing(records: list[dict], split: str, config: dict[str, Any]) -> list[dict]:
    _, dataset = make_split_loader(split, config)
    attach_aux(records, dataset)
    return records


def pred_map(rows: list[dict], key: str = "inclusion_ratio_estimated") -> dict[str, float]:
    return {row_key(row): float(row[key]) for row in rows}


def true_map(rows: list[dict]) -> dict[str, float]:
    return {row_key(row): float(row["inclusion_ratio_true"]) for row in rows}


def grid_best(
    train_rows: list[dict],
    transform: Callable[[np.ndarray, float], np.ndarray],
    values: np.ndarray,
) -> tuple[float, float]:
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in train_rows])
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in train_rows])
    best_value, best = float(values[0]), float("inf")
    for value in values:
        calibrated = np.clip(transform(pred, float(value)), 1.0, 5.0)
        med = float(np.median(np.abs(calibrated / np.maximum(true, 1e-8) - 1.0)))
        if med < best:
            best, best_value = med, float(value)
    return best_value, best


def two_param_affine(train_rows: list[dict]) -> tuple[float, float]:
    true = np.log([float(r["inclusion_ratio_true"]) for r in train_rows])
    pred = np.log([max(float(r["inclusion_ratio_estimated"]), 1e-6) for r in train_rows])
    best = (0.0, 1.0)
    best_med = float("inf")
    for intercept in np.linspace(-0.4, 0.4, 17):
        for slope in np.linspace(0.6, 1.3, 15):
            calibrated = np.exp(intercept + slope * pred)
            med = float(np.median(np.abs(np.exp(np.log(calibrated) - true) - 1.0)))
            if med < best_med:
                best_med = med
                best = (float(intercept), float(slope))
    return best


def bin_multipliers(train_rows: list[dict], bins: int = 4) -> tuple[list[float], list[float]]:
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in train_rows])
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in train_rows])
    edges = np.quantile(pred, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = 0.0
    edges[-1] = 10.0
    multipliers = []
    for index in range(bins):
        mask = (pred >= edges[index]) & (pred <= edges[index + 1] if index == bins - 1 else pred < edges[index + 1])
        if not mask.any():
            multipliers.append(1.0)
            continue
        best_m, best = 1.0, float("inf")
        for mult in np.linspace(0.55, 1.25, 29):
            calibrated = np.clip(pred[mask] * mult, 1.0, 5.0)
            med = float(np.median(np.abs(calibrated / np.maximum(true[mask], 1e-8) - 1.0)))
            if med < best:
                best, best_m = med, float(mult)
        multipliers.append(best_m)
    return edges.tolist(), multipliers


def apply_bins(pred: float, edges: list[float], multipliers: list[float]) -> float:
    for index in range(len(multipliers)):
        upper = edges[index + 1]
        if pred <= upper or index == len(multipliers) - 1:
            return clip_ratio(pred * multipliers[index])
    return clip_ratio(pred)


def design_row(row: dict) -> np.ndarray:
    pred = max(float(row["inclusion_ratio_estimated"]), 1e-6)
    radius = max(float(row["radius_fraction_estimated"]), 1e-6)
    contrast = max(float(row.get("flow_contrast", 1.0)), 1e-6)
    hetero = float(row.get("heterogeneity_probability", 0.5))
    energy = max(float(row["E_background_estimated"]), 1.0)
    return np.asarray(
        [
            math.log(pred),
            math.log(radius),
            math.log(contrast),
            hetero,
            math.log(energy),
            math.log(pred) * math.log(radius),
        ],
        dtype=np.float64,
    )


def fit_linear(train_rows: list[dict], l2: float = 1.0) -> np.ndarray:
    X = np.stack([design_row(row) for row in train_rows])
    y = np.log([float(row["inclusion_ratio_true"]) for row in train_rows])
    X = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)])
    eye = np.eye(X.shape[1])
    eye[0, 0] = 0.0
    coef = np.linalg.solve(X.T @ X + l2 * eye, X.T @ y)
    stats = np.stack([design_row(row) for row in train_rows])
    return np.concatenate([coef, stats.mean(0), np.maximum(stats.std(0), 1e-6)])


def apply_linear(row: dict, packed: np.ndarray) -> float:
    coef = packed[:7]
    mean = packed[7:13]
    std = packed[13:19]
    x = np.concatenate([[1.0], (design_row(row) - mean) / std])
    return clip_ratio(float(np.exp(x @ coef)))


def _contrast_shrink(train_rows: list[dict]) -> tuple[float, float]:
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in train_rows])
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in train_rows])
    contrast = np.asarray([float(r.get("flow_contrast", 1.0)) for r in train_rows])
    best = (1.0, 1.0)
    best_med = float("inf")
    for gate in np.linspace(0.7, 1.6, 16):
        for alpha in np.linspace(0.35, 1.05, 15):
            calibrated = np.where(contrast < gate, 1.0 + alpha * (pred - 1.0), pred)
            calibrated = np.clip(calibrated, 1.0, 5.0)
            med = float(np.median(np.abs(calibrated / np.maximum(true, 1e-8) - 1.0)))
            if med < best_med:
                best_med = med
                best = (float(gate), float(alpha))
    return best


def hetero_gate_threshold(train_rows: list[dict]) -> float:
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in train_rows])
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in train_rows])
    prob = np.asarray([float(r["heterogeneity_probability"]) for r in train_rows])
    best_t, best = 0.0, float("inf")
    for threshold in np.linspace(0.05, 0.8, 16):
        calibrated = np.where(prob < threshold, 1.0, pred)
        med = float(np.median(np.abs(calibrated / np.maximum(true, 1e-8) - 1.0)))
        if med < best:
            best, best_t = med, float(threshold)
    return best_t


def blend_weight(train_a: dict[str, float], train_b: dict[str, float], true: dict[str, float]) -> float:
    keys = sorted(set(train_a) & set(train_b) & set(true))
    best_w, best = 1.0, float("inf")
    for weight in np.linspace(0.0, 1.0, 11):
        errors = []
        for key in keys:
            pred = clip_ratio((1.0 - weight) * train_a[key] + weight * train_b[key])
            errors.append(ratio_error(true[key], pred))
        med = float(np.median(errors))
        if med < best:
            best, best_w = med, float(weight)
    return best_w


def compact_design(row: dict) -> np.ndarray:
    return np.asarray(
        [
            math.log(max(float(row.get("mean_flow", 1.0)), 1e-6)),
            math.log(max(float(row.get("max_flow", 1.0)), 1e-6)),
            math.log(max(float(row.get("flow_contrast", 1.0)), 1e-6)),
            math.log(max(float(row["radius_fraction_estimated"]), 1e-6)),
            float(row.get("heterogeneity_probability", 0.5)),
        ],
        dtype=np.float64,
    )


def ridge_ratio_maps(splits: dict[str, list[dict]], l2: float = 1.0) -> dict[str, dict[str, float]]:
    train = splits["train"]
    X_train = np.stack([compact_design(row) for row in train])
    y_train = np.log([float(row["inclusion_ratio_true"]) for row in train])
    mean = X_train.mean(0)
    std = np.maximum(X_train.std(0), 1e-6)
    Z = np.column_stack([np.ones(len(X_train)), (X_train - mean) / std])
    eye = np.eye(Z.shape[1])
    eye[0, 0] = 0.0
    coef = np.linalg.solve(Z.T @ Z + l2 * eye, Z.T @ y_train)
    out: dict[str, dict[str, float]] = {}
    for split, rows in splits.items():
        mapping = {}
        for row in rows:
            x = np.concatenate([[1.0], (compact_design(row) - mean) / std])
            mapping[row_key(row)] = clip_ratio(float(np.exp(x @ coef)))
        out[split] = mapping
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CKPT, map_location=device, weights_only=True)
    config = state["config"]
    model = build_model(
        config["model"],
        config["input_dim"],
        config["hidden_dim"],
        config["layers"],
        config["dropout"],
        config.get("dynamic_dim"),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    print("dumping train GNN predictions", flush=True)
    train_rows = dump_train_gnn(model, device, config)
    print("attaching val/test auxiliary features", flush=True)
    val_rows = attach_aux_existing(load_gnn_split(VAL_JSON), "val", config)
    test_rows = attach_aux_existing(load_gnn_split(TEST_JSON), "test", config)
    splits = {"train": train_rows, "val": val_rows, "test": test_rows}
    for name, rows in splits.items():
        (OUT / f"gnn_{name}.json").write_text(
            json.dumps({"split": name, "records": rows}, indent=2, default=_json_default),
            encoding="utf-8",
        )

    gnn = {name: pred_map(rows) for name, rows in splits.items()}
    truth = {name: true_map(rows) for name, rows in splits.items()}
    print("fitting train-only ratio Ridge on flow features", flush=True)
    ridge = ridge_ratio_maps(splits)

    shrink_alpha, _ = grid_best(
        train_rows,
        lambda pred, alpha: 1.0 + alpha * (pred - 1.0),
        np.linspace(0.35, 1.25, 37),
    )
    log_shift, _ = grid_best(
        train_rows,
        lambda pred, shift: pred * math.exp(shift),
        np.linspace(-0.35, 0.25, 31),
    )
    power, _ = grid_best(
        train_rows,
        lambda pred, beta: np.power(pred, beta),
        np.linspace(0.55, 1.25, 29),
    )
    intercept, slope = two_param_affine(train_rows)
    edges, multipliers = bin_multipliers(train_rows)
    gate = hetero_gate_threshold(train_rows)
    linear = fit_linear(train_rows)
    blend = blend_weight(gnn["train"], ridge["train"], truth["train"])
    contrast_gate, contrast_alpha = _contrast_shrink(train_rows)

    def maps_from(fn: Callable[[dict], float]) -> dict[str, dict[str, float]]:
        return {name: {row_key(row): fn(row) for row in rows} for name, rows in splits.items()}

    policies: dict[str, dict[str, dict[str, float]]] = {
        "gnn_raw": gnn,
        "ridge_compact_trainfit": ridge,
        "gnn_shrink": maps_from(
            lambda row: clip_ratio(1.0 + shrink_alpha * (float(row["inclusion_ratio_estimated"]) - 1.0))
        ),
        "gnn_log_shift": maps_from(
            lambda row: clip_ratio(float(row["inclusion_ratio_estimated"]) * math.exp(log_shift))
        ),
        "gnn_power": maps_from(
            lambda row: clip_ratio(float(row["inclusion_ratio_estimated"]) ** power)
        ),
        "gnn_median_affine": maps_from(
            lambda row: clip_ratio(
                math.exp(intercept + slope * math.log(max(float(row["inclusion_ratio_estimated"]), 1e-6)))
            )
        ),
        "gnn_pred_bins": maps_from(
            lambda row: apply_bins(float(row["inclusion_ratio_estimated"]), edges, multipliers)
        ),
        "gnn_hetero_gate": maps_from(
            lambda row: 1.0
            if float(row["heterogeneity_probability"]) < gate
            else float(row["inclusion_ratio_estimated"])
        ),
        "stack_linear_trainfit": maps_from(lambda row: apply_linear(row, linear)),
        "gnn_contrast_shrink": maps_from(
            lambda row: clip_ratio(
                1.0 + contrast_alpha * (float(row["inclusion_ratio_estimated"]) - 1.0)
            )
            if float(row.get("flow_contrast", 1.0)) < contrast_gate
            else float(row["inclusion_ratio_estimated"])
        ),
        "blend_gnn_ridge": {
            name: {
                key: clip_ratio((1.0 - blend) * gnn[name][key] + blend * ridge[name][key])
                for key in gnn[name]
                if key in ridge[name]
            }
            for name in splits
        },
    }

    leaderboard = []
    for name, mapping in policies.items():
        val_applied = apply_pred(val_rows, mapping["val"])
        test_applied = apply_pred(test_rows, mapping["test"])
        leaderboard.append(
            {
                "name": name,
                "validation": summarize(val_applied),
                "test_after_freeze": summarize(test_applied),
            }
        )
    leaderboard.sort(key=lambda row: selection_key(row["validation"]))
    selected = leaderboard[0]
    report = {
        "schema_version": 1,
        "gate": GATE,
        "selection": "minimum validation median ratio; test unused for selection",
        "selected_policy": selected["name"],
        "train_fit": {
            "shrink_alpha": shrink_alpha,
            "log_shift": log_shift,
            "power": power,
            "affine": {"intercept": intercept, "slope": slope},
            "bins": {"edges": edges, "multipliers": multipliers},
            "hetero_gate": gate,
            "gnn_ridge_blend": blend,
            "contrast_shrink": {"gate": contrast_gate, "alpha": contrast_alpha},
            "train_raw": summarize(train_rows),
        },
        "leaderboard": leaderboard,
    }
    (OUT / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_policy": selected["name"],
                "val": selected["validation"],
                "test_after_freeze": selected["test_after_freeze"],
                "val_leaderboard": [
                    {
                        "name": row["name"],
                        "val": row["validation"]["median"],
                        "val_geom": row["validation"]["geometries_passing"],
                        "test": row["test_after_freeze"]["median"],
                    }
                    for row in leaderboard
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
