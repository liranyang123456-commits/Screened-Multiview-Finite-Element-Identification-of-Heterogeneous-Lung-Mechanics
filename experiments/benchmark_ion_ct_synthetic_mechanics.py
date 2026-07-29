"""Zero-shot benchmark on the external ION CT synthetic-mechanics cohort.

The frozen synthetic training cohort is the only source used to fit or select
learned estimators.  Every ION mechanics entry is treated as external test data.
The output keeps common-input, secondary, and oracle evidence strictly separate.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import SimLungGraphDataset, collate_lung_graphs  # noqa: E402
from experiments.benchmark_lung_response_baselines import (  # noqa: E402
    _prediction,
    family_candidates,
    select_main,
    select_radius,
)
from experiments.train_lung_mesh_gnn import build_model, evaluate  # noqa: E402
from experiments.train_lung_response_calibrator import features  # noqa: E402


SEED = 2026
EXPECTED_GEOMETRY_CLUSTERS = 3
METRICS = (
    "E_background_relative_error",
    "inclusion_ratio_relative_error",
    "center_error_normalized",
    "radius_relative_error",
    "node_log_E_mae",
    "material_map_correlation",
    "partition_soft_dice",
    "force_scale_relative_error",
    "function_evaluations",
    "diagnostic_function_evaluations",
    "wall_time_seconds",
    "algorithmic_termination",
    "refinement_accepted",
    "screening_rejected",
)
COMMON_INPUT_METHODS = {
    "ridge",
    "pls",
    "extra_trees",
    "pca_ridge",
}
SECONDARY_METHODS = {
    "training_population",
    "mesh_gnn",
    "fem_fixed_init",
    "fem_deterministic_multistart",
    "fem_learned_screened_map",
}
ORACLE_METHODS = {"fem_oracle_region_force"}
DEFAULT_TRAIN = ROOT / "dataset" / "sim_lung_ai_v2_multiview250"
DEFAULT_EXTERNAL = ROOT / "dataset" / "ion_ct_synthetic_mechanics60"
DEFAULT_CHECKPOINT = (
    ROOT / "results" / "lung_mesh_ai_v2" / "stage250_node_summary" / "best_gnn.pt"
)
DEFAULT_CALIBRATOR_EVIDENCE = (
    ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json"
)


def evidence_tier(method: str) -> str:
    """Return the prespecified evidence tier; unknown methods are rejected."""
    if method in COMMON_INPUT_METHODS:
        return "common_input"
    if method in SECONDARY_METHODS:
        return "secondary"
    if method in ORACLE_METHODS:
        return "oracle"
    raise ValueError(f"Method has no evidence tier: {method}")


def entry_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    geometry_id = str(row["geometry_id"])
    scenario_id = str(row["scenario_id"])
    return geometry_id, scenario_id, f"{geometry_id}/{scenario_id}"


def normalized_external_manifest(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize geometry/scenario entries in memory to the sim_lung v2 loader."""
    manifest_path = root if root.is_file() else root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = str(payload.get("version", ""))
    if "sim_lung_v2" not in version:
        raise ValueError(f"External manifest is not sim_lung v2 compatible: {version!r}")
    source_rows = payload.get("entries", payload.get("patients", []))
    if not source_rows:
        raise ValueError("External manifest has no entries/patients")
    normalized = []
    seen: set[tuple[str, str]] = set()
    for source in source_rows:
        row = dict(source)
        geometry_id, scenario_id, patient_id = entry_identity(row)
        key = (geometry_id, scenario_id)
        if key in seen:
            raise ValueError(f"Duplicate external entry: {geometry_id}/{scenario_id}")
        seen.add(key)
        declared_split = str(row.get("split", "external_test"))
        if declared_split not in {"test", "external_test", "external"}:
            raise ValueError(
                f"External entry {patient_id} has forbidden split {declared_split!r}"
            )
        row.update(
            patient_id=patient_id,
            split="test",
            geometry_id=geometry_id,
            scenario_id=scenario_id,
        )
        normalized.append(row)
    geometries = sorted({row["geometry_id"] for row in normalized})
    if len(geometries) != EXPECTED_GEOMETRY_CLUSTERS:
        raise ValueError(
            f"Expected exactly {EXPECTED_GEOMETRY_CLUSTERS} geometry clusters, "
            f"found {len(geometries)}: {geometries}"
        )
    return payload, normalized


def external_dataset(root: Path) -> SimLungGraphDataset:
    payload, rows = normalized_external_manifest(root)
    dataset = SimLungGraphDataset(root, split=None)
    dataset.manifest = {**payload, "patients": rows}
    dataset.patients = rows
    return dataset


def arrays_from_dataset(
    dataset: Iterable[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    X, targets, heterogeneous, identities = [], [], [], []
    for graph in dataset:
        label = graph["labels"]
        geometry_id, scenario_id = graph["patient_id"].split("/", maxsplit=1)
        X.append(features(graph))
        targets.append(
            [
                float(label["log_E_background"]),
                float(label["log_ratio"]),
                *label["center_fraction"].tolist(),
                float(label["radius_fraction"]),
            ]
        )
        heterogeneous.append(bool(label["heterogeneous"]))
        identities.append(
            {
                "geometry_id": geometry_id,
                "scenario_id": scenario_id,
                "entry_id": graph["patient_id"],
            }
        )
    return (
        np.asarray(X),
        np.asarray(targets),
        np.asarray(heterogeneous, dtype=bool),
        identities,
    )


def training_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = SimLungGraphDataset(root, split="train")
    X, target, heterogeneous, _ = arrays_from_training_dataset(dataset)
    return X, target, heterogeneous


def arrays_from_training_dataset(
    dataset: Iterable[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X, targets, heterogeneous, patient_ids = [], [], [], []
    for graph in dataset:
        label = graph["labels"]
        X.append(features(graph))
        targets.append(
            [
                float(label["log_E_background"]),
                float(label["log_ratio"]),
                *label["center_fraction"].tolist(),
                float(label["radius_fraction"]),
            ]
        )
        heterogeneous.append(bool(label["heterogeneous"]))
        patient_ids.append(str(graph["patient_id"]))
    return (
        np.asarray(X),
        np.asarray(targets),
        np.asarray(heterogeneous, dtype=bool),
        patient_ids,
    )


def fit_or_load_baselines(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    artifact_path: Path,
    force_refit: bool = False,
) -> dict[str, Any]:
    if artifact_path.exists() and not force_refit:
        artifact = joblib.load(artifact_path)
        if artifact.get("schema_version") != 2:
            raise ValueError("Unsupported baseline artifact schema")
        return artifact
    X, target, heterogeneous = train
    feature_lower = np.quantile(X, 0.005, axis=0)
    feature_upper = np.quantile(X, 0.995, axis=0)
    X_model = np.clip(X, feature_lower, feature_upper)
    target_lower = np.min(target, axis=0)
    target_upper = np.max(target, axis=0)
    models: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for family, candidates in family_candidates().items():
        main, main_params, main_board = select_main(
            candidates, X_model, target, heterogeneous
        )
        radius, radius_params, radius_board = select_radius(
            candidates, X_model, target, heterogeneous
        )
        models[family] = {"main": main, "radius": radius}
        selection[family] = {
            "main": main_params,
            "radius": radius_params,
            "main_cv": main_board,
            "radius_cv": radius_board,
        }
    artifact = {
        "schema_version": 2,
        "seed": SEED,
        "training_policy": "frozen synthetic train split only; five-fold train-only CV",
        "feature_definition": "visible_flow_temporal_mean_and_max",
        "models": models,
        "selection": selection,
        "train_support_guard": {
            "feature_quantiles": [0.005, 0.995],
            "feature_lower": feature_lower,
            "feature_upper": feature_upper,
            "target_lower": target_lower,
            "target_upper": target_upper,
            "external_labels_used": False,
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return artifact


def guarded_features(X: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    """Clip external features to train-only empirical support."""
    guard = artifact["train_support_guard"]
    return np.clip(X, guard["feature_lower"], guard["feature_upper"])


def guarded_prediction(
    prediction: np.ndarray, artifact: dict[str, Any]
) -> np.ndarray:
    """Keep predictions within target bounds observed on training only."""
    guard = artifact["train_support_guard"]
    return np.clip(prediction, guard["target_lower"], guard["target_upper"])


def prediction_records(
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
    heterogeneous: np.ndarray,
    identities: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for estimate, truth, is_heterogeneous, identity in zip(
        prediction, target, heterogeneous, identities, strict=True
    ):
        estimated_E, true_E = math.exp(float(estimate[0])), math.exp(float(truth[0]))
        estimated_ratio = math.exp(float(estimate[1]))
        true_ratio = math.exp(float(truth[1]))
        row: dict[str, Any] = {
            **identity,
            "method": method,
            "evidence_tier": evidence_tier(method),
            "external_test": True,
            "E_background_true": true_E,
            "E_background_estimated": estimated_E,
            "E_background_relative_error": abs(estimated_E / true_E - 1.0),
            "inclusion_ratio_true": true_ratio,
            "inclusion_ratio_estimated": estimated_ratio,
            "inclusion_ratio_relative_error": abs(estimated_ratio / true_ratio - 1.0),
            "heterogeneous_true": bool(is_heterogeneous),
            "center_fraction_true": truth[2:5].tolist(),
            "center_fraction_estimated": estimate[2:5].tolist(),
            "radius_fraction_true": float(truth[5]),
            "radius_fraction_estimated": float(estimate[5]),
        }
        if is_heterogeneous:
            row["center_error_normalized"] = float(
                np.linalg.norm(estimate[2:5] - truth[2:5])
            )
            row["radius_relative_error"] = abs(float(estimate[5] / truth[5] - 1.0))
        else:
            row["center_error_normalized"] = None
            row["radius_relative_error"] = None
        rows.append(row)
    return rows


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    q1, median, q3 = np.quantile(array, (0.25, 0.5, 0.75))
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(median),
        "interquartile_range": [float(q1), float(q3)],
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def metric_values(records: Sequence[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(row[metric])
        for row in records
        if row.get(metric) is not None and np.isfinite(row[metric])
    ]


def attach_node_material_metrics(
    records: Sequence[dict[str, Any]], dataset: Iterable[dict[str, Any]]
) -> None:
    """Evaluate scalar/region predictions as nodal material maps."""
    graphs = {str(graph["patient_id"]): graph for graph in dataset}
    epsilon = 1e-8
    for row in records:
        graph = graphs.get(str(row["entry_id"]))
        if (
            graph is None
            or row.get("center_fraction_estimated") is None
            or row.get("radius_fraction_estimated") is None
        ):
            continue
        position = graph["pos"].detach().cpu().numpy()
        truth_log_E = graph["labels"]["node_log_E"].detach().cpu().numpy()
        truth_partition = (
            graph["labels"]["partition"].detach().cpu().numpy().astype(float)
        )
        center = np.asarray(row["center_fraction_estimated"], dtype=float)
        radius = float(row["radius_fraction_estimated"])
        occupancy = (
            np.linalg.norm(position - center[None, :], axis=1) <= radius
        ).astype(float)
        background = max(float(row["E_background_estimated"]), epsilon)
        ratio = max(float(row["inclusion_ratio_estimated"]), epsilon)
        predicted_log_E = np.log(background * (1.0 + (ratio - 1.0) * occupancy))
        row["node_log_E_mae"] = float(
            np.mean(np.abs(predicted_log_E - truth_log_E))
        )
        denominator = float(occupancy.sum() + truth_partition.sum())
        row["partition_soft_dice"] = float(
            (2.0 * np.dot(occupancy, truth_partition) + epsilon)
            / (denominator + epsilon)
        )
        if np.std(truth_log_E) > epsilon and np.std(predicted_log_E) > epsilon:
            truth_centered = truth_log_E - np.mean(truth_log_E)
            prediction_centered = predicted_log_E - np.mean(predicted_log_E)
            correlation = float(
                np.dot(prediction_centered, truth_centered)
                / np.sqrt(
                    np.dot(prediction_centered, prediction_centered)
                    * np.dot(truth_centered, truth_centered)
                )
            )
            row["material_map_correlation"] = (
                correlation if np.isfinite(correlation) else None
            )
        else:
            row["material_map_correlation"] = None


def attach_physics_regions(records: Sequence[dict[str, Any]]) -> None:
    """Restore the fixed region used by each FEM record for map-level metrics."""
    pca = {
        row["entry_id"]: row for row in records if row["method"] == "pca_ridge"
    }
    for row in records:
        if not str(row["method"]).startswith("fem_"):
            continue
        reference = pca.get(row["entry_id"])
        if reference is None:
            continue
        oracle = row["method"] == "fem_oracle_region_force"
        row["center_fraction_estimated"] = reference[
            "center_fraction_true" if oracle else "center_fraction_estimated"
        ]
        row["radius_fraction_estimated"] = reference[
            "radius_fraction_true" if oracle else "radius_fraction_estimated"
        ]
        row["center_error_normalized"] = (
            0.0 if oracle else reference["center_error_normalized"]
        )
        row["radius_relative_error"] = (
            0.0 if oracle else reference["radius_relative_error"]
        )


def geometry_aggregate(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["geometry_id"])].append(row)
    output = []
    for (method, geometry_id), rows in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "evidence_tier": evidence_tier(method),
                "geometry_id": geometry_id,
                "scenario_count": len(rows),
                "metrics": {
                    metric: descriptive(metric_values(rows, metric))
                    for metric in METRICS
                },
            }
        )
    return output


def geometry_cluster_bootstrap(
    records: Sequence[dict[str, Any]],
    metric: str,
    *,
    seed: int = SEED,
    replicates: int = 5000,
) -> dict[str, Any]:
    """Bootstrap geometry clusters, preserving all scenarios in each draw."""
    by_geometry: dict[str, list[float]] = defaultdict(list)
    for row in records:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            by_geometry[row["geometry_id"]].append(float(value))
    geometry_ids = sorted(by_geometry)
    if len(geometry_ids) != EXPECTED_GEOMETRY_CLUSTERS:
        return {
            "available": False,
            "reason": "metric_not_defined_in_all_geometry_clusters",
            "expected_cluster_count": EXPECTED_GEOMETRY_CLUSTERS,
            "observed_cluster_count": len(geometry_ids),
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        sampled = rng.choice(geometry_ids, size=len(geometry_ids), replace=True)
        values = [value for key in sampled for value in by_geometry[key]]
        draws[index] = np.median(values)
    return {
        "estimand": "scenario-level median with geometry-cluster resampling",
        "cluster_variable": "geometry_id",
        "cluster_count": EXPECTED_GEOMETRY_CLUSTERS,
        "replicates": replicates,
        "median": float(np.median(metric_values(records, metric))),
        "bootstrap_95_ci": np.quantile(draws, (0.025, 0.975)).tolist(),
    }


def method_summary(
    records: Sequence[dict[str, Any]], *, replicates: int = 5000
) -> dict[str, Any]:
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        methods[row["method"]].append(row)
    return {
        method: {
            "evidence_tier": evidence_tier(method),
            "entry_count": len(rows),
            "geometry_count": len({row["geometry_id"] for row in rows}),
            "nested_descriptive": {
                "overall_scenarios": {
                    metric: descriptive(metric_values(rows, metric))
                    for metric in METRICS
                },
                "geometry_level_medians": {
                    metric: descriptive(
                        [
                            float(np.median(metric_values(group, metric)))
                            for geometry_id in sorted(
                                {row["geometry_id"] for row in rows}
                            )
                            if (
                                group := [
                                    row for row in rows if row["geometry_id"] == geometry_id
                                ]
                            )
                            and metric_values(group, metric)
                        ]
                    )
                    for metric in METRICS
                },
            },
            "geometry_cluster_bootstrap": {
                metric: (
                    geometry_cluster_bootstrap(
                        rows,
                        metric,
                        seed=SEED + metric_index,
                        replicates=replicates,
                    )
                    if metric_values(rows, metric)
                    else {"available": False, "reason": "metric_not_emitted"}
                )
                for metric_index, metric in enumerate(METRICS)
            },
        }
        for method, rows in sorted(methods.items())
    }


def paired_comparisons(
    records: Sequence[dict[str, Any]],
    reference_method: str = "pca_ridge",
    *,
    replicates: int = 5000,
) -> dict[str, Any]:
    by_method: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_method[row["method"]][(row["geometry_id"], row["scenario_id"])] = row
    if reference_method not in by_method:
        raise ValueError(f"Reference method not found: {reference_method}")
    output = {}
    reference = by_method[reference_method]
    for method, candidate in sorted(by_method.items()):
        if method == reference_method:
            continue
        shared = sorted(set(reference) & set(candidate))
        comparison = {}
        for metric_index, metric in enumerate(METRICS):
            difference_rows = []
            for key in shared:
                ref_value, candidate_value = (
                    reference[key].get(metric),
                    candidate[key].get(metric),
                )
                if ref_value is None or candidate_value is None:
                    continue
                difference_rows.append(
                    {
                        "geometry_id": key[0],
                        "scenario_id": key[1],
                        "difference": float(candidate_value) - float(ref_value),
                    }
                )
            if not difference_rows:
                comparison[metric] = {
                    "available": False,
                    "reason": "metric_not_emitted_by_both_methods",
                }
                continue
            geometries = sorted({row["geometry_id"] for row in difference_rows})
            if len(geometries) != EXPECTED_GEOMETRY_CLUSTERS:
                comparison[metric] = {
                    "available": False,
                    "reason": "paired_metric_not_defined_in_all_geometry_clusters",
                    "observed_cluster_count": len(geometries),
                }
                continue
            rng = np.random.default_rng(SEED + 100 + 10 * metric_index)
            draws = []
            for _ in range(replicates):
                sampled = rng.choice(geometries, size=len(geometries), replace=True)
                values = [
                    row["difference"]
                    for geometry_id in sampled
                    for row in difference_rows
                    if row["geometry_id"] == geometry_id
                ]
                draws.append(float(np.median(values)))
            differences = np.asarray(
                [row["difference"] for row in difference_rows], dtype=float
            )
            comparison[metric] = {
                "estimand": "candidate minus PCA-Ridge paired error",
                "paired_entry_count": len(differences),
                "geometry_cluster_count": EXPECTED_GEOMETRY_CLUSTERS,
                "median_difference": float(np.median(differences)),
                "geometry_cluster_bootstrap_95_ci": np.quantile(
                    draws, (0.025, 0.975)
                ).tolist(),
                "candidate_win_rate": float(np.mean(differences < 0.0)),
            }
        output[method] = {
            "candidate_evidence_tier": evidence_tier(method),
            "reference_evidence_tier": evidence_tier(reference_method),
            "metrics": comparison,
        }
    return output


def load_physics_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load completed FEM outputs and select the best deterministic start."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload.get("benchmark_method", payload.get("method", "")))
        evidence_tier(method)
        for source in payload["records"]:
            row = dict(source)
            patient_id = str(row["patient_id"])
            geometry_id, scenario_id = patient_id.split("/", maxsplit=1)
            row.update(
                entry_id=patient_id,
                geometry_id=geometry_id,
                scenario_id=scenario_id,
                method=method,
                evidence_tier=evidence_tier(method),
                external_test=True,
                used_learned_initialization=(
                    method == "fem_learned_screened_map"
                ),
                region_evidence=(
                    "oracle_true_region"
                    if method == "fem_oracle_region_force"
                    else "frozen_pca_ridge_region"
                ),
                center_error_normalized=row.get("center_error_normalized"),
                radius_relative_error=row.get("radius_relative_error"),
            )
            if method == "fem_oracle_region_force":
                row.update(
                    force_scale_true=1.0,
                    force_scale_estimated=1.0,
                    force_scale_relative_error=0.0,
                    force_observation_source="simulator_truth_oracle",
                )
            row["algorithmic_termination"] = float(
                bool(row.get("converged")) or bool(row.get("screening_rejected"))
            )
            row["refinement_accepted"] = float(
                bool(row.get("refinement_accepted"))
            )
            row["screening_rejected"] = float(bool(row.get("screening_rejected")))
            grouped[(method, geometry_id, scenario_id)].append(row)
    output = []
    for (method, _, _), candidates in sorted(grouped.items()):
        if method == "fem_deterministic_multistart":
            selected = min(
                candidates,
                key=lambda row: float(row.get("cost", float("inf"))),
            )
            selected = {
                **selected,
                "deterministic_starts_completed": len(candidates),
                "function_evaluations": sum(
                    int(row.get("function_evaluations", 0)) for row in candidates
                ),
                "diagnostic_function_evaluations": sum(
                    int(row.get("diagnostic_function_evaluations", 0))
                    for row in candidates
                ),
                "wall_time_seconds": sum(
                    float(row.get("wall_time_seconds", 0.0))
                    for row in candidates
                ),
                "multistart_selection": "lowest final cost",
            }
            output.append(selected)
        elif len(candidates) == 1:
            output.append(candidates[0])
        else:
            raise ValueError(
                f"Duplicate non-multistart FEM result for "
                f"{candidates[0]['entry_id']}/{method}"
            )
    return output


def evaluate_mesh_gnn(
    dataset: Dataset,
    checkpoint: Path,
    identities: Sequence[dict[str, str]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    first = dataset[0]
    actual_dynamic = (
        int(first["dynamic_seq"].shape[-1])
        if first.get("dynamic_seq") is not None
        else None
    )
    compatibility = {
        "checkpoint": str(checkpoint),
        "input_dim_matches": int(first["x"].shape[1]) == int(config["input_dim"]),
        "dynamic_dim_matches": actual_dynamic == config.get("dynamic_dim"),
    }
    compatibility["reusable"] = all(
        compatibility[key]
        for key in ("input_dim_matches", "dynamic_dim_matches")
    )
    if not compatibility["reusable"]:
        return [], compatibility
    model = build_model(
        config["model"],
        config["input_dim"],
        config["hidden_dim"],
        config["layers"],
        config["dropout"],
        config.get("dynamic_dim"),
    ).to(device)
    model.load_state_dict(state["model"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_lung_graphs,
    )
    metrics, rows = evaluate(model, loader, device)
    identity_by_entry = {row["entry_id"]: row for row in identities}
    for row in rows:
        identity = identity_by_entry[row["patient_id"]]
        row.update(
            identity,
            method="mesh_gnn",
            evidence_tier=evidence_tier("mesh_gnn"),
            external_test=True,
        )
    compatibility["checkpoint_metrics"] = metrics
    return rows, compatibility


def write_prediction_json(
    path: Path,
    method: str,
    records: Sequence[dict[str, Any]],
    *,
    oracle_region: bool = False,
    fixed_material: tuple[float, float] | None = None,
    prior_log_stds: tuple[float, float] = (0.25, 0.35),
) -> None:
    output = []
    for row in records:
        E_value, ratio_value = (
            fixed_material
            if fixed_material is not None
            else (
                float(row["E_background_estimated"]),
                float(row["inclusion_ratio_estimated"]),
            )
        )
        center_key = (
            "center_fraction_true" if oracle_region else "center_fraction_estimated"
        )
        radius_key = (
            "radius_fraction_true" if oracle_region else "radius_fraction_estimated"
        )
        output.append(
            {
                "patient_id": row["entry_id"],
                "geometry_id": row["geometry_id"],
                "scenario_id": row["scenario_id"],
                "E_background_estimated": E_value,
                "inclusion_ratio_estimated": ratio_value,
                "log_E_std": 10.0 if fixed_material else prior_log_stds[0],
                "log_ratio_std": 10.0 if fixed_material else prior_log_stds[1],
                "center_fraction_estimated": row[center_key],
                "radius_fraction_estimated": row[radius_key],
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": method,
                "evidence_tier": evidence_tier(method),
                "records": output,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_compat_manifest(
    external_root: Path, output_path: Path
) -> None:
    payload, rows = normalized_external_manifest(external_root)
    source_root = (
        external_root.parent if external_root.is_file() else external_root
    ).resolve()
    absolute_rows = []
    for row in rows:
        row = dict(row)
        row["experiments"] = [
            {
                **experiment,
                "relative_path": str(
                    (source_root / experiment["relative_path"]).resolve()
                ),
            }
            for experiment in row["experiments"]
        ]
        absolute_rows.append(row)
    output_path.write_text(
        json.dumps(
            {
                **payload,
                "patients": absolute_rows,
                "entries": absolute_rows,
                "external_test_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def physics_commands(
    compat_manifest: Path,
    output_dir: Path,
    prediction_paths: dict[str, Path],
    *,
    multistart_total_budget: int = 96,
    multistart_count: int = 4,
) -> list[dict[str, Any]]:
    if multistart_total_budget % multistart_count:
        raise ValueError("Multistart total budget must divide evenly across starts")
    evaluator = ROOT / "lung_inverse_rendering" / "evaluate_sim_lung_v2.py"
    common = [
        sys.executable,
        str(evaluator),
        "--dataset",
        str(compat_manifest),
        "--results",
        str(output_dir / "fem_results"),
        "--split",
        "test",
        "--observation",
        "image_tracks",
        "--multiview-tracks",
        "--track-smoothing-iterations",
        "2",
        "--track-noise-px",
        "0.05",
        "--minimum-track-confidence",
        "0.2",
        "--region-track-weight",
        "8.0",
        "--force-prior-sigma",
        "0.05",
    ]
    jobs = []

    def add(
        job_id: str,
        method: str,
        predictions: Path,
        max_nfev: int,
        extras: Sequence[str] = (),
    ) -> None:
        jobs.append(
            {
                "job_id": job_id,
                "method": method,
                "evidence_tier": evidence_tier(method),
                "max_nfev_per_entry": max_nfev,
                "command": common
                + [
                    "--max-nfev",
                    str(max_nfev),
                    "--initial-predictions",
                    str(predictions),
                    "--use-predicted-region",
                    "--benchmark-method",
                    method,
                    "--output-tag",
                    job_id,
                ]
                + list(extras),
            }
        )

    add(
        "fixed_init",
        "fem_fixed_init",
        prediction_paths["fixed"],
        multistart_total_budget,
    )
    per_start = multistart_total_budget // multistart_count
    deterministic_starts = (
        (2500.0, 1.25),
        (5000.0, 1.8),
        (8000.0, 2.8),
        (12000.0, 4.0),
    )
    if multistart_count != len(deterministic_starts):
        raise ValueError("This frozen driver requires exactly four deterministic starts")
    for index in range(multistart_count):
        add(
            f"multistart_{index}",
            "fem_deterministic_multistart",
            prediction_paths[f"multistart_{index}"],
            per_start,
        )
    add(
        "learned_screened_map",
        "fem_learned_screened_map",
        prediction_paths["learned"],
        multistart_total_budget,
        (
            "--material-prior-weight",
            "0.05",
            "--screening-minimum-cost-reduction",
            "0.05",
            "--minimum-refinement-cost-reduction",
            "0.05",
        ),
    )
    add(
        "oracle_region_force",
        "fem_oracle_region_force",
        prediction_paths["oracle"],
        multistart_total_budget,
        ("--use-true-forces",),
    )
    return jobs


def run_physics_driver(config_path: Path) -> None:
    """Execute pending jobs only; completion markers make the driver resumable."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    state_path = config_path.with_name("physics_driver_state.json")
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"completed": []}
    )
    completed = set(state["completed"])
    for job in payload["jobs"]:
        if job["job_id"] in completed:
            continue
        subprocess.run(job["command"], cwd=ROOT, check=True)
        completed.add(job["job_id"])
        state_path.write_text(
            json.dumps({"completed": sorted(completed)}, indent=2),
            encoding="utf-8",
        )


def prepare_physics(
    output_dir: Path,
    external_root: Path,
    pca_records: Sequence[dict[str, Any]],
    *,
    total_budget: int,
    learned_prior_log_stds: tuple[float, float],
) -> Path:
    physics_dir = output_dir / "physics"
    physics_dir.mkdir(parents=True, exist_ok=True)
    compat_manifest = physics_dir / "external_sim_lung_v2_manifest.json"
    write_compat_manifest(external_root, compat_manifest)
    paths = {
        "fixed": physics_dir / "fixed_init_predictions.json",
        "learned": physics_dir / "learned_screened_map_predictions.json",
        "oracle": physics_dir / "oracle_region_force_predictions.json",
    }
    write_prediction_json(
        paths["fixed"],
        "fem_fixed_init",
        pca_records,
        fixed_material=(5000.0, 1.8),
    )
    write_prediction_json(
        paths["learned"],
        "fem_learned_screened_map",
        pca_records,
        prior_log_stds=learned_prior_log_stds,
    )
    write_prediction_json(
        paths["oracle"],
        "fem_oracle_region_force",
        pca_records,
        oracle_region=True,
        fixed_material=(5000.0, 1.8),
    )
    starts = (
        (2500.0, 1.25),
        (5000.0, 1.8),
        (8000.0, 2.8),
        (12000.0, 4.0),
    )
    for index, start in enumerate(starts):
        path = physics_dir / f"multistart_{index}_predictions.json"
        paths[f"multistart_{index}"] = path
        write_prediction_json(
            path,
            "fem_deterministic_multistart",
            pca_records,
            fixed_material=start,
        )
    jobs = physics_commands(
        compat_manifest,
        physics_dir,
        paths,
        multistart_total_budget=total_budget,
    )
    config_path = physics_dir / "physics_driver.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "external_test_only": True,
                "long_running_tasks_executed_during_preparation": False,
                "inverse_fem_frame_protocol": (
                    "peak frame from each of four loads, all three views; the "
                    "learned estimators retain all seven frames"
                ),
                "multistart": {
                    "deterministic": True,
                    "start_count": 4,
                    "total_nfev_budget_per_entry": total_budget,
                    "nfev_per_start": total_budget // 4,
                    "selection": "lowest final objective across completed starts",
                },
                "evidence_tiers": {
                    "common_input": sorted(COMMON_INPUT_METHODS),
                    "secondary": sorted(SECONDARY_METHODS),
                    "oracle": sorted(ORACLE_METHODS),
                },
                "jobs": jobs,
                "resume_command": [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--run-physics-driver",
                    str(config_path),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--external-dataset", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "ion_ct_synthetic_mechanics60",
    )
    parser.add_argument("--baseline-artifact", type=Path)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--mesh-gnn-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--calibrator-evidence",
        type=Path,
        default=DEFAULT_CALIBRATOR_EVIDENCE,
    )
    parser.add_argument("--skip-mesh-gnn", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--multistart-total-budget", type=int, default=96)
    parser.add_argument(
        "--physics-results",
        nargs="*",
        type=Path,
        default=(),
        help="Completed evaluator JSON files to merge into unified analysis",
    )
    parser.add_argument("--run-physics-driver", type=Path)
    args = parser.parse_args()
    if args.run_physics_driver:
        run_physics_driver(args.run_physics_driver)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.baseline_artifact or args.output_dir / "baseline_models.joblib"
    train = training_arrays(args.train_dataset)
    external = external_dataset(args.external_dataset)
    X_external, target, heterogeneous, identities = arrays_from_dataset(external)
    artifact = fit_or_load_baselines(train, artifact_path, args.force_refit)
    X_external_model = guarded_features(X_external, artifact)

    all_records: list[dict[str, Any]] = []
    population = np.tile(np.mean(train[1], axis=0), (len(X_external), 1))
    population[:, 5] = np.mean(train[1][train[2], 5])
    all_records.extend(
        prediction_records(
            "training_population",
            population,
            target,
            heterogeneous,
            identities,
        )
    )
    for family in ("ridge", "pls", "extra_trees", "pca_ridge"):
        estimators = artifact["models"][family]
        prediction = _prediction(estimators["main"], X_external_model)
        prediction[:, 5] = _prediction(
            estimators["radius"], X_external_model
        ).reshape(-1)
        prediction = guarded_prediction(prediction, artifact)
        all_records.extend(
            prediction_records(
                family, prediction, target, heterogeneous, identities
            )
        )

    mesh_audit: dict[str, Any] = {
        "requested": not args.skip_mesh_gnn,
        "checkpoint": str(args.mesh_gnn_checkpoint),
        "reusable": False,
    }
    if not args.skip_mesh_gnn and args.mesh_gnn_checkpoint.exists():
        mesh_records, mesh_audit = evaluate_mesh_gnn(
            external, args.mesh_gnn_checkpoint, identities, args.batch_size
        )
        all_records.extend(mesh_records)
    elif not args.skip_mesh_gnn:
        mesh_audit["reason"] = "checkpoint_not_found"
    if args.physics_results:
        all_records.extend(load_physics_records(args.physics_results))
    attach_physics_regions(all_records)
    attach_node_material_metrics(all_records, external)

    pca_records = [row for row in all_records if row["method"] == "pca_ridge"]
    calibrator_evidence = json.loads(
        args.calibrator_evidence.read_text(encoding="utf-8")
    )
    prior_payload = calibrator_evidence[
        "validation_residual_log_std_for_fem_prior"
    ]
    learned_prior_log_stds = (
        float(prior_payload["log_E_background"]),
        float(prior_payload["log_ratio"]),
    )
    physics_config = prepare_physics(
        args.output_dir,
        args.external_dataset,
        pca_records,
        total_budget=args.multistart_total_budget,
        learned_prior_log_stds=learned_prior_log_stds,
    )
    output = {
        "schema_version": 1,
        "protocol": {
            "training_dataset": str(args.train_dataset),
            "external_dataset": str(args.external_dataset),
            "external_test_only": True,
            "selection": "frozen synthetic train cohort only; no external labels used",
            "train_support_guard": (
                "features clipped to train 0.5--99.5% quantiles and outputs to "
                "train target extrema; no external labels used"
            ),
            "identity": ["geometry_id", "scenario_id"],
            "geometry_cluster_count": EXPECTED_GEOMETRY_CLUSTERS,
            "geometry_cluster_bootstrap_replicates": args.bootstrap_replicates,
            "evidence_tiers": {
                "common_input": sorted(COMMON_INPUT_METHODS),
                "secondary": sorted(SECONDARY_METHODS),
                "oracle": sorted(ORACLE_METHODS),
            },
        },
        "artifact_audit": {
            "baseline_artifact": str(artifact_path),
            "baseline_loaded_or_trained": sorted(artifact["models"]),
            "mesh_gnn": mesh_audit,
        },
        "records": all_records,
        "geometry_aggregates": geometry_aggregate(all_records),
        "methods": method_summary(
            all_records, replicates=args.bootstrap_replicates
        ),
        "paired_against_pca_ridge": paired_comparisons(
            all_records,
            replicates=args.bootstrap_replicates,
        ),
        "physics_driver": str(physics_config),
        "evidence_boundary": (
            "Common-input results use identical response summaries. MeshGNN and "
            "all FEM variants are secondary. Oracle region+force is isolated and "
            "must never be pooled with non-oracle performance."
        ),
    }
    output_path = args.output_dir / "benchmark.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "methods": sorted(output["methods"]),
                "physics_driver": str(physics_config),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
