#!/usr/bin/env python3
"""Auditable paired CT FEM and development-only ratio stability driver.

No held-out test labels are used to select a setting. The locked CT test
geometries are only scored by the paired FEM sweep; all threshold selection
for ratio calibration is restricted to development geometries (train/val).
Outputs are isolated under results/ct_reviewer_drivers_v1 and never overwrite
frozen result directories. Defaults to dry-run; pass --run to execute FEM.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
PREDICTIONS = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
RATIO_PREDICTIONS = ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2"
OUT = ROOT / "results" / "ct_reviewer_drivers_v1"
FEM_SCRIPT = ROOT / "experiments" / "run_screened_fem_locked.py"
WIDTHS = (1.0, 2.0, 3.0)
THRESHOLDS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)
PROBES = (2, 4, 6)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_splits() -> dict[str, list[dict[str, Any]]]:
    patients = load_json(DATASET / "manifest.json")["patients"]
    rows = {split: [row for row in patients if row["split"] == split] for split in ("train", "val", "test")}
    rows["locked_test"] = rows["test"]
    return rows


def verify_inputs(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    benchmark = load_json(PREDICTIONS)
    benchmark_test = [row for row in benchmark["records"] if row.get("method") == "ours_hybrid_initializer"]

    def identity(row: dict[str, Any]) -> str:
        if row.get("geometry_id") and row.get("scenario_id"):
            return f"{row['geometry_id']}/{row['scenario_id']}"
        return str(row.get("patient_id", "")).replace("\\", "/")

    test_ids = {identity(row) for row in splits["locked_test"]}
    bench_ids = {identity(row) for row in benchmark_test}
    if len(splits["locked_test"]) != 120 or test_ids != bench_ids:
        raise RuntimeError(f"locked CT test mismatch: manifest={len(test_ids)} benchmark={len(bench_ids)} overlap={len(test_ids & bench_ids)}")
    for split in ("train", "val", "test"):
        if not (RATIO_PREDICTIONS / f"gnn_{split}.json").exists():
            raise FileNotFoundError(RATIO_PREDICTIONS / f"gnn_{split}.json")
    return {
        "ct_test_count": len(test_ids),
        "ct_test_geometries": sorted({row["geometry_id"] for row in splits["locked_test"]}),
        "development_geometries": sorted({row["geometry_id"] for split in ("train", "val") for row in splits[split]}),
        "frozen_input_hashes": {
            "dataset_manifest": sha256(DATASET / "manifest.json"),
            "hybrid_benchmark": sha256(PREDICTIONS),
            "ratio_train": sha256(RATIO_PREDICTIONS / "gnn_train.json"),
            "ratio_val": sha256(RATIO_PREDICTIONS / "gnn_val.json"),
        },
    }


def candidate_directions(probe_count: int) -> list[float]:
    if probe_count == 2:
        return [-1.0, 1.0]
    import numpy as np
    magnitudes = np.linspace(1.0 / probe_count, 1.0, probe_count // 2)
    return [float(-value) for value in magnitudes[::-1]] + [float(value) for value in magnitudes]


def fem_matrix(
    patient_limit: int | None = None,
    include_mismatch: bool = True,
    output_root: Path = OUT,
    patient_ids: list[str] | None = None,
    include_controls: bool = False,
) -> list[dict[str, Any]]:
    # Reviewer-directed one-factor-at-a-time matrix: complete threshold sweep,
    # probe sweep at the default screen, width sweep at the default screen.
    configs = []
    settings = set()
    for threshold in THRESHOLDS:
        settings.add((2.0, threshold, 2))
    for probes in PROBES:
        settings.add((2.0, 0.05, probes))
    for width in WIDTHS:
        settings.add((width, 0.05, 2))
    for width, threshold, probes in sorted(settings):
        config_id = f"w{width:g}_s{threshold:g}_p{probes}".replace(".", "p")
        result_dir = output_root / "fem_paired" / config_id
        command = [sys.executable, str(FEM_SCRIPT), "--cohort", "ct120", "--results", str(result_dir),
                   "--prior-width-scale", str(width), "--screen-threshold", str(threshold),
                   "--probe-count", str(probes), "--max-nfev", "48"]
        if patient_ids:
            command += ["--patient-ids", *patient_ids]
        elif patient_limit is not None:
            command += ["--patient-limit", str(patient_limit)]
        configs.append({
            "config_id": config_id,
            "factors": {"prior_width_scale": width, "screen_threshold": threshold,
                        "accept_threshold": 0.05, "candidate_probe_count": probes,
                        "candidate_directions": candidate_directions(probes), "model_force_bias": 0.0},
            "paired_population": "same explicit locked test patient ids for every arm",
            "result_dir": str(result_dir), "command": command,
        })
    if include_mismatch:
        config_id = "mismatch_force_p10_baseline"
        result_dir = output_root / "model_mismatch" / config_id
        command = [sys.executable, str(FEM_SCRIPT), "--cohort", "ct120", "--results", str(result_dir),
                   "--prior-width-scale", "2.0", "--screen-threshold", "0.05", "--probe-count", "2",
                   "--model-force-bias", "0.10", "--max-nfev", "48"]
        if patient_ids:
            command += ["--patient-ids", *patient_ids]
        elif patient_limit is not None:
            command += ["--patient-limit", str(patient_limit)]
        configs.append({"config_id": config_id,
                        "factors": {"model_force_bias": 0.10, "paired_baseline": "w2_s0p05_p2"},
                        "paired_population": "same explicit locked test patient ids; intentional +10% force mismatch",
                        "result_dir": str(result_dir), "command": command})
    if include_mismatch:
        for config_id, bias, pose_deg in (
            ("mismatch_force_m10_baseline", -0.10, 0.0),
            ("mismatch_pose_p5_baseline", 0.0, 5.0),
        ):
            result_dir = output_root / "model_mismatch" / config_id
            command = [sys.executable, str(FEM_SCRIPT), "--cohort", "ct120", "--results", str(result_dir),
                       "--prior-width-scale", "2.0", "--screen-threshold", "0.05", "--probe-count", "2",
                       "--model-force-bias", str(bias), "--pose-rotation-error-deg", str(pose_deg), "--max-nfev", "48"]
            if patient_ids:
                command += ["--patient-ids", *patient_ids]
            elif patient_limit is not None:
                command += ["--patient-limit", str(patient_limit)]
            configs.append({"config_id": config_id,
                            "factors": {"model_force_bias": bias, "pose_rotation_error_deg": pose_deg,
                                        "paired_baseline": "w2_s0p05_p2"},
                            "paired_population": "same explicit locked test patient ids; single-factor deliberate mismatch",
                            "result_dir": str(result_dir), "command": command})
    if include_controls:
        controls = [
            ("control_fixed_init_fixed_prior", {"--initialization-mode": "fixed", "--prior-center-mode": "fixed"}),
            ("control_learned_init_fixed_prior", {"--initialization-mode": "learned", "--prior-center-mode": "fixed"}),
            ("control_fixed_init_learned_prior", {"--initialization-mode": "fixed", "--prior-center-mode": "learned"}),
            ("control_learned_init_learned_prior", {"--initialization-mode": "learned", "--prior-center-mode": "learned"}),
            ("control_no_material_prior", {"--material-prior-weight": "0", "--prior-width-scale": "1"}),
            ("control_residual_only", {"--screening-metric": "residual"}),
        ]
        for config_id, overrides in controls:
            result_dir = output_root / "paired_controls" / config_id
            command = [sys.executable, str(FEM_SCRIPT), "--cohort", "ct120", "--results", str(result_dir),
                       "--prior-width-scale", "2.0", "--screen-threshold", "0.05", "--probe-count", "2", "--max-nfev", "48"]
            for key, value in overrides.items():
                command.extend([key, value])
            if patient_ids:
                command += ["--patient-ids", *patient_ids]
            elif patient_limit is not None:
                command += ["--patient-limit", str(patient_limit)]
            configs.append({"config_id": config_id,
                            "factors": {"paired_control": True, **{key.lstrip("-").replace("-", "_"): value for key, value in overrides.items()}},
                            "paired_population": "same explicit locked test patient ids; factorial matched initializer/prior control",
                            "result_dir": str(result_dir), "command": command})
    return configs


def geometry_median(rows: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        geom = str(row.get("geometry_id") or str(row.get("patient_id", "")).split("_scenario_", 1)[0])
        groups.setdefault(geom, []).append(float(row["inclusion_ratio_relative_error"]))
    output = {}
    for geom, values in sorted(groups.items()):
        ordered = sorted(values)
        mid = len(ordered) // 2
        output[geom] = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return output


def score_policy(rows: list[dict[str, Any]], edges: list[float], multipliers: list[float]) -> dict[str, Any]:
    calibrated = []
    for row in rows:
        pred = float(row["inclusion_ratio_estimated"])
        index = next((i for i in range(len(multipliers)) if pred <= edges[i + 1] or i == len(multipliers) - 1), len(multipliers) - 1)
        estimate = min(5.0, max(1.0, pred * multipliers[index]))
        true = float(row["inclusion_ratio_true"])
        calibrated.append({**row, "inclusion_ratio_relative_error": abs(estimate / max(true, 1e-8) - 1.0)})
    errors = sorted(row["inclusion_ratio_relative_error"] for row in calibrated)
    middle = len(errors) // 2
    median = errors[middle] if len(errors) % 2 else 0.5 * (errors[middle - 1] + errors[middle])
    by_geometry = geometry_median(calibrated)
    return {"n": len(rows), "median_error": median, "geometry_medians": by_geometry,
            "geometries_passing_0.25": sum(value <= 0.25 for value in by_geometry.values()),
            "geometry_count": len(by_geometry)}


def fit_bins(train_rows: list[dict[str, Any]], bins: int, width_scale: float = 1.0) -> tuple[list[float], list[float]]:
    import numpy as np
    pred = np.asarray([float(row["inclusion_ratio_estimated"]) for row in train_rows])
    true = np.asarray([float(row["inclusion_ratio_true"]) for row in train_rows])
    edges = np.quantile(pred, np.linspace(0, 1, bins + 1)).tolist()
    edges[0], edges[-1] = 0.0, 10.0
    multipliers = []
    grid = np.linspace(max(0.55, 1.0 - 0.45 * width_scale), min(1.45, 1.0 + 0.45 * width_scale), 37)
    for index in range(bins):
        mask = (pred >= edges[index]) & ((pred < edges[index + 1]) if index < bins - 1 else (pred <= edges[index + 1]))
        if not mask.any():
            multipliers.append(1.0)
            continue
        candidates = []
        for factor in grid:
            error = np.median(np.abs(np.clip(pred[mask] * factor, 1.0, 5.0) / np.maximum(true[mask], 1e-8) - 1.0))
            candidates.append((float(error), abs(float(factor) - 1.0), float(factor)))
        multipliers.append(min(candidates)[2])
    return edges, multipliers


def calibration_stability() -> dict[str, Any]:
    # The held-out test prediction file is intentionally never opened here.
    train = load_json(RATIO_PREDICTIONS / "gnn_train.json")["records"]
    val = load_json(RATIO_PREDICTIONS / "gnn_val.json")["records"]
    geom = lambda row: str(row.get("geometry_id") or str(row["patient_id"]).split("_scenario_", 1)[0])
    train_geoms = sorted({geom(row) for row in train})
    val_geoms = sorted({geom(row) for row in val})
    nested = []
    for bins, width in itertools.product((2, 3, 4), (0.5, 1.0, 1.5)):
        folds = []
        for held in train_geoms:
            fit = [row for row in train if geom(row) != held]
            held_rows = [row for row in train if geom(row) == held]
            edges, multipliers = fit_bins(fit, bins, width)
            folds.append({"held_train_geometry": held, "score": score_policy(held_rows, edges, multipliers)})
        fold_medians = sorted(fold["score"]["median_error"] for fold in folds)
        nested.append({"bins": bins, "width_scale": width,
                       "inner_loo_median": fold_medians[len(fold_medians) // 2],
                       "inner_loo_pass_folds": sum(fold["score"]["geometries_passing_0.25"] > 0 for fold in folds),
                       "folds": folds})
    selected = min(nested, key=lambda row: (row["inner_loo_median"], -row["inner_loo_pass_folds"], row["bins"], row["width_scale"]))
    # Nested leave-one-training-geometry-out resampling quantifies how often
    # each calibration policy is selected without consulting validation/test.
    outer_resampling = []
    for outer_held in train_geoms:
        outer_train = [row for row in train if geom(row) != outer_held]
        outer_test = [row for row in train if geom(row) == outer_held]
        outer_candidates = []
        remaining_geoms = sorted({geom(row) for row in outer_train})
        for bins, width in itertools.product((2, 3, 4), (0.5, 1.0, 1.5)):
            fold_medians = []
            pass_folds = 0
            for inner_held in remaining_geoms:
                inner_fit = [row for row in outer_train if geom(row) != inner_held]
                inner_test = [row for row in outer_train if geom(row) == inner_held]
                inner_edges, inner_multipliers = fit_bins(inner_fit, bins, width)
                inner_score = score_policy(inner_test, inner_edges, inner_multipliers)
                fold_medians.append(inner_score["median_error"])
                pass_folds += int(inner_score["geometries_passing_0.25"] > 0)
            outer_candidates.append({
                "bins": bins, "width_scale": width,
                "inner_loo_median": float(sorted(fold_medians)[len(fold_medians) // 2]),
                "inner_loo_pass_folds": pass_folds,
            })
        outer_selected = min(outer_candidates, key=lambda row: (
            row["inner_loo_median"], -row["inner_loo_pass_folds"], row["bins"], row["width_scale"]
        ))
        outer_edges, outer_multipliers = fit_bins(outer_train, outer_selected["bins"], outer_selected["width_scale"])
        outer_resampling.append({
            "held_geometry": outer_held,
            "selected_policy": {"bins": outer_selected["bins"], "width_scale": outer_selected["width_scale"]},
            "score": score_policy(outer_test, outer_edges, outer_multipliers),
        })
    selection_counts = Counter(
        (row["selected_policy"]["bins"], row["selected_policy"]["width_scale"])
        for row in outer_resampling
    )
    edges, multipliers = fit_bins(train, selected["bins"], selected["width_scale"])
    return {
        "selection_boundary": "inner geometry LOO on train selects; separate val geometries scored once; no held-out test labels loaded",
        "train_geometry_count": len(train_geoms), "val_geometry_count": len(val_geoms),
        "development_geometries": sorted(set(train_geoms + val_geoms)),
        "selected_development_policy": {"bins": selected["bins"], "width_scale": selected["width_scale"]},
        "inner_policy_selection": nested,
        "outer_geometry_resampling": outer_resampling,
        "outer_policy_selection_counts": [
            {"bins": bins, "width_scale": width, "selected_folds": count}
            for (bins, width), count in sorted(selection_counts.items())
        ],
        "outer_geometry_count": len(outer_resampling),
        "outer_selection_frequency_summary": {
            "most_selected_policy": (
                {"bins": selection_counts.most_common(1)[0][0][0],
                 "width_scale": selection_counts.most_common(1)[0][0][1],
                 "folds": selection_counts.most_common(1)[0][1]}
                if selection_counts else None
            ),
            "unique_policies_selected": len(selection_counts),
        },
        "outer_validation": score_policy(val, edges, multipliers),
        "policy_full_train_fit": {"edges": edges, "multipliers": multipliers},
    }



def repeated_outer_geometry_resampling(
    train_rows: list[dict[str, Any]],
    *,
    repeats: int = 100,
    heldout_geometry_count: int = 4,
    seed: int = 20261007,
) -> dict[str, Any]:
    """Repeated nested outer geometry holdouts; validation/test are never read."""
    import numpy as np

    geom = lambda row: str(row.get("geometry_id") or str(row["patient_id"]).split("_scenario_", 1)[0])
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        groups.setdefault(geom(row), []).append(row)
    geometry_ids = sorted(groups)
    if len(geometry_ids) < heldout_geometry_count + 4:
        raise ValueError("not enough training geometries for repeated nested outer resampling")
    rng = np.random.default_rng(seed)
    policies = list(itertools.product((2, 3, 4), (0.5, 1.0, 1.5)))
    selections: Counter[tuple[int, float]] = Counter()
    outer_scores: list[float] = []
    outer_geometry_scores: list[float] = []
    runs = []
    for repeat in range(repeats):
        permutation = rng.permutation(geometry_ids)
        outer_ids = sorted(str(value) for value in permutation[:heldout_geometry_count])
        fit_ids = sorted(str(value) for value in permutation[heldout_geometry_count:])
        outer_train = [row for key in fit_ids for row in groups[key]]
        outer_test = [row for key in outer_ids for row in groups[key]]
        candidates = []
        for bins, width in policies:
            inner_scores = []
            for inner_held in fit_ids:
                inner_fit = [row for key in fit_ids if key != inner_held for row in groups[key]]
                inner_test = groups[inner_held]
                edges, multipliers = fit_bins(inner_fit, bins, width)
                inner_scores.append(score_policy(inner_test, edges, multipliers)["median_error"])
            candidates.append({
                "bins": bins,
                "width_scale": width,
                "inner_loo_median": float(np.median(inner_scores)),
            })
        selected = min(candidates, key=lambda item: (item["inner_loo_median"], item["bins"], item["width_scale"]))
        chosen = (int(selected["bins"]), float(selected["width_scale"]))
        selections[chosen] += 1
        edges, multipliers = fit_bins(outer_train, *chosen)
        score = score_policy(outer_test, edges, multipliers)
        outer_scores.append(float(score["median_error"]))
        outer_geometry_scores.extend(float(value) for value in score["geometry_medians"].values())
        runs.append({
            "repeat": repeat,
            "outer_held_geometries": outer_ids,
            "outer_training_geometries": fit_ids,
            "selected_policy": {"bins": chosen[0], "width_scale": chosen[1]},
            "inner_loo_median": float(selected["inner_loo_median"]),
            "outer_score": score,
        })
    return {
        "protocol": f"{repeats} repeated random {len(geometry_ids) - heldout_geometry_count}/{heldout_geometry_count} geometry holdouts from training geometries; each outer training subset selects among 9 policies by inner leave-one-geometry-out median; held-out outer geometries scored once",
        "seed": seed,
        "repeats": repeats,
        "training_geometry_count": len(geometry_ids),
        "heldout_geometry_count_per_repeat": heldout_geometry_count,
        "selection_counts": [
            {"bins": bins, "width_scale": width, "selected_repeats": count}
            for (bins, width), count in sorted(selections.items())
        ],
        "outer_median_error_quantiles": {
            "q05": float(np.quantile(outer_scores, 0.05)),
            "median": float(np.median(outer_scores)),
            "q95": float(np.quantile(outer_scores, 0.95)),
        },
        "outer_geometry_median_error_quantiles": {
            "q05": float(np.quantile(outer_geometry_scores, 0.05)),
            "median": float(np.median(outer_geometry_scores)),
            "q95": float(np.quantile(outer_geometry_scores, 0.95)),
        },
        "runs": runs,
    }


def paired_effects(configs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = "w2_s0p05_p2"
    files = {row["config_id"]: Path(row["result_dir"]) / "records.jsonl" for row in configs}
    if baseline not in files or not files[baseline].exists():
        return {"status": "baseline_not_run", "baseline": baseline, "comparisons": []}
    def load_map(path: Path) -> dict[str, dict[str, Any]]:
        return {str(row["patient_id"]): row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}
    base = load_map(files[baseline])
    comparisons = []
    for config_id, path in files.items():
        if config_id == baseline or not path.exists():
            continue
        current = load_map(path)
        paired = sorted(set(base) & set(current))
        metrics = {}
        for metric in ("E_background_relative_error", "inclusion_ratio_relative_error", "force_scale_relative_error"):
            deltas = [float(current[key][metric]) - float(base[key][metric]) for key in paired]
            ordered = sorted(deltas)
            median = None if not ordered else ordered[len(ordered)//2] if len(ordered) % 2 else 0.5 * (ordered[len(ordered)//2-1] + ordered[len(ordered)//2])
            metrics[metric] = {"paired_n": len(ordered), "median_delta_vs_baseline": median,
                               "fraction_improved": None if not ordered else sum(value < 0 for value in ordered) / len(ordered)}
        comparisons.append({"config_id": config_id, "paired_patient_ids": paired, "metrics": metrics})
    return {"status": "scored", "baseline": baseline, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="write protocol manifest only; default")
    parser.add_argument("--run", action="store_true", help="execute bounded FEM arm(s)")
    parser.add_argument("--patient-limit", type=int, help="required smoke limit for any execution")
    parser.add_argument("--patient-ids", nargs="+", help="explicit, geometry-balanced paired test subset")
    parser.add_argument("--patient-ids-file", type=Path, help="JSON file containing an explicit paired test subset")
    parser.add_argument("--config-id", action="append", help="run one FEM arm; repeat to run a paired subset")
    parser.add_argument("--output-root", type=Path, default=OUT, help="isolated results directory")
    parser.add_argument("--only-calibration", action="store_true")
    parser.add_argument("--include-controls", action="store_true", help="include matched init/prior and residual-only controls")
    parser.add_argument("--max-arms", type=int, default=55)
    args = parser.parse_args()
    splits = dataset_splits()
    audit = verify_inputs(splits)
    if args.patient_ids_file:
        file_ids = json.loads(args.patient_ids_file.read_text(encoding="utf-8"))
        if not isinstance(file_ids, list) or not all(isinstance(value, str) for value in file_ids):
            parser.error("--patient-ids-file must contain a JSON array of patient ID strings")
        args.patient_ids = file_ids
    if args.run and args.dry_run:
        parser.error("choose --run or --dry-run")
    if args.patient_limit is not None and not 1 <= args.patient_limit <= 120:
        parser.error("--patient-limit must be in 1..120")
    if args.patient_ids:
        locked_ids = {str(row["patient_id"]) for row in splits["locked_test"]}
        unknown_ids = set(args.patient_ids) - locked_ids
        if unknown_ids:
            parser.error(f"--patient-ids include non-test IDs: {sorted(unknown_ids)[:3]}")
        observed_geometries = {str(row["geometry_id"]) for row in splits["locked_test"] if str(row["patient_id"]) in set(args.patient_ids)}
        if observed_geometries != set(audit["ct_test_geometries"]):
            parser.error("--patient-ids must include at least one scene from every held-out geometry")
    arms = [] if args.only_calibration else fem_matrix(
        args.patient_limit, output_root=args.output_root, patient_ids=args.patient_ids,
        include_controls=args.include_controls,
    )
    if args.config_id:
        requested_ids = set(args.config_id)
        arms = [arm for arm in arms if arm["config_id"] in requested_ids]
        missing_ids = requested_ids - {arm["config_id"] for arm in arms}
        if missing_ids:
            parser.error(f"unknown --config-id: {sorted(missing_ids)}")
    if args.run and args.patient_limit is None and not args.patient_ids:
        parser.error("--run requires --patient-limit or explicit --patient-ids; full cohort is blocked by design")
    if len(arms) > args.max_arms:
        parser.error(f"planned {len(arms)} arms exceeds --max-arms={args.max_arms}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "patient_ids_file": str(args.patient_ids_file.resolve()) if args.patient_ids_file else None,
        "patient_ids": list(args.patient_ids) if args.patient_ids else None,
        "frozen_paths_write_protected_by_design": [
            "results/ion_ct_official_prior_screened_fem120_v1",
            "results/ion_ct_expanded120_final", "results/ion_ct_ratio_optimization"],
        "fem_arm_count": len(arms), "fem_matrix": arms,
        "calibration_stability": calibration_stability(),
        "repeated_outer_geometry_resampling": repeated_outer_geometry_resampling(
            load_json(RATIO_PREDICTIONS / "gnn_train.json")["records"]
        ),
        "paired_effects": paired_effects(arms),
        "mode": "run" if args.run else "dry-run",
    }
    (args.output_root / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"plan": str((args.output_root / "plan.json")), "fem_arms": len(arms),
                      "locked_test_count": audit["ct_test_count"], "test_geometries": audit["ct_test_geometries"],
                      "development_geometries": len(audit["development_geometries"]), "mode": plan["mode"]}, indent=2))
    if not args.run:
        return
    for arm in arms:
        result_dir = Path(arm["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        command = list(arm["command"])
        arm_manifest = {
            "schema_version": 1, "config_id": arm["config_id"], "factors": arm["factors"],
            "paired_population": arm["paired_population"], "patient_limit": len(args.patient_ids) if args.patient_ids else args.patient_limit,
            "patient_start": 0,
            "patient_ids": list(args.patient_ids) if args.patient_ids else [str(row["patient_id"]) for row in splits["locked_test"][:args.patient_limit]],
            "frozen_input_hashes": audit["frozen_input_hashes"],
        }
        driver_manifest_path = result_dir / "driver_manifest.json"
        if driver_manifest_path.exists() and load_json(driver_manifest_path) != arm_manifest:
            raise RuntimeError(f"driver checkpoint protocol mismatch at {driver_manifest_path}; choose a fresh directory")
        if not driver_manifest_path.exists():
            driver_manifest_path.write_text(json.dumps(arm_manifest, indent=2), encoding="utf-8")
        print(f"[run] {arm['config_id']}: {subprocess.list2cmdline(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
