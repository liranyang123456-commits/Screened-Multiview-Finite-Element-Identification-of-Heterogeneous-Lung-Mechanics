"""Run the PINN inverse arm on the frozen 50-patient and 120-scene splits."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.evaluate_sim_lung_v2 import check_patient_consistency, load_patient
from lung_inverse_rendering.pinn_inverse import PINNFitConfig, fit_patient_pinn, official_bins3_regularized


SYNTHETIC_DATASET = ROOT / "dataset" / "sim_lung_ai_v2_multiview250"
SYNTHETIC_INIT = ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json"
CT_DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
CT_HYBRID = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
CT_GNN_2028 = ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2" / "gnn_test.json"


def _official_ratio_key(row: dict) -> str:
    if row.get("geometry_id") and row.get("scenario_id"):
        return f"{row['geometry_id']}|{row['scenario_id']}"
    patient = str(row.get("patient_id") or row.get("entry_id") or "").replace("\\", "/")
    if "/" in patient:
        geometry, scenario = patient.split("/", 1)
        return f"{geometry}|{scenario}"
    if "_scenario_" in patient:
        geometry, scenario = patient.split("_scenario_", 1)
        return f"{geometry}|scenario_{scenario}"
    return patient


def _official_2028_ratio_index() -> dict[str, dict]:
    """Seed-2028 raw ratio plus the locked bins3 calibrator (official hybrid)."""
    index: dict[str, dict] = {}
    for row in _load_json(CT_GNN_2028)["records"]:
        raw = float(row["inclusion_ratio_estimated"])
        calibrated = official_bins3_regularized(raw)
        payload = {
            "inclusion_ratio_estimated": calibrated,
            "inclusion_ratio_raw": raw,
            "inclusion_ratio_true": float(row["inclusion_ratio_true"]),
            "log_ratio_std": float(row.get("log_ratio_std", 0.23)),
        }
        key = _official_ratio_key(row)
        index[key] = payload
        index[_normalize_id(str(row["patient_id"]))] = payload
    return index


def _normalize_id(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_index(path: Path, *, method: str | None = None) -> dict[str, dict]:
    payload = _load_json(path)
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    index: dict[str, dict] = {}
    for row in records:
        if method is not None and row.get("method") != method:
            continue
        key = _normalize_id(str(row.get("patient_id") or row.get("entry_id")))
        index[key] = row
        if "geometry_id" in row and "scenario_id" in row:
            index[_normalize_id(f"{row['geometry_id']}_{row['scenario_id']}")] = row
    return index


def _predicted_region(scene: dict, prior: dict) -> tuple[torch.Tensor, float]:
    minimum = scene["nodes"].amin(dim=0)
    extent = scene["nodes"].amax(dim=0) - minimum
    center_fraction = torch.tensor(prior["center_fraction_estimated"], dtype=torch.float64)
    center = minimum + center_fraction * extent
    radius = float(prior["radius_fraction_estimated"]) * min(float(extent[0]), float(extent[1]))
    return center, radius


def _median_and_bootstrap(values: list[float], replicates: int = 5000) -> dict:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    if len(array) == 0:
        return {"median": math.nan, "bootstrap_95_ci": [math.nan, math.nan]}
    rng = np.random.default_rng(2026)
    samples = [float(np.median(rng.choice(array, size=len(array), replace=True))) for _ in range(replicates)]
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"median": median, "bootstrap_95_ci": [float(low), float(high)]}


def _summarize(records: list[dict]) -> dict:
    heterogeneous = [row for row in records if row.get("heterogeneous_true")]
    return {
        "n": len(records),
        "heterogeneous_n": len(heterogeneous),
        "E_background_median_relative_error": _median_and_bootstrap(
            [row["E_background_relative_error"] for row in records]
        ),
        "inclusion_ratio_median_relative_error": _median_and_bootstrap(
            [row["inclusion_ratio_relative_error"] for row in records]
        ),
        "center_error_normalized_median": _median_and_bootstrap(
            [row["center_error_normalized"] for row in heterogeneous if row["center_error_normalized"] is not None]
        ),
        "radius_relative_error_median": _median_and_bootstrap(
            [row["radius_relative_error"] for row in heterogeneous if row["radius_relative_error"] is not None]
        ),
        "force_scale_median_relative_error": _median_and_bootstrap(
            [row["force_scale_relative_error"] for row in records]
        ),
        "wall_time_seconds_median": float(np.median([row["wall_time_seconds"] for row in records])),
        "steps": records[0]["steps"] if records else None,
    }


def run_split(args: argparse.Namespace) -> dict:
    if args.cohort == "synthetic50":
        dataset = SYNTHETIC_DATASET
        predictions = _prediction_index(SYNTHETIC_INIT)
        official_ratios: dict[str, dict] = {}
        calibrate_ratio = False
        results = ROOT / "results" / "lung_mesh_ai_v2" / "pinn_inverse50_v1"
    else:
        dataset = CT_DATASET
        predictions = _prediction_index(CT_HYBRID, method="ours_hybrid_initializer")
        official_ratios = _official_2028_ratio_index() if args.ratio_source == "official_2028" else {}
        calibrate_ratio = args.ratio_source == "official_2028"
        results = ROOT / "results" / (
            "ion_ct_pinn_inverse120_officialstart_v1"
            if args.ratio_source == "official_2028"
            else "ion_ct_pinn_inverse120_v1"
        )
    if args.results:
        results = args.results
    results.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(dataset / "manifest.json")
    patients = [row for row in manifest["patients"] if row["split"] == "test"]
    patients = patients[args.patient_start :]
    if args.patient_limit is not None:
        patients = patients[: args.patient_limit]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = PINNFitConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        physics_weight=args.physics_weight,
    )
    checkpoint = results / "records.jsonl"
    done: dict[str, dict] = {}
    if args.overwrite and checkpoint.exists():
        checkpoint.unlink()
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["patient_id"])] = row
    records = []
    with checkpoint.open("a", encoding="utf-8") as writer:
        for row in patients:
            patient_id = row["patient_id"]
            if patient_id in done:
                records.append(done[patient_id])
                print(f"[skip] {patient_id}", flush=True)
                continue
            scene, experiments = load_patient(dataset, row)
            check_patient_consistency(scene, experiments)
            prior = predictions.get(_normalize_id(patient_id))
            if prior is None:
                raise KeyError(f"No initializer record for {patient_id}")
            initial_E = float(prior["E_background_estimated"])
            initial_ratio = float(prior["inclusion_ratio_estimated"])
            ratio_source = "initializer"
            official = official_ratios.get(_official_ratio_key(row)) or official_ratios.get(
                _normalize_id(patient_id)
            )
            if args.cohort == "ct120" and args.ratio_source == "official_2028" and official is None:
                raise KeyError(f"No official 2028 ratio for {patient_id}")
            if official is not None:
                initial_ratio = float(official["inclusion_ratio_estimated"])
                ratio_source = "official_2028_bins3"
            elif args.cohort == "ct120" and args.ratio_source == "hybrid_bins3":
                initial_ratio = official_bins3_regularized(initial_ratio)
                ratio_source = "hybrid_initializer_bins3"
            center, radius = _predicted_region(scene, prior)
            material_prior_sigma = (
                max(0.03, float(prior.get("log_E_std", 0.06))),
                max(
                    0.05,
                    float((official or prior).get("log_ratio_std", prior.get("log_ratio_std", 0.23))),
                ),
            )
            estimate = fit_patient_pinn(
                scene,
                experiments,
                initial_E_background=initial_E,
                initial_ratio=initial_ratio,
                inclusion_center=center,
                inclusion_radius=radius,
                material_prior_sigma=material_prior_sigma,
                device=device,
                config=config,
                center_fraction_estimated=list(prior["center_fraction_estimated"]),
                radius_fraction_estimated=float(prior["radius_fraction_estimated"]),
            )
            estimate["geometry_id"] = row.get("geometry_id")
            estimate["scenario_id"] = row.get("scenario_id")
            estimate["ratio_source"] = ratio_source
            writer.write(json.dumps(estimate) + "\n")
            writer.flush()
            records.append(estimate)
            print(
                f"[{len(records)}/{len(patients)}] {patient_id} "
                f"E={estimate['E_background_relative_error']:.3f} "
                f"ratio={estimate['inclusion_ratio_relative_error']:.3f} "
                f"t={estimate['wall_time_seconds']:.1f}s",
                flush=True,
            )
    summary = {
        "protocol": {
            "cohort": args.cohort,
            "same_observations": "image_tracks + multiview temporal regression + Huber 1.5",
            "same_region": "initializer predicted sphere, not optimized",
            "same_endpoints": ["E_background", "inclusion_ratio", "center", "radius"],
            "official_ratio_calibrate": calibrate_ratio,
            "ratio_source": args.ratio_source if args.cohort == "ct120" else "initializer",
            "device": str(device),
            "steps": args.steps,
            "physics_weight": args.physics_weight,
        },
        "pinn_inverse": _summarize(records),
        "reference_screened_map_synthetic": {
            "E_background_median_relative_error": 0.041107,
            "inclusion_ratio_median_relative_error": 0.186924,
            "note": "frozen fair_multiview_fem50_strict; not re-run",
        },
        "reference_official_hybrid_ct": {
            "E_background_median_relative_error": 0.0855,
            "inclusion_ratio_median_relative_error": 0.1777,
            "center_error_normalized_median": 0.0960,
            "radius_relative_error_median": 0.1146,
            "note": "official_lock.json; officialstart uses 2027 E/center/radius + 2028 bins3 ratio",
        },
        "records": records,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["pinn_inverse"], indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("synthetic50", "ct120"), default="synthetic50")
    parser.add_argument("--patient-start", type=int, default=0)
    parser.add_argument("--patient-limit", type=int)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--device")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ratio-source",
        choices=("official_2028", "hybrid_bins3"),
        default="official_2028",
        help="CT ratio start: official 2028+bins3 (default) or bins3 on hybrid initializer",
    )
    parser.add_argument(
        "--no-official-ratio-calibrate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    run_split(args)


if __name__ == "__main__":
    main()
