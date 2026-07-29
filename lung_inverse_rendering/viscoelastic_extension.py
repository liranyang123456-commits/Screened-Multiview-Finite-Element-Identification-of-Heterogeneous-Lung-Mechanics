"""First-order viscoelastic loading/hold/unloading extension for sim_lung_v2.

This is a reduced-order relaxation model over FEM equilibrium displacements:
tau * du/dt + u = u_equilibrium. It tests temporal identifiability, but is not
yet a finite-strain fibre-reinforced constitutive integration.
"""
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


DATASET = ROOT / "dataset" / "sim_lung_v2"
RESULTS = ROOT / "results" / "sim_lung_v2"


def relaxation_sequence(
    equilibrium: torch.Tensor, tau_seconds: float, dt_seconds: float
) -> torch.Tensor:
    decay = math.exp(-dt_seconds / tau_seconds)
    state = torch.zeros_like(equilibrium[0])
    rows = []
    for target in equilibrium:
        state = decay * state + (1.0 - decay) * target
        rows.append(state)
    return torch.stack(rows)


def generate_extension(
    dataset: Path,
    manifest: dict,
    *,
    dt_seconds: float,
    noise_std: float,
) -> dict:
    rows = []
    for patient_index, patient in enumerate(manifest["patients"]):
        tau = 0.35 + 0.15 * patient_index
        experiments = []
        for experiment_index, experiment in enumerate(patient["experiments"]):
            path = dataset / experiment["relative_path"]
            gt = torch.load(path, map_location="cpu", weights_only=False)
            equilibrium = gt["surface_motion_true"].to(torch.float64)
            true_motion = relaxation_sequence(equilibrium, tau, dt_seconds)
            generator = torch.Generator().manual_seed(
                patient["geometry_seed"] + 4000 + experiment_index
            )
            observed = true_motion + noise_std * torch.randn(
                true_motion.shape, generator=generator, dtype=true_motion.dtype
            )
            output = path.parent / "viscoelastic.pt"
            torch.save(
                {
                    "schema_version": "sim_lung_v2_viscoelastic",
                    "patient_id": patient["patient_id"],
                    "experiment": experiment["name"],
                    "dt_seconds": dt_seconds,
                    "tau_seconds": tau,
                    "model": "first_order_relaxation_over_FEM_equilibrium",
                    "surface_motion_equilibrium": equilibrium,
                    "surface_motion_viscoelastic_true": true_motion,
                    "surface_motion_viscoelastic_observed": observed,
                    "noise_std": noise_std,
                },
                output,
            )
            experiments.append(str(output.relative_to(dataset)))
        rows.append(
            {
                "patient_id": patient["patient_id"],
                "split": patient["split"],
                "tau_seconds": tau,
                "experiments": experiments,
            }
        )
    extension = {
        "version": "sim_lung_v2_viscoelastic",
        "model": "tau*du/dt + u = u_equilibrium",
        "dt_seconds": dt_seconds,
        "noise_std": noise_std,
        "patients": rows,
    }
    (dataset / "viscoelastic_manifest.json").write_text(
        json.dumps(extension, indent=2), encoding="utf-8"
    )
    return extension


def fit_tau(experiment_paths: list[Path]) -> tuple[float, float]:
    experiments = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in experiment_paths
    ]
    candidates = np.geomspace(0.10, 3.0, 100)
    scored = []
    for tau in candidates:
        losses = []
        for experiment in experiments:
            prediction = relaxation_sequence(
                experiment["surface_motion_equilibrium"],
                float(tau),
                float(experiment["dt_seconds"]),
            )
            observation = experiment["surface_motion_viscoelastic_observed"]
            scale = max(float(torch.sqrt(observation.square().mean())), 1e-6)
            losses.append(float((((prediction - observation) / scale) ** 2).mean()))
        scored.append(float(np.mean(losses)))
    index = int(np.argmin(scored))
    return float(candidates[index]), scored[index]


def evaluate(dataset: Path, extension: dict) -> dict:
    records = []
    for patient in [row for row in extension["patients"] if row["split"] == "test"]:
        estimate, cost = fit_tau([dataset / path for path in patient["experiments"]])
        error = abs(estimate - patient["tau_seconds"]) / patient["tau_seconds"]
        record = {
            "patient_id": patient["patient_id"],
            "tau_seconds_true": patient["tau_seconds"],
            "tau_seconds_estimated": estimate,
            "tau_relative_error": error,
            "experiment_count": len(patient["experiments"]),
            "cost": cost,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "protocol": "known FEM equilibrium response; noisy temporal surface motion",
        "model": extension["model"],
        "test_patient_count": len(records),
        "tau_median_relative_error": float(
            np.median([row["tau_relative_error"] for row in records])
        ),
        "records": records,
        "limitation": (
            "conditional temporal test; elastic parameters and equilibrium "
            "responses are not jointly estimated"
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics_viscoelastic.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--dt", type=float, default=0.20)
    parser.add_argument("--noise-std", type=float, default=2.5e-4)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    extension = generate_extension(
        args.dataset, manifest, dt_seconds=args.dt, noise_std=args.noise_std
    )
    print(json.dumps(evaluate(args.dataset, extension), indent=2))


if __name__ == "__main__":
    main()
