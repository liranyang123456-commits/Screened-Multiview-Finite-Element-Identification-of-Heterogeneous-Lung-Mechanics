"""AIIM pre-submission sensitivity sweeps (R1) and no-screen counterfactual (R2).

Runs the frozen screened-MAP protocol on the 50 test patients under:
  - noscreen        : screening=0 and acceptance=0 (bare FEM from learned init)
  - noise_{px}      : track-noise sweep 0.01/0.10/0.20/0.50 px (0.05 is frozen)
  - force_{bias}    : force-scale systematic bias -0.10/+0.05/+0.10/+0.20 (0 frozen)

Each configuration is sharded (5 x 10 patients) and shards run concurrently.
Shard metrics JSONs are merged into one JSON per configuration plus a combined
sensitivity_summary.json. Frozen evidence directories are never touched.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "lung_inverse_rendering" / "evaluate_sim_lung_v2.py"
DATASET = ROOT / "dataset" / "sim_lung_ai_v2_multiview250"
PREDICTIONS = ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json"
FROZEN_SCREENED = (
    ROOT
    / "results"
    / "lung_mesh_ai_v2"
    / "fem_multiview_screened50_strict"
    / "metrics_image_tracks_force_map_sigma_0.05_px_0.05_nodes_allframes_"
    "temporalreg_smooth2_multiview_conf0.20_regionweight8.0_"
    "frozen50_screen05_strict_gnn_init_predicted_region.json"
)

SHARDS = 5
PATIENTS = 50

CONFIGS: dict[str, dict[str, object]] = {
    "noscreen": {
        "screening": 0.0,
        "acceptance": 0.0,
        "track_noise_px": 0.05,
        "force_scale_error": 0.0,
    },
    "noise_0.01": {"track_noise_px": 0.01},
    "noise_0.10": {"track_noise_px": 0.10},
    "noise_0.20": {"track_noise_px": 0.20},
    "noise_0.50": {"track_noise_px": 0.50},
    "force_m0.10": {"force_scale_error": -0.10},
    "force_p0.05": {"force_scale_error": 0.05},
    "force_p0.10": {"force_scale_error": 0.10},
    "force_p0.20": {"force_scale_error": 0.20},
}


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def shard_command(
    config: str,
    shard: int,
    results_dir: Path,
    spec: dict[str, object],
) -> list[str]:
    screening = float(spec.get("screening", 0.05))
    acceptance = float(spec.get("acceptance", 0.05))
    noise = float(spec.get("track_noise_px", 0.05))
    force_error = float(spec.get("force_scale_error", 0.0))
    return [
        sys.executable,
        str(EVALUATOR),
        "--dataset",
        str(DATASET),
        "--results",
        str(results_dir),
        "--observation",
        "image_tracks",
        "--multiview-tracks",
        "--minimum-track-confidence",
        "0.2",
        "--region-track-weight",
        "8.0",
        "--all-track-frames",
        "--temporal-track-regression",
        "--track-smoothing-iterations",
        "2",
        "--max-nfev",
        "96",
        "--initial-predictions",
        str(PREDICTIONS),
        "--use-predicted-region",
        "--force-prior-sigma",
        "0.05",
        "--material-prior-weight",
        "0.05",
        "--split",
        "test",
        "--track-noise-px",
        f"{noise:.2f}",
        "--force-scale-error",
        f"{force_error:.2f}",
        "--screening-minimum-cost-reduction",
        f"{screening:.2f}",
        "--minimum-refinement-cost-reduction",
        f"{acceptance:.2f}",
        "--patient-start",
        str(shard * (PATIENTS // SHARDS)),
        "--patient-limit",
        str(PATIENTS // SHARDS),
        "--output-tag",
        f"aiim_{config}_shard{shard}",
    ]


def find_shard_file(results_dir: Path, config: str, shard: int) -> Path:
    tag = f"aiim_{config}_shard{shard}"
    matches = sorted(results_dir.glob(f"metrics_*{tag}_gnn_init_predicted_region.json"))
    if not matches:
        raise FileNotFoundError(f"missing shard output for {config} shard {shard}")
    return matches[0]


def run_shard(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"shard failed ({completed.returncode}): {log_path}")


def summarize_records(records: list[dict]) -> dict:
    return {
        "patients": len(records),
        "E_background_median_relative_error": median(
            [float(r["E_background_relative_error"]) for r in records]
        ),
        "inclusion_ratio_median_relative_error": median(
            [float(r["inclusion_ratio_relative_error"]) for r in records]
        ),
        "force_scale_median_relative_error": median(
            [float(r["force_scale_relative_error"]) for r in records]
        ),
        "refinement_acceptance_rate": sum(
            bool(r["refinement_accepted"]) for r in records
        )
        / len(records),
        "screening_rejection_rate": sum(
            bool(r.get("screening_rejected", False)) for r in records
        )
        / len(records),
        "function_evaluations_median": median(
            [float(r["function_evaluations"]) for r in records]
        ),
        "wall_time_seconds_median": median(
            [float(r["wall_time_seconds"]) for r in records]
        ),
    }


def run_config(
    config: str,
    spec: dict[str, object],
    results_dir: Path,
    logs_dir: Path,
    workers: int,
) -> dict:
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for shard in range(SHARDS):
            command = shard_command(config, shard, results_dir, spec)
            log_path = logs_dir / f"{config}_shard{shard}.log"
            futures[pool.submit(run_shard, command, log_path)] = shard
        for future in as_completed(futures):
            future.result()
            print(f"[{config}] shard {futures[future]} done", flush=True)
    records: list[dict] = []
    for shard in range(SHARDS):
        payload = json.loads(find_shard_file(results_dir, config, shard).read_text())
        records.extend(payload["records"])
    records.sort(key=lambda row: row["patient_id"])
    merged = {
        "config": config,
        "spec": {
            "screening_minimum_cost_reduction": float(spec.get("screening", 0.05)),
            "minimum_refinement_cost_reduction": float(spec.get("acceptance", 0.05)),
            "track_noise_px": float(spec.get("track_noise_px", 0.05)),
            "force_scale_error": float(spec.get("force_scale_error", 0.0)),
        },
        "runtime_seconds": time.time() - started,
        **summarize_records(records),
        "records": records,
    }
    (results_dir / f"sensitivity_{config}.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    print(
        f"[{config}] done in {merged['runtime_seconds']:.0f}s; "
        f"E={merged['E_background_median_relative_error']:.4f} "
        f"ratio={merged['inclusion_ratio_median_relative_error']:.4f} "
        f"accept={merged['refinement_acceptance_rate']:.2f} "
        f"nfev={merged['function_evaluations_median']:.0f}",
        flush=True,
    )
    return merged


def paired_deltas(
    candidate: dict, reference: dict, key_a: str, key_b: str
) -> dict:
    ref_records = {row["patient_id"]: row for row in reference["records"]}
    deltas = []
    wins = 0
    for row in candidate["records"]:
        ref = ref_records.get(row["patient_id"])
        if ref is None:
            continue
        delta = float(row[key_a]) - float(ref[key_a])
        deltas.append(delta)
        if delta < 0:
            wins += 1
    return {
        "median_delta": median(deltas),
        "win_rate": wins / max(len(deltas), 1),
        "pairs": len(deltas),
        "metric": key_a,
        "reference": key_b,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "lung_mesh_ai_v2" / "aiim_sensitivity",
    )
    parser.add_argument("--workers", type=int, default=SHARDS)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=list(CONFIGS),
        choices=list(CONFIGS),
        help="subset of configurations to run",
    )
    args = parser.parse_args()

    results_dir: Path = args.results
    logs_dir = results_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    reference = json.loads(FROZEN_SCREENED.read_text(encoding="utf-8"))
    summary = {
        "reference": {
            "name": "frozen screened MAP (track noise 0.05 px, force bias 0)",
            **summarize_records(reference["records"]),
        },
        "configurations": {},
    }

    for config in args.configs:
        merged = run_config(
            config, CONFIGS[config], results_dir, logs_dir, args.workers
        )
        summary["configurations"][config] = {
            key: value for key, value in merged.items() if key != "records"
        }
        summary["configurations"][config]["paired_vs_reference_E"] = paired_deltas(
            merged, reference, "E_background_relative_error", "frozen_screened"
        )
        summary["configurations"][config]["paired_vs_reference_ratio"] = (
            paired_deltas(
                merged, reference, "inclusion_ratio_relative_error", "frozen_screened"
            )
        )
        (results_dir / "sensitivity_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary["configurations"], indent=2)[:2000], flush=True)


if __name__ == "__main__":
    main()
