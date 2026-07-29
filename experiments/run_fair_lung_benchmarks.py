"""Run paired fixed- versus GNN-initialized FEM fits on identical patients."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "lung_inverse_rendering" / "evaluate_sim_lung_v2.py"
AGGREGATOR = ROOT / "lung_inverse_rendering" / "aggregate_large_eval.py"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def fixed_region_record(row: dict, source: Path, region_keys: set[str]) -> dict:
    region = {key: row[key] for key in region_keys if key in row}
    for key, value in list(region.items()):
        if key.endswith("_path"):
            path = Path(value)
            region[key] = str(path if path.is_absolute() else (source.parent / path).resolve())
    return {
        "patient_id": row["patient_id"],
        "E_background_estimated": 5000.0,
        "inclusion_ratio_estimated": 1.8,
        "log_E_std": 10.0,
        "log_ratio_std": 10.0,
        **region,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gnn-predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--patient-start", type=int, default=0)
    parser.add_argument("--patient-limit", type=int, default=50)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--convergence-threshold", type=float, default=0.01)
    parser.add_argument(
        "--force-prior-sigma", type=float, choices=(0.02, 0.05, 0.10), default=0.05
    )
    parser.add_argument(
        "--region-condition", choices=("known", "predicted"), default="known"
    )
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    fixed_predictions: Path | None = None
    if args.region_condition == "predicted":
        payload = json.loads(args.gnn_predictions.read_text(encoding="utf-8"))
        region_keys = {
            "center_fraction_estimated",
            "radius_fraction_estimated",
            "node_partition",
            "node_partition_path",
            "sdf",
            "sdf_path",
            "soft_occupancy",
            "soft_occupancy_path",
            "E_nodes",
            "E_nodes_path",
        }
        fixed_payload = {
            **{key: value for key, value in payload.items() if key != "records"},
            "records": [
                fixed_region_record(row, args.gnn_predictions, region_keys)
                for row in payload["records"]
            ],
        }
        fixed_predictions = args.results / "_fixed_shared_region_predictions.json"
        fixed_predictions.write_text(json.dumps(fixed_payload), encoding="utf-8")

    common = [
        sys.executable,
        str(EVALUATOR),
        "--dataset",
        str(args.dataset),
        "--results",
        str(args.results),
        "--observation",
        "noisy",
        "--patient-start",
        str(args.patient_start),
        "--patient-limit",
        str(args.patient_limit),
        "--max-nfev",
        str(args.max_nfev),
        "--convergence-threshold",
        str(args.convergence_threshold),
        "--force-prior-sigma",
        str(args.force_prior_sigma),
    ]
    region_args = ["--use-predicted-region"] if args.region_condition == "predicted" else []
    fixed = common + [
        "--benchmark-method",
        "fixed_init",
        "--output-tag",
        "fair_fixed",
    ]
    if fixed_predictions is not None:
        fixed += ["--initial-predictions", str(fixed_predictions)]
    run(fixed + region_args)
    run(
        common
        + [
            "--benchmark-method",
            "gnn_init",
            "--output-tag",
            "fair_gnn",
            "--initial-predictions",
            str(args.gnn_predictions),
        ]
        + region_args
    )

    sigma_suffix = f"{args.force_prior_sigma:.2f}"
    predicted_suffix = "_predicted_region" if region_args else ""
    fixed_gnn_suffix = "_gnn_init" if fixed_predictions is not None else ""
    fixed_result = (
        args.results
        / (
            f"metrics_noisy_force_map_sigma_{sigma_suffix}_fair_fixed"
            f"{fixed_gnn_suffix}{predicted_suffix}.json"
        )
    )
    gnn_result = (
        args.results
        / (
            f"metrics_noisy_force_map_sigma_{sigma_suffix}_fair_gnn"
            f"_gnn_init{predicted_suffix}.json"
        )
    )
    run(
        [
            sys.executable,
            str(AGGREGATOR),
            "--inputs",
            str(fixed_result),
            str(gnn_result),
            "--reference-method",
            "fixed_init",
            "--out",
            str(args.results / "fair_benchmark_summary.json"),
        ]
    )


if __name__ == "__main__":
    main()
