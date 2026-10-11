#!/usr/bin/env python3
"""Four-start deterministic multistart FEM on the 50 synthetic test patients."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "lung_inverse_rendering" / "evaluate_sim_lung_v2.py"


STARTS = (
    (2500.0, 1.25),
    (5000.0, 1.8),
    (8000.0, 2.8),
    (12000.0, 4.0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "lung_mesh_ai_v2" / "fem_multistart50_v1",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json",
    )
    parser.add_argument("--patient-limit", type=int, default=50)
    parser.add_argument("--total-budget", type=int, default=96)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.predictions.read_text(encoding="utf-8"))
    per_start = args.total_budget // len(STARTS)
    jobs = []
    for index, (e_bg, ratio) in enumerate(STARTS):
        records = []
        for row in source["records"][: args.patient_limit]:
            records.append(
                {
                    "patient_id": row["patient_id"],
                    "E_background_estimated": e_bg,
                    "inclusion_ratio_estimated": ratio,
                    "log_E_std": 10.0,
                    "log_ratio_std": 10.0,
                    "center_fraction_estimated": row["center_fraction_estimated"],
                    "radius_fraction_estimated": row["radius_fraction_estimated"],
                }
            )
        pred_path = args.results / f"multistart_{index}_predictions.json"
        pred_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
        out = args.results / f"start_{index}"
        out.mkdir(exist_ok=True)
        jobs.append(
            [
                sys.executable,
                str(EVALUATOR),
                "--dataset",
                str(ROOT / "dataset" / "sim_lung_ai_v2_multiview250"),
                "--results",
                str(out),
                "--patient-limit",
                str(args.patient_limit),
                "--observation",
                "image_tracks",
                "--multiview-tracks",
                "--minimum-track-confidence",
                "0.2",
                "--region-track-weight",
                "8.0",
                "--track-noise-px",
                "0.05",
                "--all-track-frames",
                "--temporal-track-regression",
                "--track-smoothing-iterations",
                "2",
                "--max-nfev",
                str(per_start),
                "--initial-predictions",
                str(pred_path),
                "--use-predicted-region",
                "--force-prior-sigma",
                "0.05",
                "--material-prior-weight",
                "0.05",
                "--minimum-refinement-cost-reduction",
                "0.05",
                "--benchmark-method",
                "fem_deterministic_multistart",
                "--output-tag",
                f"multistart_{index}",
            ]
        )

    state_path = args.results / "driver_state.json"
    completed = set(json.loads(state_path.read_text(encoding="utf-8"))["completed"]) if state_path.exists() else set()

    def run(index: int, command: list[str]) -> int:
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return index

    pending = [(i, cmd) for i, cmd in enumerate(jobs) if i not in completed]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run, i, cmd) for i, cmd in pending]
        for future in as_completed(futures):
            completed.add(future.result())
            state_path.write_text(json.dumps({"completed": sorted(completed)}, indent=2), encoding="utf-8")

    print(json.dumps({"completed": sorted(completed), "per_start_nfev": per_start}, indent=2))


if __name__ == "__main__":
    main()
