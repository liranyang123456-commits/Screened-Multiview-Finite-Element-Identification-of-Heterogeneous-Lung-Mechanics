#!/usr/bin/env python3
"""50-patient screened MAP under structured tracking error. Writes a new folder."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "lung_inverse_rendering" / "evaluate_sim_lung_v2.py"
AGGREGATOR = ROOT / "lung_inverse_rendering" / "aggregate_large_eval.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "lung_mesh_ai_v2" / "structured_tracking_v1",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json",
    )
    parser.add_argument("--patient-limit", type=int, default=50)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mixed"],
    )
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in args.models:
        out = args.results / model
        out.mkdir(exist_ok=True)
        command = [
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
            "--tracking-error-model",
            model,
            "--all-track-frames",
            "--temporal-track-regression",
            "--track-smoothing-iterations",
            "2",
            "--max-nfev",
            "96",
            "--initial-predictions",
            str(args.predictions),
            "--use-predicted-region",
            "--force-prior-sigma",
            "0.05",
            "--material-prior-weight",
            "0.05",
            "--screening-minimum-cost-reduction",
            "0.05",
            "--minimum-refinement-cost-reduction",
            "0.05",
            "--benchmark-method",
            "fem_learned_screened_map",
            "--output-tag",
            f"track_{model}",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        metrics = sorted(out.glob("metrics_*.json"))
        if metrics:
            summaries.append({"model": model, "metrics": str(metrics[-1])})
    (args.results / "sweep_index.json").write_text(
        json.dumps({"jobs": summaries}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
