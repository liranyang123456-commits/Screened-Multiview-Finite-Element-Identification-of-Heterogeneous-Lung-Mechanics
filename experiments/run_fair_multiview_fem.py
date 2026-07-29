"""Run fixed-init FEM under the frozen multiview protocol and pair with AI-init."""
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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ai-result", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--patient-limit", type=int, default=50)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    fixed_records = []
    for row in payload["records"]:
        fixed_records.append(
            {
                "patient_id": row["patient_id"],
                "E_background_estimated": 5000.0,
                "inclusion_ratio_estimated": 1.8,
                "log_E_std": 10.0,
                "log_ratio_std": 10.0,
                "center_fraction_estimated": row["center_fraction_estimated"],
                "radius_fraction_estimated": row["radius_fraction_estimated"],
            }
        )
    fixed_predictions = args.results / "fixed_shared_region_predictions.json"
    fixed_predictions.write_text(
        json.dumps({"records": fixed_records}, indent=2), encoding="utf-8"
    )
    command = [
        sys.executable,
        str(EVALUATOR),
        "--dataset",
        str(args.dataset),
        "--results",
        str(args.results),
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
        "96",
        "--initial-predictions",
        str(fixed_predictions),
        "--use-predicted-region",
        "--force-prior-sigma",
        "0.05",
        "--material-prior-weight",
        "0.05",
        "--minimum-refinement-cost-reduction",
        "0.05",
        "--benchmark-method",
        "fixed_init",
        "--output-tag",
        "frozen50_fixed",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    matches = sorted(args.results.glob("metrics_*frozen50_fixed*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fixed result, found {len(matches)}")
    subprocess.run(
        [
            sys.executable,
            str(AGGREGATOR),
            "--inputs",
            str(matches[0]),
            str(args.ai_result),
            "--reference-method",
            "fixed_init",
            "--out",
            str(args.results / "fair_multiview_summary.json"),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
