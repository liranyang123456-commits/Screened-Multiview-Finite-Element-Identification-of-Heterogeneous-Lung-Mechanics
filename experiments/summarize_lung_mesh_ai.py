"""Create the frozen AIIM-oriented report from persisted JSON evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def percent(value: float | None) -> str:
    return "not run" if value is None else f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-metrics", type=Path)
    args = parser.parse_args()
    stage_files = [
        (
            "40-patient smoke",
            args.smoke_metrics
            if args.smoke_metrics is not None
            else args.results / "smoke40" / "metrics_gnn.json",
        ),
        (
            "100-patient model selection",
            args.results / "stage100" / "cycle_status.json",
        ),
        ("250-patient frozen test", args.results / "full250" / "metrics_gnn.json"),
    ]
    lines = [
        "# Lung Mesh AI v1 frozen report",
        "",
        "## Evidence scope",
        "",
        "- Physics-guided geometric deep learning for patient-specific material identification.",
        "- All material labels and deformations in this report are synthetic.",
        "- Geometry is patient-conditioned or synthetic-CT-surrogate geometry; no real-patient material accuracy is claimed.",
        "- Absolute Young's modulus assumes measured or externally calibrated force, contact and boundary conditions.",
        "",
        "## Stage results",
        "",
    ]
    for name, path in stage_files:
        payload = load(path)
        if payload is None:
            continue
        test = payload["test"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Patients: {test['patient_count']}",
                f"- Background E median relative error: {percent(test['E_background_median_relative_error'])}",
                f"- Inclusion ratio median relative error: {percent(test['inclusion_ratio_median_relative_error'])}",
                f"- Region-center normalized median error: {percent(test.get('center_error_normalized_median'))}",
                f"- Region-radius median relative error: {percent(test.get('radius_relative_error_median'))}",
                f"- 90% E interval coverage: {percent(test['log_E_uncertainty']['coverage']['90'])}",
                f"- 90% ratio interval coverage: {percent(test['log_ratio_uncertainty']['coverage']['90'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Baselines and ablations",
            "",
            "- Population, global-feature MLP and legacy visual ResNet results are retained as separate JSON files.",
            "- Single-load evaluation is an input ablation, not a separately tuned model.",
            "- Force-bias tests perturb measured force only; they do not claim force and E can be jointly identified from deformation.",
            "- The 0.05 px protocol uses persisted image-plane tracks and independent hold-frame noise.",
            "",
            "## FEM correction",
            "",
            "- Fixed-initialization and MeshGNN-initialized pattern searches use the same FEM objective and evaluation budget.",
            "- Reduced-budget comparisons quantify speed through function-evaluation budget, not wall-clock alone.",
            "- Small-sample differentiable-FEM fine-tuning remains exploratory and is not promoted unless it improves held-out patients.",
            "",
            "## AIIM claim boundary",
            "",
            "The supported claim is a synthetic-ground-truth demonstration of physics-guided geometric deep learning for patient-specific material identification. Real de-identified CT geometry without force/material ground truth may support only geometry-domain stability and motion-reconstruction validation.",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
