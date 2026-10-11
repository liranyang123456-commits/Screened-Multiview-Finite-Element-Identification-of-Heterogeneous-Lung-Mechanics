"""Merge AIIM sensitivity outputs from the two sweep drivers (dirs A and B).

Copies sensitivity_<config>.json from both directories into dir A and rebuilds
sensitivity_summary.json with paired deltas against the frozen screened arm.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR_A = ROOT / "results" / "lung_mesh_ai_v2" / "aiim_sensitivity"
DIR_B = ROOT / "results" / "lung_mesh_ai_v2" / "aiim_sensitivity_b"
FROZEN = (
    ROOT
    / "results"
    / "lung_mesh_ai_v2"
    / "fem_multiview_screened50_strict"
    / "metrics_image_tracks_force_map_sigma_0.05_px_0.05_nodes_allframes_"
    "temporalreg_smooth2_multiview_conf0.20_regionweight8.0_"
    "frozen50_screen05_strict_gnn_init_predicted_region.json"
)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


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


def paired(candidate: dict, reference: dict, key: str) -> dict:
    ref = {row["patient_id"]: row for row in reference["records"]}
    deltas, wins = [], 0
    for row in candidate["records"]:
        if row["patient_id"] not in ref:
            continue
        delta = float(row[key]) - float(ref[row["patient_id"]][key])
        deltas.append(delta)
        wins += delta < 0
    return {"median_delta": median(deltas), "win_rate": wins / len(deltas), "pairs": len(deltas)}


def main() -> None:
    merged: dict[str, dict] = {}
    for directory in (DIR_A, DIR_B):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("sensitivity_*.json")):
            if path.stem == "sensitivity_summary":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            merged[payload["config"]] = payload
            if directory != DIR_A:
                shutil.copy2(path, DIR_A / path.name)

    reference = json.loads(FROZEN.read_text(encoding="utf-8"))
    summary = {
        "reference": {
            "name": "frozen screened MAP (track noise 0.05 px, force bias 0)",
            **summarize_records(reference["records"]),
        },
        "configurations": {},
    }
    for name, payload in sorted(merged.items()):
        entry = {k: v for k, v in payload.items() if k != "records"}
        entry["paired_vs_reference_E"] = paired(
            payload, reference, "E_background_relative_error"
        )
        entry["paired_vs_reference_ratio"] = paired(
            payload, reference, "inclusion_ratio_relative_error"
        )
        summary["configurations"][name] = entry
    (DIR_A / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"merged {len(merged)} configs: {sorted(merged)}")
    for name, row in summary["configurations"].items():
        print(
            f"{name:14s} E={row['E_background_median_relative_error']:.4f} "
            f"ratio={row['inclusion_ratio_median_relative_error']:.4f} "
            f"accept={row['refinement_acceptance_rate']:.2f} "
            f"nfev={row['function_evaluations_median']:.0f}"
        )


if __name__ == "__main__":
    main()
