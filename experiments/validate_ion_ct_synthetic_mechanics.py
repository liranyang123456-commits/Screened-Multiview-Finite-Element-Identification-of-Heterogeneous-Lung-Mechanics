"""Validate the frozen real-CT-geometry/synthetic-mechanics dataset protocol."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics60"
DEFAULT_OUTPUT = (
    ROOT / "results" / "ion_ct_synthetic_mechanics" / "protocol_validation.json"
)
FORBIDDEN = re.compile(
    r"(?:[A-Za-z]:\\|省医|case_\d|SeriesInstanceUID|StudyInstanceUID|PatientName)"
)
MATCHED_FIELDS = (
    "E_background",
    "inclusion_ratio",
    "inclusion_center_fraction",
    "inclusion_radius_fraction",
    "boundary_condition",
)


def validate_dataset(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    if FORBIDDEN.search(manifest_text):
        raise ValueError("Manifest contains a forbidden source identifier or path")
    payload = json.loads(manifest_text)
    patients = payload["patients"]
    geometry_ids = sorted({str(row["geometry_id"]) for row in patients})
    if len(geometry_ids) != 3:
        raise ValueError(f"Expected three geometries, found {len(geometry_ids)}")
    by_geometry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in patients:
        by_geometry[str(row["geometry_id"])].append(row)
        if row.get("split") != "test":
            raise ValueError("Every CT geometry scene must be external test")
        if row.get("material_source") != "synthetic":
            raise ValueError("Material source must be synthetic")
        if row.get("mechanics_source") != "synthetic":
            raise ValueError("Mechanics source must be synthetic")
        if row.get("geometry_source") != "deidentified_ct_mesh":
            raise ValueError("Geometry source must be the de-identified CT mesh")
    counts = {key: len(rows) for key, rows in by_geometry.items()}
    if set(counts.values()) != {20}:
        raise ValueError(f"Expected 20 scenes per geometry, found {counts}")

    scenarios = sorted({str(row["scenario_id"]) for row in patients})
    if len(scenarios) != 20:
        raise ValueError(f"Expected 20 matched scenario IDs, found {len(scenarios)}")
    for scenario_id in scenarios:
        rows = [row for row in patients if row["scenario_id"] == scenario_id]
        if len(rows) != 3:
            raise ValueError(f"{scenario_id} is not present in all geometries")
        reference = rows[0]
        for row in rows[1:]:
            for field in MATCHED_FIELDS:
                if row[field] != reference[field]:
                    raise ValueError(
                        f"{scenario_id}/{field} is not matched across geometries"
                    )

    load_names: Counter[str] = Counter()
    minimum_jacobian = float("inf")
    tensor_count = 0
    for row in patients:
        if len(row["experiments"]) != 4:
            raise ValueError(f"{row['patient_id']} does not contain four loads")
        for experiment_record in row["experiments"]:
            experiment = torch.load(
                dataset_root / experiment_record["relative_path"],
                map_location="cpu",
                weights_only=False,
            )
            tensor_count += 1
            load_names[str(experiment["experiment"])] += 1
            if int(experiment.get("num_views", 0)) != 3:
                raise ValueError("Experiment does not contain three views")
            if tuple(experiment["poses_multiview"].shape[:2]) != (7, 3):
                raise ValueError("Experiment does not contain seven frames by three views")
            if experiment["image_uv_deformed_multiview_seq"].shape[:2] != (7, 3):
                raise ValueError("Multiview trajectory tensor has an invalid protocol")
            if "image_occlusion_confidence_multiview_seq" not in experiment:
                raise ValueError("Occlusion confidence is absent")
            if "forces" not in experiment or "forces_measured" not in experiment:
                raise ValueError("True and measured force fields are both required")
            if str(experiment["geometry_id"]) != str(row["geometry_id"]):
                raise ValueError("Experiment geometry ID differs from its manifest row")
            minimum_jacobian = min(
                minimum_jacobian, float(experiment_record["minimum_jacobian"])
            )

    return {
        "schema_version": 1,
        "dataset_version": payload["version"],
        "geometry_count": len(geometry_ids),
        "scenario_count": len(patients),
        "scenarios_per_geometry": counts,
        "matched_scenario_template_count": len(scenarios),
        "experiment_count": tensor_count,
        "load_name_counts": dict(sorted(load_names.items())),
        "frame_count": 7,
        "view_count": 3,
        "minimum_jacobian": minimum_jacobian,
        "all_geometry_splits_external_test": True,
        "geometry_cross_split_leakage": False,
        "raw_identifiers_or_paths_in_manifest": False,
        "material_truth_present": True,
        "material_truth_interpretation": "synthetic_mechanics_only",
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = validate_dataset(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
