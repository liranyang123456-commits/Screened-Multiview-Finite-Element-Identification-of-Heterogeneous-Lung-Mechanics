"""Aggregate geometry-only OOD stability without material-accuracy claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    geometry_ids = [row["geometry_id"] for row in records]
    if len(geometry_ids) != len(set(geometry_ids)):
        raise ValueError("Duplicate geometry IDs in OOD aggregate")
    construction = [
        bool(row["fem_mesh_qc"]["construction_success"]) for row in records
    ]
    stable = [
        bool(row["model_output_stability"].get("finite_outputs", False))
        and bool(
            row["model_output_stability"].get(
                "within_declared_output_ranges", False
            )
        )
        for row in records
    ]
    jacobians = [
        float(row["fem_mesh_qc"]["minimum_deformation_jacobian"])
        for row in records
    ]
    result = {
        "evidence_scope": "geometry_domain_stability_only",
        "geometry_count": len(records),
        "fem_construction_success_rate": float(np.mean(construction)),
        "geometry_forward_success_rate": float(
            np.mean([first and second for first, second in zip(construction, stable)])
        ),
        "minimum_deformation_jacobian": float(min(jacobians)),
        "material_ground_truth_metrics_reported": False,
        "geometry_ids": geometry_ids,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
