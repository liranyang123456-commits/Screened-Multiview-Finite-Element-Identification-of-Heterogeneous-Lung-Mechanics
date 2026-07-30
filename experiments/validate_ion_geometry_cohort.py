"""Validate de-identified CT surfaces and their derived finite-element meshes."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_lung_geometry_ood import (  # noqa: E402
    fem_mesh_qc,
    surface_mesh_qc,
)
from lung_inverse_rendering.ct_geometry import build_scene_from_ct_mesh  # noqa: E402


DEFAULT_MESH_DIR = ROOT / "results" / "ion_geometry_ood" / "deidentified_meshes27"
DEFAULT_OUTPUT = ROOT / "results" / "ion_geometry_ood" / "geometry_qc_27.json"
FORBIDDEN = re.compile(r"(?:[A-Za-z]:\\|省医|case_\d|SeriesInstanceUID)")


def validate_cohort(mesh_dir: Path) -> dict[str, Any]:
    privacy = json.loads(
        (mesh_dir / "privacy_manifest.json").read_text(encoding="utf-8")
    )
    geometry_ids = [str(value) for value in privacy["exported_geometry_ids"]]
    records = []
    for geometry_id in geometry_ids:
        mesh_path = mesh_dir / f"{geometry_id}.npz"
        try:
            with np.load(mesh_path) as mesh:
                persisted_id = str(mesh["geometry_id"].item())
                surface = surface_mesh_qc(mesh["vertices"], mesh["faces"])
            if persisted_id != geometry_id:
                raise ValueError("Persisted opaque geometry ID differs from filename")
            scene = build_scene_from_ct_mesh(
                mesh_path,
                geometry_id=geometry_id,
                E_true=5_000.0,
            )
            finite_element = fem_mesh_qc(scene)
            passed = bool(
                surface["valid"]
                and finite_element["construction_success"]
                and finite_element["finite_nodes"]
                and finite_element["rest_jacobian_positive_magnitude"]
                and finite_element["minimum_deformation_jacobian"] > 0
            )
            records.append(
                {
                    "geometry_id": geometry_id,
                    "passed": passed,
                    "surface": surface,
                    "finite_element": finite_element,
                }
            )
        except Exception as error:
            records.append(
                {
                    "geometry_id": geometry_id,
                    "passed": False,
                    "failure_type": type(error).__name__,
                }
            )
    vertex_counts = [
        record["surface"]["vertex_count"]
        for record in records
        if "surface" in record
    ]
    result = {
        "schema_version": 1,
        "geometry_count": len(geometry_ids),
        "passed_count": sum(bool(record["passed"]) for record in records),
        "failed_count": sum(not bool(record["passed"]) for record in records),
        "surface_vertex_count": {
            "minimum": int(np.min(vertex_counts)) if vertex_counts else 0,
            "median": float(np.median(vertex_counts)) if vertex_counts else 0.0,
            "maximum": int(np.max(vertex_counts)) if vertex_counts else 0,
        },
        "records": records,
        "privacy": {
            "raw_paths_persisted": False,
            "source_identifiers_persisted": False,
            "dicom_identifiers_persisted": False,
        },
    }
    if FORBIDDEN.search(json.dumps(result)):
        raise AssertionError("Geometry QC output contains a forbidden identifier")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = validate_cohort(args.mesh_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "geometry_count": result["geometry_count"],
                "passed_count": result["passed_count"],
                "failed_count": result["failed_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
