"""Create a privacy-safe geometry-level split with locked external test cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = ROOT / "results" / "ion_geometry_ood" / "deidentified_meshes27"
DEFAULT_LOCKED = (
    ROOT / "results" / "ion_geometry_ood" / "deidentified_meshes"
    / "privacy_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "ion_geometry_ood" / "geometry_split_27.json"
)


def _geometry_ids(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = payload.get("exported_geometry_ids", [])
    return sorted({str(value) for value in values})


def build_split(
    geometry_ids: list[str],
    locked_test_ids: list[str],
    *,
    validation_count: int = 3,
    test_count: int = 6,
    seed: int = 20260730,
) -> dict[str, Any]:
    """Assign every opaque geometry ID exactly once."""

    geometry_ids = sorted(set(geometry_ids))
    locked = sorted(set(locked_test_ids) & set(geometry_ids))
    if test_count < len(locked):
        raise ValueError("test_count is smaller than the locked external test set")
    if validation_count <= 0 or test_count <= 0:
        raise ValueError("validation_count and test_count must be positive")
    if validation_count + test_count >= len(geometry_ids):
        raise ValueError("The requested split leaves no training geometries")
    remaining = sorted(
        set(geometry_ids) - set(locked),
        key=lambda value: hashlib.sha256(
            f"{seed}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    additional_test_count = test_count - len(locked)
    additional_test = remaining[:additional_test_count]
    validation = remaining[
        additional_test_count : additional_test_count + validation_count
    ]
    training = remaining[additional_test_count + validation_count :]
    test = sorted(locked + additional_test)
    assignments = {
        geometry_id: split
        for split, values in (
            ("train", training),
            ("val", validation),
            ("test", test),
        )
        for geometry_id in values
    }
    return {
        "schema_version": 1,
        "split_unit": "deidentified_patient_geometry",
        "seed": seed,
        "geometry_count": len(geometry_ids),
        "counts": {
            "train": len(training),
            "val": len(validation),
            "test": len(test),
        },
        "locked_external_test_geometry_ids": locked,
        "assignments": dict(sorted(assignments.items())),
        "privacy": {
            "raw_paths_persisted": False,
            "source_identifiers_persisted": False,
            "dicom_identifiers_persisted": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--locked-manifest", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-count", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    geometry_ids = _geometry_ids(args.mesh_dir / "privacy_manifest.json")
    locked_ids = _geometry_ids(args.locked_manifest)
    result = build_split(
        geometry_ids,
        locked_ids,
        validation_count=args.validation_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
