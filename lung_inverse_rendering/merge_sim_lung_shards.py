"""Merge independently generated sim_lung shards into one dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifests = [
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for root in args.shards
    ]
    reference = manifests[0]
    for manifest in manifests[1:]:
        for key in ("version", "patient_count", "generation_config"):
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"Shard mismatch for {key}")
    rows: dict[str, dict] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for root, manifest in zip(args.shards, manifests):
        for row in manifest["patients"]:
            patient_id = row["patient_id"]
            if patient_id in rows:
                raise ValueError(f"Duplicate patient across shards: {patient_id}")
            source = root / patient_id
            destination = args.out / patient_id
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copytree(source, destination)
            rows[patient_id] = row
    patients = [rows[key] for key in sorted(rows)]
    merged = {
        **reference,
        "generated_patient_count": len(patients),
        "experiment_count": sum(len(row["experiments"]) for row in patients),
        "patients": patients,
        "merged_from": [str(path) for path in args.shards],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    print(f"Merged {len(patients)} patients into {args.out}")


if __name__ == "__main__":
    main()
