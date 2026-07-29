"""Create leakage-safe 40/100-patient views of an interleaved full dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--counts", nargs="+", type=int, default=[40, 100])
    args = parser.parse_args()
    manifest = json.loads(
        (args.dataset / "manifest.json").read_text(encoding="utf-8")
    )
    patients = sorted(manifest["patients"], key=lambda row: row["patient_id"])
    for count in args.counts:
        if not 5 <= count <= len(patients):
            raise ValueError(f"Invalid stage patient count: {count}")
        selected = patients[:count]
        stage = {
            **manifest,
            "version": f"{manifest['version']}_stage{count}",
            "patient_count": count,
            "generated_patient_count": count,
            "experiment_count": sum(len(row["experiments"]) for row in selected),
            "stage_view_of": manifest["version"],
            "patients": selected,
        }
        path = args.dataset / f"manifest_{count}.json"
        path.write_text(json.dumps(stage, indent=2), encoding="utf-8")
        split_counts = {
            split: sum(row["split"] == split for row in selected)
            for split in ("train", "val", "test")
        }
        print(f"{path}: {split_counts}")


if __name__ == "__main__":
    main()
