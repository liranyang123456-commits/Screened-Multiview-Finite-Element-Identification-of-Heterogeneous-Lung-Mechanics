"""Create a patient-prefix manifest without duplicating ground-truth tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patients", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = payload["patients"][: args.patients]
    if len(selected) != args.patients:
        raise ValueError(
            f"Requested {args.patients} patients, manifest has {len(payload['patients'])}"
        )
    split_counts = {
        split: sum(patient.get("split") == split for patient in selected)
        for split in ("train", "val", "test")
    }
    output = {
        **payload,
        "version": f"{payload.get('version', 'dataset')}_n{args.patients}",
        "patient_count": args.patients,
        "generated_patient_count": args.patients,
        "experiment_count": sum(len(row.get("experiments", [])) for row in selected),
        "patients": selected,
        "subset": {
            "source_manifest": str(args.manifest.resolve()),
            "patient_prefix_count": args.patients,
            "split_counts": split_counts,
        },
    }
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["subset"], indent=2))


if __name__ == "__main__":
    main()
