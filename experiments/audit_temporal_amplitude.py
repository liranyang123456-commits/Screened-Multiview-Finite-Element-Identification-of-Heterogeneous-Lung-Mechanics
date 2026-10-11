from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lung_inverse_rendering.evaluate_sim_lung_v2 import load_patient  # noqa: E402

DATASET = ROOT / "dataset" / "sim_lung_ai_v2_multiview250"
OUT = ROOT / "results" / "lung_mesh_ai_v2" / "temporal_amplitude_audit_test50.json"
FRAMES = (1, 2, 3, 4, 5)
ENVELOPES = (0.45, 1.0, 1.0, 0.65, 0.25)


def main() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    patients = [row for row in manifest["patients"] if row["split"] == "test"]
    records = []
    for patient in patients:
        scene, experiments = load_patient(DATASET, patient)
        for load_index, experiment in enumerate(experiments):
            if "image_uv_rest_multiview_seq" not in experiment:
                raise KeyError("Persisted multiview track sequence is required")
            rest = experiment["image_uv_rest_multiview_seq"].to(torch.float64)
            deformed = experiment["image_uv_deformed_multiview_seq"].to(torch.float64)
            confidence = experiment["image_occlusion_confidence_multiview_seq"]
            load_rows = []
            for view in range(rest.shape[1]):
                tracks = torch.stack(
                    [deformed[frame, view] - rest[frame, view] for frame in FRAMES]
                )
                mask = torch.stack(
                    [confidence[frame, view] >= 0.2 for frame in FRAMES]
                ).all(dim=0)
                if not bool(mask.any()):
                    continue
                observed = tracks[:, mask, :]
                weights = torch.tensor(ENVELOPES, dtype=observed.dtype).view(-1, 1, 1)
                coefficient = (weights * observed).sum(dim=0) / (weights.square().sum())
                reconstructed = weights * coefficient.unsqueeze(0)
                # Normalize by observed RMS so errors are comparable across loads/views.
                denom = observed.square().mean().sqrt().clamp_min(1e-12)
                frame_rel = ((reconstructed - observed).square().mean(dim=(1, 2)).sqrt() / denom)
                load_rows.append({
                    "view": view,
                    "common_visible_tracks": int(mask.sum()),
                    "relative_rmse_by_frame": [float(x) for x in frame_rel],
                    "relative_rmse_all_frames": float((reconstructed - observed).square().mean().sqrt() / denom),
                })
            records.append({
                "patient_id": patient["patient_id"],
                "load": patient.get("experiments", [{}] * len(experiments))[load_index].get("name", str(load_index)),
                "views": load_rows,
            })
        print(f"{len(records) // max(len(experiments), 1)}/{len(patients)} patients", flush=True)

    values = [view["relative_rmse_all_frames"] for row in records for view in row["views"]]
    per_frame = [
        [view["relative_rmse_by_frame"][i] for row in records for view in row["views"]]
        for i in range(len(FRAMES))
    ]
    result = {
        "experiment": "temporal_amplitude_rank1_reconstruction_audit",
        "split": "locked synthetic test patients; descriptive only",
        "patients": len(patients),
        "patient_load_records": len(records),
        "frames": list(FRAMES),
        "assumed_amplitude_envelope": list(ENVELOPES),
        "metric": "common-visible multiview image-track RMS reconstruction error divided by observed-track RMS",
        "view_load_observations": len(values),
        "median_relative_rmse_all_frames": float(np.median(values)),
        "q05_median_q95_relative_rmse_all_frames": [float(x) for x in np.quantile(values, [0.05, 0.5, 0.95])],
        "median_relative_rmse_by_frame": [float(np.median(v)) for v in per_frame],
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
