"""Audit patient-level integrity of a generated lung AI dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


MULTIVIEW_SCHEMA_PREFIX = "sim_lung_v2_multiview"
MULTIVIEW_FIELDS = (
    "image_uv_rest_multiview_seq",
    "image_uv_deformed_multiview_seq",
    "image_depth_rest_multiview_seq",
    "image_depth_deformed_multiview_seq",
    "image_in_frame_rest_multiview_seq",
    "image_in_frame_deformed_multiview_seq",
    "image_zbuffer_depth_rest_multiview_seq",
    "image_zbuffer_depth_deformed_multiview_seq",
    "image_foreground_confidence_rest_multiview_seq",
    "image_foreground_confidence_deformed_multiview_seq",
)


def validate_experiment_protocol(gt: dict, path: Path | str = "<memory>") -> bool:
    """Validate v2 multiview fields, while accepting historical single-view GT."""
    multiview = "poses_multiview" in gt
    schema = str(gt.get("schema_version", ""))
    if schema.startswith(MULTIVIEW_SCHEMA_PREFIX) and not multiview:
        raise ValueError(f"Multiview schema has no multiview fields in {path}")
    if not multiview:
        # Old schema remains valid and is intentionally not upgraded in place.
        return False

    T, V = 7, 3
    poses = gt["poses_multiview"]
    if tuple(poses.shape) != (T, V, 4, 4) or int(gt.get("num_views", -1)) != V:
        raise ValueError(f"Invalid multiview camera shape in {path}")
    if tuple(gt["intrinsics_multiview"].shape) != (V, 3, 3):
        raise ValueError(f"Invalid multiview intrinsics in {path}")
    expected_frames = torch.arange(T, dtype=torch.long)[:, None].repeat(1, V)
    frame_indices = gt.get("camera_frame_index_multiview")
    if not isinstance(frame_indices, torch.Tensor) or not torch.equal(
        frame_indices, expected_frames
    ):
        raise ValueError(f"Multiview cameras do not share frame timestamps in {path}")
    if not torch.equal(gt["poses"], poses[:, 1]):
        raise ValueError(f"Legacy and multiview cameras are not synchronized in {path}")
    if gt["forces"].shape[0] != T or gt["u_seq"].shape[0] != T:
        raise ValueError(f"Camera, force and deformation sequences are not synchronized in {path}")

    N = len(gt["surface_node_ids"])
    for key in MULTIVIEW_FIELDS:
        if key not in gt:
            raise ValueError(f"Missing {key} in {path}")
        expected = (T, V, N, 2) if "_uv_" in key else (T, V, N)
        if tuple(gt[key].shape) != expected:
            raise ValueError(f"Invalid shape for {key} in {path}")
    gaussian_count = len(gt["image_gaussian_host_tri"])
    for key in MULTIVIEW_FIELDS:
        gaussian_key = key.replace("image_", "image_gaussian_", 1)
        if gaussian_key not in gt:
            raise ValueError(f"Missing {gaussian_key} in {path}")
        expected = (
            (T, V, gaussian_count, 2)
            if "_uv_" in gaussian_key
            else (T, V, gaussian_count)
        )
        if tuple(gt[gaussian_key].shape) != expected:
            raise ValueError(f"Invalid shape for {gaussian_key} in {path}")
    for state in ("rest", "deformed"):
        depth = gt[f"image_depth_{state}_multiview_seq"]
        in_frame = gt[f"image_in_frame_{state}_multiview_seq"]
        zbuffer = gt[f"image_zbuffer_depth_{state}_multiview_seq"]
        confidence = gt[f"image_foreground_confidence_{state}_multiview_seq"]
        if not bool((depth > 0).all()):
            raise ValueError(f"Non-positive camera depth in {path}")
        if not bool((zbuffer[in_frame] > 0).all()):
            raise ValueError(f"Non-positive in-frame z-buffer depth in {path}")
        if not bool(((confidence >= 0) & (confidence <= 1)).all()):
            raise ValueError(f"Foreground confidence outside [0,1] in {path}")
        if not bool((confidence[~in_frame] == 0).all()):
            raise ValueError(f"Out-of-frame points have foreground confidence in {path}")

    seed = int(gt["force_measurement_seed"])
    other_seeds = {
        int(gt.get("motion_noise_seed", -1)),
        int(gt.get("geometry_seed", -2)),
    }
    if seed in other_seeds:
        raise ValueError(f"Force perturbation seed is not independent in {path}")
    generator = torch.Generator().manual_seed(seed)
    expected_error = float(gt["force_measurement_prior_fraction"]) * torch.randn(
        gt["forces"].shape[0], generator=generator, dtype=gt["forces"].dtype
    )
    expected_forces = gt["forces"] * (1.0 + expected_error[:, None])
    if not torch.equal(expected_error, gt["force_measurement_relative_error"]):
        raise ValueError(f"Force perturbation seed is not reproducible in {path}")
    if not torch.equal(expected_forces, gt["forces_measured"]):
        raise ValueError(f"Measured force is not reproducible in {path}")
    if float(gt["force_scale_true_N"]) <= 0:
        raise ValueError(f"Invalid true force scale in {path}")
    return True


def audit(dataset: Path, require_complete: bool = False) -> dict:
    manifest_path = dataset if dataset.is_file() else dataset / "manifest.json"
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["patients"]
    patient_ids = [row["patient_id"] for row in rows]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Duplicate patient IDs")
    if require_complete and len(rows) != int(manifest["patient_count"]):
        raise ValueError("Dataset is only partially generated")
    split_ids: dict[str, set[str]] = {
        split: {row["patient_id"] for row in rows if row["split"] == split}
        for split in ("train", "val", "test")
    }
    if any(
        split_ids[first] & split_ids[second]
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Patient leakage across splits")
    minimum_jacobians, experiment_count, multiview_experiment_count = [], 0, 0
    for row in rows:
        if len(row["experiments"]) != 4:
            raise ValueError(f"{row['patient_id']} does not have four load cases")
        references = []
        for item in row["experiments"]:
            path = root / item["relative_path"]
            if not path.exists():
                raise FileNotFoundError(path)
            gt = torch.load(path, map_location="cpu", weights_only=False)
            if gt["patient_id"] != row["patient_id"] or gt["split"] != row["split"]:
                raise ValueError(f"Metadata mismatch in {path}")
            if gt["forces"].shape[0] != 7 or gt["surface_motion_observed"].shape[0] != 7:
                raise ValueError(f"Invalid temporal length in {path}")
            multiview_experiment_count += int(validate_experiment_protocol(gt, path))
            references.append(gt)
            minimum_jacobians.append(float(item["minimum_jacobian"]))
            experiment_count += 1
        for candidate in references[1:]:
            for key in ("nodes", "elems", "fixed", "surface_node_ids"):
                if not torch.equal(references[0][key], candidate[key]):
                    raise ValueError(f"{row['patient_id']} changes {key} across loads")
            for key in ("E_background", "inclusion_ratio"):
                if float(references[0][key]) != float(candidate[key]):
                    raise ValueError(f"{row['patient_id']} changes {key} across loads")
    split_counts = Counter(row["split"] for row in rows)
    result = {
        "version": manifest["version"],
        "declared_patient_count": int(manifest["patient_count"]),
        "generated_patient_count": len(rows),
        "experiment_count": experiment_count,
        "multiview_experiment_count": multiview_experiment_count,
        "split_counts": dict(split_counts),
        "minimum_jacobian": min(minimum_jacobians) if minimum_jacobians else None,
        "patient_level_no_leakage": True,
        "complete": len(rows) == int(manifest["patient_count"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.dataset, require_complete=args.require_complete)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
