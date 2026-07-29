"""Prepare de-identified ION CT meshes without exposing source identifiers.

The default mode reads only the existing de-identified audit manifest.  Pixel
data is read only with ``--export`` and outputs are restricted to a
de-identified directory inside this workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.ct_loader import lung_air_mask, mask_to_surface  # noqa: E402


DEFAULT_AUDIT = ROOT / "results" / "ion_audit" / "audit_manifest.json"
DEFAULT_OUTPUT = ROOT / "results" / "ion_geometry_ood" / "deidentified_meshes"
DEFAULT_SOURCE = Path(r"external_data/ion_ct")
ID_NAMESPACE = "ion-ct-geometry-ood-v1"


def opaque_geometry_id(case_id: str) -> str:
    """Create a stable identifier that does not reproduce the audit case ID."""
    digest = hashlib.sha256(f"{ID_NAMESPACE}:{case_id}".encode("utf-8")).hexdigest()
    return f"geom_{digest[:20]}"


def select_ct_candidates(audit: dict[str, Any]) -> list[dict[str, str]]:
    """Select cases whose audit sample contains at least one CT object."""
    selected = []
    for case in audit.get("case_records", []):
        dicom = case.get("dicom", {})
        modalities = dicom.get("modalities", {})
        if int(dicom.get("candidate_files", 0)) > 0 and int(modalities.get("CT", 0)) > 0:
            selected.append(
                {
                    "_case_id": str(case["case_id"]),
                    "geometry_id": opaque_geometry_id(str(case["case_id"])),
                }
            )
    return selected


def _safe_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Export directory must be inside the workspace") from error
    if not any("deidentified" in part.lower() for part in resolved.parts):
        raise ValueError("Export directory must be explicitly named as deidentified")
    return resolved


def _discover_largest_ct_series(case_dir: Path) -> list[Path]:
    """Find one CT series internally; paths and DICOM values are never returned."""
    import pydicom

    grouped: dict[str, list[Path]] = defaultdict(list)
    for root, _, names in os.walk(case_dir):
        for name in names:
            path = Path(root) / name
            try:
                dataset = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=False,
                    specific_tags=["Modality", "SeriesInstanceUID"],
                )
            except Exception:
                continue
            if str(getattr(dataset, "Modality", "")) != "CT":
                continue
            # The UID is used transiently as an in-memory grouping key only.
            series_key = str(getattr(dataset, "SeriesInstanceUID", ""))
            if series_key:
                grouped[series_key].append(path)
    if not grouped:
        raise ValueError("No readable CT series found for selected geometry")
    return max(grouped.values(), key=len)


def _discover_ct_series(case_dir: Path) -> list[list[Path]]:
    """Return CT series sorted by size; identifiers remain transient in memory."""
    import pydicom

    grouped: dict[str, list[Path]] = defaultdict(list)
    for root, _, names in os.walk(case_dir):
        for name in names:
            path = Path(root) / name
            try:
                dataset = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=False,
                    specific_tags=["Modality", "SeriesInstanceUID"],
                )
            except Exception:
                continue
            if str(getattr(dataset, "Modality", "")) != "CT":
                continue
            series_key = str(getattr(dataset, "SeriesInstanceUID", ""))
            if series_key:
                grouped[series_key].append(path)
    return sorted(grouped.values(), key=len, reverse=True)


def _load_series_hu(paths: list[Path]) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read pixels from one internally selected series without retaining tags."""
    import pydicom

    records = []
    for path in paths:
        dataset = pydicom.dcmread(path, force=False)
        if "PixelData" not in dataset:
            continue
        position = getattr(dataset, "ImagePositionPatient", None)
        z = float(position[2]) if position is not None and len(position) >= 3 else float(
            getattr(dataset, "InstanceNumber", len(records))
        )
        records.append((z, dataset))
    if len(records) < 8:
        raise ValueError("Selected CT series has fewer than eight readable slices")
    records.sort(key=lambda item: item[0])
    first = records[0][1]
    spacing_yx = tuple(float(value) for value in first.PixelSpacing)
    z_values = np.asarray([item[0] for item in records], dtype=np.float64)
    spacing_z = abs(float(np.median(np.diff(z_values))))
    slices = []
    for _, dataset in records:
        pixels = dataset.pixel_array.astype(np.float32)
        pixels *= float(getattr(dataset, "RescaleSlope", 1.0))
        pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
        slices.append(pixels)
    return np.stack(slices), (spacing_z, spacing_yx[0], spacing_yx[1])


def _bilateral_lung_score(mask: np.ndarray) -> float:
    """Score axial evidence for two separated, substantial lung-air regions."""
    from scipy import ndimage

    if mask.ndim != 3:
        raise ValueError("Expected a 3-D candidate lung mask")
    height, width = mask.shape[1:]
    minimum_area = 0.01 * height * width
    best = 0.0
    for image in mask:
        labels, count = ndimage.label(image)
        if count < 2:
            continue
        sizes = np.bincount(labels.ravel())
        candidates: list[tuple[float, float]] = []
        for label in range(1, len(sizes)):
            area = float(sizes[label])
            if area < minimum_area:
                continue
            _, x = ndimage.center_of_mass(image, labels, label)
            candidates.append((area, float(x) / max(width - 1, 1)))
        left = [item for item in candidates if item[1] < 0.48]
        right = [item for item in candidates if item[1] > 0.52]
        if not left or not right:
            continue
        left_area = max(left)[0]
        right_area = max(right)[0]
        balance = min(left_area, right_area) / max(left_area, right_area)
        coverage = (left_area + right_area) / (height * width)
        best = max(best, coverage * balance)
    return float(best)


def _discover_best_lung_ct_series(case_dir: Path) -> list[Path]:
    """Select a thoracic series using pixels without persisting source metadata."""
    candidates = [paths for paths in _discover_ct_series(case_dir) if len(paths) >= 8]
    if not candidates:
        raise ValueError("No readable CT series found for selected geometry")
    best_paths: list[Path] | None = None
    best_score = 0.0
    # Large diagnostic series are most plausible and limit repeated pixel reads.
    for paths in candidates[:10]:
        try:
            volume, _ = _load_series_hu(paths)
            score = _bilateral_lung_score(lung_air_mask(volume))
        except Exception:
            continue
        if score > best_score:
            best_paths, best_score = paths, score
    if best_paths is None or best_score <= 0.01:
        raise ValueError("No CT series with bilateral thoracic lung-air evidence")
    return best_paths


def _export_case_mesh(case_dir: Path, output: Path, geometry_id: str) -> None:
    paths = _discover_best_lung_ct_series(case_dir)
    volume, spacing = _load_series_hu(paths)
    vertices, faces = mask_to_surface(lung_air_mask(volume), spacing)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=vertices,
        faces=faces,
        geometry_id=np.asarray(geometry_id),
        preprocessing=np.asarray("HU_threshold_baseline_not_clinical_segmentation"),
    )


def prepare(
    audit_path: Path,
    output_dir: Path,
    *,
    export: bool = False,
    source_root: Path = DEFAULT_SOURCE,
) -> dict[str, Any]:
    """Prepare candidates; dry-run never accesses ``source_root``."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidates = select_ct_candidates(audit)
    safe_output = _safe_output_dir(output_dir)
    exported: list[str] = []
    failures: list[dict[str, str]] = []

    if export:
        if not source_root.is_dir():
            raise FileNotFoundError("Authorized ION source collection is unavailable")
        safe_output.mkdir(parents=True, exist_ok=True)
        case_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
        for candidate in candidates:
            try:
                case_number = int(candidate["_case_id"].split("_")[-1])
                case_dir = case_dirs[case_number - 1]
                destination = safe_output / f"{candidate['geometry_id']}.npz"
                _export_case_mesh(case_dir, destination, candidate["geometry_id"])
                exported.append(candidate["geometry_id"])
            except Exception as error:
                failures.append(
                    {
                        "geometry_id": candidate["geometry_id"],
                        "error_type": type(error).__name__,
                    }
                )

    public_candidates = [{"geometry_id": item["geometry_id"]} for item in candidates]
    manifest = {
        "schema_version": 1,
        "mode": "export" if export else "dry_run",
        "source_accessed": export,
        "pixels_read": export,
        "source_modified": False,
        "candidate_count": len(public_candidates),
        "candidates": public_candidates,
        "exported_geometry_ids": exported,
        "failures": failures,
        "privacy": {
            "raw_paths_persisted": False,
            "uids_persisted": False,
            "names_persisted": False,
            "dicom_tags_persisted": False,
            "outputs_contain_patient_derived_geometry": export,
        },
        "limitations": [
            "Geometry-only external validation; no material or force ground truth.",
            "HU thresholding is a preprocessing baseline, not clinical segmentation.",
        ],
    }
    safe_output.mkdir(parents=True, exist_ok=True)
    (safe_output / "privacy_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    result = prepare(args.audit_manifest, args.out_dir, export=args.export)
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "candidate_count": result["candidate_count"],
                "exported_count": len(result["exported_geometry_ids"]),
                "failure_count": len(result["failures"]),
                "privacy_manifest": "privacy_manifest.json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
