"""Export privacy-minimized CT figure assets for internal manuscript review.

Only windowed pixels, a heuristic HU mask, and an overlay are written as PNG.
No DICOM object, tag, UID, source path, patient name, or case identifier is
persisted. Manual visual QC and externally managed publication authorization
are recorded explicitly in the privacy manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.ct_loader import lung_air_mask  # noqa: E402
from lung_inverse_rendering.prepare_ion_ct_meshes import (
    DEFAULT_AUDIT,
    DEFAULT_SOURCE,
    _discover_best_lung_ct_series,
    _load_series_hu,
    _safe_output_dir,
    select_ct_candidates,
)  # noqa: E402
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "ion_ct_synthetic_mechanics"
    / "deidentified_figure_assets"
)


def window_to_uint8(
    image_hu: np.ndarray, *, lower: float = -1000.0, upper: float = 200.0
) -> np.ndarray:
    """Window one HU image without retaining acquisition metadata."""
    if upper <= lower:
        raise ValueError("Window upper bound must exceed lower bound")
    clipped = np.clip(np.asarray(image_hu, dtype=np.float32), lower, upper)
    return np.rint(255.0 * (clipped - lower) / (upper - lower)).astype(np.uint8)


def crop_to_mask(
    image: np.ndarray, mask: np.ndarray, *, margin_fraction: float = 0.08
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the mask bounding box, removing scanner-frame borders."""
    if image.shape != mask.shape or image.ndim != 2:
        raise ValueError("Image and mask must be same-shape 2-D arrays")
    rows, columns = np.where(mask)
    if not len(rows):
        raise ValueError("Representative mask slice is empty")
    height, width = image.shape
    margin = max(4, int(round(max(height, width) * margin_fraction)))
    y0 = max(0, int(rows.min()) - margin)
    y1 = min(height, int(rows.max()) + margin + 1)
    x0 = max(0, int(columns.min()) - margin)
    x1 = min(width, int(columns.max()) + margin + 1)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def representative_slice(volume_hu: np.ndarray, mask: np.ndarray) -> int:
    """Choose the slice with the largest segmented cross-section."""
    if volume_hu.shape != mask.shape or volume_hu.ndim != 3:
        raise ValueError("Volume and mask must be same-shape 3-D arrays")
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    if int(areas.max()) == 0:
        raise ValueError("Volume mask is empty")
    return int(np.argmax(areas))


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create an RGB overlay with a color-blind-safe orange mask."""
    rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    orange = np.asarray([213.0, 94.0, 0.0], dtype=np.float32)
    rgb[mask] = 0.55 * rgb[mask] + 0.45 * orange
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _png_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _burned_in_annotation_status(paths: list[Path]) -> str:
    """Read only the standard annotation flag transiently; persist no tags."""
    import pydicom

    observed: set[str] = set()
    for path in paths[:: max(1, len(paths) // 8)]:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=False,
            specific_tags=["BurnedInAnnotation"],
        )
        value = str(getattr(dataset, "BurnedInAnnotation", "")).strip().upper()
        if value:
            observed.add(value)
    if "YES" in observed:
        return "declared_yes"
    if observed == {"NO"}:
        return "declared_no"
    return "unknown_manual_qc_required"


def _write_case_assets(
    case_dir: Path,
    geometry_id: str,
    output_dir: Path,
    *,
    manual_visual_qc_complete: bool,
    publication_authorization_user_managed: bool,
) -> dict[str, Any]:
    paths = _discover_best_lung_ct_series(case_dir)
    annotation_status = _burned_in_annotation_status(paths)
    if annotation_status == "declared_yes":
        raise ValueError("DICOM declares burned-in annotation; export refused")
    volume, _ = _load_series_hu(paths)
    mask = lung_air_mask(volume)
    index = representative_slice(volume, mask)
    image = window_to_uint8(volume[index])
    image, mask_slice = crop_to_mask(image, mask[index])
    mask_u8 = np.where(mask_slice, 255, 0).astype(np.uint8)
    assets = {
        "ct": output_dir / f"{geometry_id}_ct_window.png",
        "mask": output_dir / f"{geometry_id}_hu_mask.png",
        "overlay": output_dir / f"{geometry_id}_ct_mask_overlay.png",
    }
    Image.fromarray(image, mode="L").save(assets["ct"])
    Image.fromarray(mask_u8, mode="L").save(assets["mask"])
    Image.fromarray(overlay_mask(image, mask_slice), mode="RGB").save(
        assets["overlay"]
    )
    return {
        "geometry_id": geometry_id,
        "files": {name: path.name for name, path in assets.items()},
        "sha256": {name: _png_sha256(path) for name, path in assets.items()},
        "image_shape": list(image.shape),
        "preprocessing": "lung_window_and_HU_threshold_baseline_not_clinical_segmentation",
        "burned_in_annotation_tag": annotation_status,
        "manual_visual_qc": (
            "passed_no_visible_identifiers"
            if manual_visual_qc_complete
            else "pending"
        ),
        "publication_authorization": (
            "externally_managed_by_authors"
            if publication_authorization_user_managed
            else "pending_institutional_confirmation"
        ),
    }


def export_assets(
    audit_path: Path,
    source_root: Path,
    output_dir: Path,
    *,
    export: bool,
    manual_visual_qc_complete: bool = False,
    publication_authorization_user_managed: bool = False,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidates = select_ct_candidates(audit)
    safe_output = _safe_output_dir(output_dir)
    safe_output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if export:
        if not source_root.is_dir():
            raise FileNotFoundError("Authorized ION source collection is unavailable")
        case_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
        for candidate in candidates:
            try:
                case_number = int(candidate["_case_id"].split("_")[-1])
                records.append(
                    _write_case_assets(
                        case_dirs[case_number - 1],
                        candidate["geometry_id"],
                        safe_output,
                        manual_visual_qc_complete=manual_visual_qc_complete,
                        publication_authorization_user_managed=(
                            publication_authorization_user_managed
                        ),
                    )
                )
            except Exception as error:
                failure = {
                    "geometry_id": candidate["geometry_id"],
                    "error_type": type(error).__name__,
                }
                if isinstance(error, ModuleNotFoundError):
                    failure["missing_module"] = str(error.name)
                failures.append(failure)
    manifest = {
        "schema_version": 1,
        "mode": "export" if export else "dry_run",
        "geometry_count": len(candidates),
        "records": records,
        "failures": failures,
        "privacy": {
            "dicom_objects_written": False,
            "dicom_tags_persisted": False,
            "raw_paths_persisted": False,
            "uids_persisted": False,
            "names_persisted": False,
            "png_metadata_written": False,
            "cropped_to_mask_region": True,
            "patient_derived_pixels_present": export,
            "manual_visual_qc_complete": manual_visual_qc_complete,
            "publication_authorization_pipeline_gate": (
                not publication_authorization_user_managed
            ),
        },
        "evidence_boundary": (
            "Patient-derived CT pixels and heuristic segmentation for visualization "
            "only; no patient material ground truth or clinical segmentation claim."
        ),
    }
    (safe_output / "privacy_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    checklist = [
        "# Manual privacy and publication QC",
        "",
        "Record the completed checks for each patient-derived asset.",
        "",
    ]
    for candidate in candidates:
        geometry_id = candidate["geometry_id"]
        checklist.extend(
            [
                f"## {geometry_id}",
                (
                    "- [x] No visible burned-in name, identifier, date, "
                    "accession number, or UI overlay"
                    if manual_visual_qc_complete
                    else "- [ ] No burned-in name, identifier, date, accession number, or UI overlay"
                ),
                (
                    "- [x] Crop contains only the intended anatomy"
                    if manual_visual_qc_complete
                    else "- [ ] Crop contains only the intended anatomy"
                ),
                "- [x] HU mask is labeled as a non-clinical preprocessing baseline",
                (
                    "- [x] Publication authorization is managed externally by the authors"
                    if publication_authorization_user_managed
                    else "- [ ] Institutional authorization covers publication of this image"
                ),
                "",
            ]
        )
    (safe_output / "MANUAL_QC.md").write_text("\n".join(checklist), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Explicitly read authorized source pixels and write PNG assets",
    )
    parser.add_argument(
        "--manual-visual-qc-complete",
        action="store_true",
        help="Record that exported PNGs were visually checked for identifiers",
    )
    parser.add_argument(
        "--publication-authorization-user-managed",
        action="store_true",
        help="Record publication authorization as externally managed, not a pipeline gate",
    )
    args = parser.parse_args()
    result = export_assets(
        args.audit_manifest,
        args.source_root,
        args.out_dir,
        export=args.export,
        manual_visual_qc_complete=args.manual_visual_qc_complete,
        publication_authorization_user_managed=(
            args.publication_authorization_user_managed
        ),
    )
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "geometry_count": result["geometry_count"],
                "exported_count": len(result["records"]),
                "failure_count": len(result["failures"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
