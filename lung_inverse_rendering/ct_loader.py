"""CT volume to de-identified local-lung mesh preprocessing.

This utility reads pixels only from a user-authorized CT directory.  It does
not copy DICOM headers, raw paths, UIDs, names, dates, or source filenames into
its output.  The emitted mesh remains sensitive patient-derived data and must
stay in the approved storage boundary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_ct_hu(dicom_dir: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load one CT series into HU volume ordered as (z, y, x)."""
    import pydicom

    records = []
    for path in dicom_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, force=False)
        except Exception:
            continue
        if str(getattr(dataset, "Modality", "")) != "CT" or "PixelData" not in dataset:
            continue
        position = getattr(dataset, "ImagePositionPatient", None)
        z = float(position[2]) if position is not None and len(position) >= 3 else float(
            getattr(dataset, "InstanceNumber", len(records))
        )
        records.append((z, dataset))
    if len(records) < 8:
        raise ValueError("Expected at least 8 readable CT slices")
    records.sort(key=lambda item: item[0])
    first = records[0][1]
    spacing_yx = tuple(float(value) for value in first.PixelSpacing)
    z_values = np.asarray([row[0] for row in records], dtype=np.float64)
    spacing_z = float(np.median(np.diff(z_values))) if len(z_values) > 1 else float(
        getattr(first, "SliceThickness", 1.0)
    )
    slices = []
    for _, dataset in records:
        pixels = dataset.pixel_array.astype(np.float32)
        pixels *= float(getattr(dataset, "RescaleSlope", 1.0))
        pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
        slices.append(pixels)
    return np.stack(slices), (abs(spacing_z), spacing_yx[0], spacing_yx[1])


def lung_air_mask(volume_hu: np.ndarray) -> np.ndarray:
    """Return a conservative lung-air candidate mask from a CT HU volume.

    This is a preprocessing baseline, not a clinically validated segmentation.
    An approved lobe/airway segmentation should replace it for any study claim.
    """
    try:
        from scipy import ndimage
    except ImportError as error:
        raise RuntimeError("CT mask extraction requires scipy") from error
    candidate = (volume_hu > -1000) & (volume_hu < -320)
    labels, count = ndimage.label(candidate)
    if count == 0:
        raise ValueError("No candidate lung-air region found in HU range")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    # Exterior/background air is usually the largest low-HU component.  Remove
    # components connected to the lateral image borders before ranking internal
    # air regions.  The superior/inferior z borders are intentionally excluded
    # because a thoracic series can truncate a lung at either end.
    border_labels = np.unique(
        np.concatenate(
            [
                labels[:, 0, :].ravel(),
                labels[:, -1, :].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
    )
    sizes[border_labels] = 0
    positive = np.flatnonzero(sizes > 0)
    # Suppress posterior scanner-table/gantry air pockets that can be enclosed
    # by padding and therefore escape the lateral-border test.
    for label, center in zip(
        positive, ndimage.center_of_mass(candidate, labels, positive)
    ):
        normalized_row = float(center[1]) / max(volume_hu.shape[1] - 1, 1)
        if normalized_row > 0.82:
            sizes[label] = 0
    positive = np.flatnonzero(sizes > 0)
    if len(positive) == 0:
        raise ValueError("No internal lung-air region remains after border removal")
    keep = positive[np.argsort(sizes[positive])[-2:]]
    mask = np.isin(labels, keep)
    return ndimage.binary_closing(mask, iterations=2)


def mask_to_surface(
    mask: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an isosurface in physical spacing coordinates."""
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise RuntimeError("Mesh extraction requires scikit-image") from error
    vertices, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=spacing_zyx
    )
    # marching cubes supplies z,y,x; FEM/rendering convention uses x,y,z.
    return vertices[:, [2, 1, 0]].astype(np.float64), faces.astype(np.int64)


def export_deidentified_mesh(
    dicom_dir: Path, output_path: Path, *, geometry_id: str
) -> None:
    """Write vertices/faces and a user-provided opaque geometry ID only."""
    volume, spacing = load_ct_hu(dicom_dir)
    vertices, faces = mask_to_surface(lung_air_mask(volume), spacing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vertices=vertices,
        faces=faces,
        geometry_id=np.asarray(geometry_id),
        preprocessing=np.asarray("HU_threshold_baseline_not_clinical_segmentation"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geometry-id", required=True)
    args = parser.parse_args()
    export_deidentified_mesh(args.dicom_dir, args.out, geometry_id=args.geometry_id)
    print(f"Wrote de-identified mesh: {args.out.name}")


if __name__ == "__main__":
    main()
