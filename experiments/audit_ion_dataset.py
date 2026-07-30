"""Read-only, de-identified audit for the local ION patient collection.

The source directory is never modified.  Outputs contain only ``case_XXX``
identifiers; raw directory names and DICOM patient tags are deliberately never
persisted or printed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.ion_dicom_sources import scan_case_dicoms  # noqa: E402


DEFAULT_ROOT = Path(r"external_data/ion_ct")
OUTPUT = ROOT / "results" / "ion_audit"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DICOM_EXTENSIONS = {".dcm", ".ima"}


def looks_like_dicom(path: Path) -> bool:
    """Test the standard preamble without reading a full image."""
    if path.suffix.lower() in DICOM_EXTENSIONS:
        return True
    try:
        with path.open("rb") as stream:
            stream.seek(128)
            return stream.read(4) == b"DICM"
    except OSError:
        return False


def video_summary(paths: list[Path]) -> dict[str, object]:
    rows = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        rows.append(
            {
                "frames": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
            }
        )
    return {"count": len(rows), "files": rows}


def archive_summary(paths: list[Path]) -> dict[str, object]:
    """Inventory archive members without extracting or retaining filenames."""
    member_extensions: Counter[str] = Counter()
    readable = 0
    has_video = 0
    has_dicom = 0
    for path in paths:
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        except (OSError, zipfile.BadZipFile):
            continue
        readable += 1
        extensions = [Path(name).suffix.lower() for name in names if not name.endswith("/")]
        member_extensions.update(extension or "[no_extension]" for extension in extensions)
        has_video += any(extension in VIDEO_EXTENSIONS for extension in extensions)
        has_dicom += any(extension in DICOM_EXTENSIONS for extension in extensions)
    return {
        "archive_count": len(paths),
        "readable_archives": readable,
        "archives_with_video_member": has_video,
        "archives_with_dicom_extension_member": has_dicom,
        "member_extension_counts": dict(sorted(member_extensions.items())),
    }


def audit_case(case_dir: Path, case_id: str, dicom_limit: int) -> dict[str, object]:
    extension_counts: Counter[str] = Counter()
    files: list[Path] = []
    for root, _, names in os.walk(case_dir):
        files.extend(Path(root) / name for name in names)
    for path in files:
        extension_counts[path.suffix.lower() or "[no_extension]"] += 1

    categories = {path.parts[len(case_dir.parts)] for path in files if len(path.parts) > len(case_dir.parts)}
    video_files = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    archive_files = [path for path in files if path.suffix.lower() == ".zip"]
    dicom_inventory = scan_case_dicoms(case_dir)
    category_tokens = " ".join(categories)
    return {
        "case_id": case_id,
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "top_level_category_count": len(categories),
        "has_intraoperative_video_category": "术中视频" in category_tokens,
        "has_planning_screenshot_category": "规划截图" in category_tokens,
        "has_pathology_category": "病理结果" in category_tokens,
        "has_medical_record_category": "病历" in category_tokens,
        "extension_counts": dict(sorted(extension_counts.items())),
        "images": {"count": len(image_files)},
        "videos": video_summary(video_files),
        "archives": archive_summary(archive_files),
        "dicom": {
            "candidate_files": dicom_inventory.candidate_object_count,
            "headers_read": dicom_inventory.candidate_object_count,
            "modalities": dicom_inventory.modality_counts,
            "ct_object_count": dicom_inventory.ct_object_count,
            "ct_unique_object_count": dicom_inventory.ct_unique_object_count,
            "ct_series_count": len(dicom_inventory.ct_series),
            "largest_ct_series_size": (
                len(dicom_inventory.ct_series[0])
                if dicom_inventory.ct_series
                else 0
            ),
            "unreadable_archive_count": dicom_inventory.unreadable_archive_count,
            "unreadable_member_count": dicom_inventory.unreadable_member_count,
        },
    }


def make_patient_splits(cases: list[dict], seed: int) -> dict[str, list[str]]:
    """Deterministic 18/6/6 patient-level split; no patient can cross splits."""
    eligible = [
        case["case_id"]
        for case in cases
        if case["videos"]["count"] > 0
        or case["archives"]["archives_with_video_member"] > 0
    ]
    if len(eligible) < 10:
        return {
            "status": "not_generated_insufficient_video_cases",
            "candidate_video_cases": sorted(eligible),
            "minimum_required_for_60_20_20": 10,
        }
    rng = random.Random(seed)
    rng.shuffle(eligible)
    train_count = round(len(eligible) * 0.60)
    val_count = round(len(eligible) * 0.20)
    return {
        "status": "candidate_split_requires_deidentification_before_use",
        "train": sorted(eligible[:train_count]),
        "val": sorted(eligible[train_count : train_count + val_count]),
        "test": sorted(eligible[train_count + val_count :]),
        "excluded_no_video": sorted(
            case["case_id"]
            for case in cases
            if case["videos"]["count"] == 0
            and case["archives"]["archives_with_video_member"] == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dicom-header-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    case_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
    cases = [
        audit_case(case_dir, f"case_{index:03d}", args.dicom_header_limit)
        for index, case_dir in enumerate(case_dirs, start=1)
    ]
    splits = make_patient_splits(cases, args.seed)
    result = {
        "source_modified": False,
        "raw_identifiers_persisted": False,
        "patient_count": len(cases),
        "source_root": "local ION collection (path intentionally omitted)",
        "case_records": cases,
        "patient_level_video_split": splits,
        "limitations": [
            "No DICOM patient-identifying tags were read or retained.",
            "Pathology content was not parsed; only directory-level presence was audited.",
            "This audit does not establish ethics approval, consent, or legal data-use authority.",
            "No Young's-modulus, force, or indentation ground truth is inferred.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# De-identified ION dataset audit",
        "",
        f"- Patients: {len(cases)}",
        f"- Cases with directly readable video: {sum(case['videos']['count'] > 0 for case in cases)}",
        f"- Cases with video archived in ZIP: {sum(case['archives']['archives_with_video_member'] > 0 for case in cases)}",
        f"- Cases with readable CT: {sum(case['dicom']['ct_unique_object_count'] > 0 for case in cases)}",
        f"- Cases with a CT series of at least 100 unique slices: "
        f"{sum(case['dicom']['largest_ct_series_size'] >= 100 for case in cases)}",
        f"- Cases with pathology directory: {sum(case['has_pathology_category'] for case in cases)}",
        f"- Cases with planning screenshots: {sum(case['has_planning_screenshot_category'] for case in cases)}",
    ]
    if splits["status"].startswith("candidate_split"):
        lines.append(
            f"- Candidate patient-level video split: {len(splits['train'])}/"
            f"{len(splits['val'])}/{len(splits['test'])}; excluded without video: "
            f"{len(splits['excluded_no_video'])}."
        )
    else:
        lines.append(
            f"- Patient-level split withheld: only {len(splits['candidate_video_cases'])} "
            "candidate video cases were found."
        )
    lines.extend(
        [
        "",
        "The audit is metadata-only: it neither copies nor changes source data and "
        "does not expose raw patient identifiers or DICOM patient tags.",
        ]
    )
    (OUTPUT / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
