"""Build a privacy-preserving, frame-level ION video candidate dataset.

The source collection is read-only.  The output contains no source path,
patient name, DICOM header, audio, or raw video.  Frames are center-cropped
then padded-edge redacted.  Every case remains marked ``needs_visual_qc``:
fixed crop/redaction reduces the risk of burned-in text but cannot replace
human de-identification review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np


SOURCE = Path(r"external_data/ion_ct")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dataset" / "ion_bronchoscopy_candidate"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def safe_case_id(index: int) -> str:
    return f"ion_case_{index:03d}"


def discover_videos(case_dir: Path) -> list[tuple[str, Path, str | None]]:
    """Return direct videos and ZIP members without emitting source filenames."""
    records: list[tuple[str, Path, str | None]] = []
    for root, _, filenames in os.walk(case_dir):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                records.append(("direct", path, None))
            elif path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        for member in archive.namelist():
                            if (
                                not member.endswith("/")
                                and Path(member).suffix.lower() in VIDEO_EXTENSIONS
                            ):
                                records.append(("zip", path, member))
                except zipfile.BadZipFile:
                    continue
    return records


def materialize_video(
    kind: str, container: Path, member: str | None, temporary: Path
) -> Path:
    if kind == "direct":
        return container
    assert member is not None
    suffix = Path(member).suffix.lower()
    target = temporary / f"video{suffix}"
    with zipfile.ZipFile(container) as archive, archive.open(member) as source, target.open(
        "wb"
    ) as destination:
        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    return target


def extract_endoscopic_roi(frame: np.ndarray, output_size: int) -> np.ndarray:
    """Extract the lower-screen, high-saturation endoscopic field only.

    The ION recordings include planning and status panels around the operative
    image.  Those panels are excluded rather than merely blacked out.  A frame
    without a reliable endoscopic colour region is rejected for manual QC.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.zeros((height, width), dtype=np.uint8)
    lower[int(height * 0.40) :, :] = 1
    mask = (
        (hsv[..., 1] > 45) & (hsv[..., 2] > 45) & lower.astype(bool)
    ).astype(np.uint8)
    count, _, statistics, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for x, y, roi_width, roi_height, area in statistics[1:count]:
        aspect = roi_width / max(roi_height, 1)
        if area >= 0.003 * height * width and 0.45 <= aspect <= 1.8:
            candidates.append((int(area), int(x), int(y), int(roi_width), int(roi_height)))
    if not candidates:
        raise ValueError("endoscopic_roi_not_found")
    _, x, y, roi_width, roi_height = max(candidates)
    pad = int(0.12 * max(roi_width, roi_height))
    center_x, center_y = x + roi_width // 2, y + roi_height // 2
    side = min(
        max(roi_width, roi_height) + 2 * pad,
        width,
        height,
    )
    x0 = max(0, min(width - side, center_x - side // 2))
    y0 = max(0, min(height - side, center_y - side // 2))
    crop = frame[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def frame_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()[:16]


def extract_video(
    path: Path,
    case_id: str,
    video_index: int,
    output: Path,
    sample_fps: float,
    output_size: int,
    max_frames: int | None,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or source_width <= 0 or source_height <= 0:
        capture.release()
        raise ValueError("video metadata unavailable")
    stride = max(1, round(source_fps / sample_fps))
    target_dir = output / "frames" / case_id / f"video_{video_index:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    frame_records = []
    source_index = 0
    retained = 0
    rejected_no_roi = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % stride == 0:
            try:
                redacted = extract_endoscopic_roi(frame, output_size)
            except ValueError:
                rejected_no_roi += 1
                source_index += 1
                continue
            filename = f"frame_{retained:06d}.jpg"
            cv2.imwrite(
                str(target_dir / filename),
                redacted,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            frame_records.append(
                {
                    "frame": filename,
                    "source_time_s": round(source_index / source_fps, 3),
                    "image_sha256_16": frame_sha256(redacted),
                }
            )
            retained += 1
            if max_frames is not None and retained >= max_frames:
                break
        source_index += 1
    capture.release()
    return {
        "video_id": f"{case_id}_video_{video_index:02d}",
        "output_path": str(target_dir.relative_to(output)),
        "source_fps": source_fps,
        "source_frame_count": source_frames,
        "source_resolution": [source_width, source_height],
        "sampling_fps": source_fps / stride,
        "retained_frames": retained,
        "rejected_frames_without_reliable_roi": rejected_no_roi,
        "frames": frame_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=2000,
        help="Use 0 for full sampling after visual QC.",
    )
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    max_frames = args.max_frames_per_video or None
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(path for path in args.source.iterdir() if path.is_dir())
    if args.limit_cases is not None:
        case_dirs = case_dirs[: args.limit_cases]
    records = []
    for case_index, case_dir in enumerate(case_dirs, start=1):
        case_id = safe_case_id(case_index)
        video_sources = discover_videos(case_dir)
        case_record: dict[str, object] = {
            "case_id": case_id,
            "needs_visual_qc": True,
            "source_patient_identifier_retained": False,
            "videos": [],
        }
        for video_index, (kind, container, member) in enumerate(video_sources, start=1):
            with tempfile.TemporaryDirectory(prefix="ion_video_") as temporary_name:
                try:
                    video_path = materialize_video(
                        kind, container, member, Path(temporary_name)
                    )
                    record = extract_video(
                        video_path,
                        case_id,
                        video_index,
                        args.output,
                        args.sample_fps,
                        args.size,
                        max_frames,
                    )
                except Exception as error:
                    record = {
                        "video_id": f"{case_id}_video_{video_index:02d}",
                        "extraction_error": type(error).__name__,
                    }
            # Never expose the raw path/member in the manifest.
            record["source_container"] = kind
            case_record["videos"].append(record)
        records.append(case_record)
        saved = sum(
            video.get("retained_frames", 0) for video in case_record["videos"]
        )
        print(f"{case_id}: {len(video_sources)} videos, {saved} frames", flush=True)

    manifest = {
        "dataset_name": "ION bronchoscopy candidate frames",
        "status": "candidate_only_requires_visual_deidentification_qc",
        "source_modified": False,
        "raw_identifiers_persisted": False,
        "audio_retained": False,
        "image_transform": {
            "method": "lower-screen high-saturation endoscopic ROI",
            "output_size": args.size,
            "sample_fps_requested": args.sample_fps,
        },
        "cases": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total = sum(
        video.get("retained_frames", 0)
        for record in records
        for video in record["videos"]
    )
    (args.output / "README.md").write_text(
        "\n".join(
            [
                "# ION bronchoscopy candidate frame dataset",
                "",
                f"- Cases processed: {len(records)}",
                f"- Frames retained: {total}",
                f"- Sampling: requested {args.sample_fps} fps, capped at "
                f"{args.max_frames_per_video} frames/video.",
                "",
                "All frames require visual de-identification QC before modelling "
                "or sharing. The manifest deliberately excludes raw patient IDs, "
                "source paths, labels, DICOM metadata, and audio.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
