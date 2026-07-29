"""ZIP-native loader and preprocessed cache for the 50-recording benchmark."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import mmap
import re
import struct
import zipfile
import zlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CACHE = ROOT / "cache_112"
MANIFEST = ROOT / "manifest.json"
FRAME_RE = re.compile(r"(\d+)\.npy$")
IMAGE_SIZE = 112
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _zip_path(recording: int) -> Path:
    raw = RAW / f"{recording}.zip"
    if raw.exists():
        return raw
    sample = ROOT / "sample" / f"{recording}.zip"
    if sample.exists():
        return sample
    raise FileNotFoundError(raw)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_rows(archive: zipfile.ZipFile, recording: int) -> list[dict]:
    name = f"{recording}/dataset.csv"
    rows = list(csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8")))
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        match = FRAME_RE.search(row["image"])
        if match:
            grouped[int(match.group(1))].append(float(row["force"]))
    return [
        {
            "frame": frame,
            "force_n": float(np.mean(values)),
            "duplicate_labels": len(values),
        }
        for frame, values in sorted(grouped.items())
    ]


def _recover_stored_entry(path: Path, entry_name: str) -> bytes:
    """Read an early stored entry from a ZIP whose central directory is absent.

    Zenodo's checksum-valid 26.zip is truncated after its early entries. Its
    video and CSV files are stored (compression method 0) and remain auditable.
    """
    with path.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as mapped:
        position = 0
        while position + 30 <= len(mapped) and mapped[position : position + 4] == b"PK\x03\x04":
            fields = struct.unpack_from("<IHHHHHIIIHH", mapped, position)
            flags, compression, compressed_size = fields[2], fields[3], fields[7]
            name_length, extra_length = fields[9], fields[10]
            name_start = position + 30
            name = mapped[name_start : name_start + name_length].decode("utf-8")
            data_start = name_start + name_length + extra_length
            if flags & 0x08:
                descriptor = mapped.find(b"PK\x07\x08", data_start)
                while descriptor >= 0:
                    if descriptor + 20 <= len(mapped):
                        _, _, candidate_size, _ = struct.unpack_from(
                            "<IIII", mapped, descriptor
                        )
                        next_position = descriptor + 16
                        if (
                            candidate_size == descriptor - data_start
                            and mapped[next_position : next_position + 4] == b"PK\x03\x04"
                        ):
                            compressed_size = candidate_size
                            break
                    descriptor = mapped.find(b"PK\x07\x08", descriptor + 1)
                if descriptor < 0:
                    raise ValueError(f"Cannot locate data descriptor after {name}")
                position_after = descriptor + 16
            else:
                position_after = data_start + compressed_size
            if name == entry_name:
                if data_start + compressed_size > len(mapped):
                    raise ValueError(f"{entry_name} is truncated")
                data = bytes(mapped[data_start : data_start + compressed_size])
                if compression == 0:
                    return data
                if compression == 8:
                    return zlib.decompress(data, -zlib.MAX_WBITS)
                raise ValueError(
                    f"{entry_name} uses unsupported compression method {compression}"
                )
            position = position_after
    raise KeyError(entry_name)


def _recovered_labels(path: Path, recording: int) -> list[dict]:
    data = _recover_stored_entry(path, f"{recording}/dataset.csv")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        match = FRAME_RE.search(row["image"])
        if match:
            grouped[int(match.group(1))].append(float(row["force"]))
    return [
        {
            "frame": frame,
            "force_n": float(np.mean(values)),
            "duplicate_labels": len(values),
        }
        for frame, values in sorted(grouped.items())
    ]


def inspect_recording(recording: int) -> dict:
    path = _zip_path(recording)
    archive_valid = zipfile.is_zipfile(path)
    if archive_valid:
        with zipfile.ZipFile(path) as archive:
            labels = _label_rows(archive, recording)
            frame_indices = sorted(
                int(match.group(1))
                for name in archive.namelist()
                if (match := re.search(rf"^{recording}/imgs/(\d+)\.npy$", name))
            )
    else:
        labels = _recovered_labels(path, recording)
        video_bytes = _recover_stored_entry(path, f"{recording}/video.mp4")
        temporary = CACHE / f".inspect_{recording}.mp4"
        CACHE.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(video_bytes)
        capture = cv2.VideoCapture(str(temporary))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        temporary.unlink()
        frame_indices = list(range(frame_count))
    if not labels or not frame_indices:
        raise ValueError(f"Recording {recording} has no labels or frames")
    label_frames = {row["frame"] for row in labels}
    missing = sorted(label_frames - set(frame_indices))
    if missing:
        raise ValueError(f"Recording {recording} labels missing frames: {missing}")
    forces = np.asarray([row["force_n"] for row in labels], dtype=float)
    return {
        "recording": recording,
        "geometry_id": (recording - 1) // 10 + 1,
        "camera_id": (recording - 1) % 10 + 1,
        "zip": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "md5": _md5(path),
        "source_archive_valid": archive_valid,
        "frames": len(frame_indices),
        "first_frame": frame_indices[0],
        "last_frame": frame_indices[-1],
        "labels": len(labels),
        "duplicate_label_rows": sum(max(0, row["duplicate_labels"] - 1) for row in labels),
        "force_min_n": float(forces.min()),
        "force_max_n": float(forces.max()),
        "force_mean_n": float(forces.mean()),
    }


def build_manifest(
    recordings: Iterable[int] = range(1, 51),
    path: Path = MANIFEST,
) -> dict:
    rows = [inspect_recording(recording) for recording in recordings]
    payload = {
        "dataset": "https://doi.org/10.5281/zenodo.19370452",
        "license": "CC-BY-4.0",
        "recordings": rows,
        "recording_count": len(rows),
        "total_frames": sum(row["frames"] for row in rows),
        "total_labels": sum(row["labels"] for row in rows),
        "split_unit": "recording",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _prepare_recording(recording: int, overwrite: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames_path = CACHE / f"{recording:02d}_frames.npy"
    labels_path = CACHE / f"{recording:02d}_labels.json"
    if frames_path.exists() and labels_path.exists() and not overwrite:
        return json.loads(labels_path.read_text(encoding="utf-8"))

    path = _zip_path(recording)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            labels = _label_rows(archive, recording)
            frame_names = {
                int(match.group(1)): name
                for name in archive.namelist()
                if (match := re.search(rf"^{recording}/imgs/(\d+)\.npy$", name))
            }
            indices = sorted(frame_names)
            frames = np.empty((indices[-1] + 1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            for index in range(indices[-1] + 1):
                source_index = index if index in frame_names else max(i for i in indices if i < index)
                frame = np.load(io.BytesIO(archive.read(frame_names[source_index])))
                height, width = frame.shape[:2]
                side = min(height, width)
                x0 = (width - side) // 2
                y0 = (height - side) // 2
                crop = frame[y0 : y0 + side, x0 : x0 + side]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                frames[index] = cv2.resize(
                    rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
                )
    else:
        labels = _recovered_labels(path, recording)
        temporary = CACHE / f".recover_{recording}.mp4"
        temporary.write_bytes(_recover_stored_entry(path, f"{recording}/video.mp4"))
        capture = cv2.VideoCapture(str(temporary))
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            side = min(height, width)
            x0 = (width - side) // 2
            y0 = (height - side) // 2
            crop = frame[y0 : y0 + side, x0 : x0 + side]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            decoded.append(
                cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            )
        capture.release()
        temporary.unlink()
        frames = np.stack(decoded)

    np.save(frames_path, frames)
    payload = {
        "recording": recording,
        "geometry_id": (recording - 1) // 10 + 1,
        "camera_id": (recording - 1) % 10 + 1,
        "frames_path": frames_path.name,
        "shape": list(frames.shape),
        "labels": labels,
    }
    labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def prepare_cache(
    recordings: Iterable[int] = range(1, 51),
    overwrite: bool = False,
) -> list[dict]:
    rows = []
    for recording in recordings:
        rows.append(_prepare_recording(recording, overwrite=overwrite))
        print(f"Prepared recording {recording:02d}", flush=True)
    return rows


@lru_cache(maxsize=12)
def _cached_frames(recording: int) -> np.ndarray:
    return np.load(CACHE / f"{recording:02d}_frames.npy", mmap_mode="r")


class SmallBowelForceDataset(Dataset):
    """Visual windows with scalar force targets; split is recording-level."""

    def __init__(
        self,
        recordings: Iterable[int],
        window: int = 10,
        augment: bool = False,
        input_size: int = 112,
    ):
        if window not in (10, 20, 30):
            raise ValueError("window must be 10, 20, or 30")
        self.window = window
        self.augment = augment
        self.input_size = input_size
        self.samples: list[tuple[int, int, float]] = []
        for recording in recordings:
            labels_path = CACHE / f"{recording:02d}_labels.json"
            if not labels_path.exists():
                _prepare_recording(recording)
            payload = json.loads(labels_path.read_text(encoding="utf-8"))
            for row in payload["labels"]:
                self.samples.append((recording, int(row["frame"]), float(row["force_n"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        recording, target_frame, force = self.samples[index]
        frames = _cached_frames(recording)
        indices = np.arange(target_frame - self.window + 1, target_frame + 1)
        indices = np.clip(indices, 0, len(frames) - 1)
        clip = np.asarray(frames[indices], dtype=np.float32) / 255.0
        if self.input_size < clip.shape[1]:
            offset = (clip.shape[1] - self.input_size) // 2
            clip = clip[
                :,
                offset : offset + self.input_size,
                offset : offset + self.input_size,
            ]
        if self.augment and torch.rand(()) < 0.5:
            clip = clip[:, :, ::-1].copy()
        if self.augment and torch.rand(()) < 0.2:
            clip = np.stack(
                [cv2.GaussianBlur(frame, (5, 5), 0.8) for frame in clip]
            )
        if self.augment and torch.rand(()) < 0.3:
            shift = (
                torch.empty(3).uniform_(-0.04, 0.04).numpy().reshape(1, 1, 1, 3)
            )
            clip = np.clip(clip + shift, 0.0, 1.0)
        clip = (clip - MEAN) / STD
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()
        return {
            "video": tensor,
            "force": torch.tensor(force, dtype=torch.float32),
            "recording": recording,
            "frame": target_frame,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--recordings", nargs="*", type=int, default=list(range(1, 51)))
    args = parser.parse_args()
    manifest = build_manifest(args.recordings)
    if args.prepare:
        prepare_cache(args.recordings)
    print(json.dumps({key: value for key, value in manifest.items() if key != "recordings"}, indent=2))
