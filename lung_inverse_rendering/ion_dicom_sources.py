"""Read DICOM objects from case folders and ZIP archives without extraction.

Source paths, archive member names, and DICOM identifiers remain transient.
Callers must not serialize :class:`DicomSource` instances or raw datasets.
"""
from __future__ import annotations

import io
import os
import zipfile
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom


DICOM_EXTENSIONS = {".dcm", ".ima"}
HEADER_TAGS = [
    "Modality",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SeriesNumber",
    "Rows",
    "Columns",
]


@dataclass(frozen=True)
class DicomSource:
    """Transient locator for one direct or archived DICOM object."""

    container: Path
    member: str | None = None


@dataclass
class DicomInventory:
    """In-memory case inventory; only aggregate fields may be persisted."""

    candidate_object_count: int
    modality_counts: dict[str, int]
    ct_object_count: int
    ct_unique_object_count: int
    ct_series: list[list[DicomSource]]
    unreadable_archive_count: int
    unreadable_member_count: int


def _has_dicom_preamble(stream: Any) -> bool:
    try:
        stream.seek(128)
        result = stream.read(4) == b"DICM"
        stream.seek(0)
        return result
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _direct_candidate(path: Path) -> bool:
    if path.suffix.lower() in DICOM_EXTENSIONS:
        return True
    if path.suffix:
        return False
    try:
        with path.open("rb") as stream:
            return _has_dicom_preamble(stream)
    except OSError:
        return False


def _archive_member_candidate(name: str, stream: Any) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix in DICOM_EXTENSIONS:
        return True
    if suffix:
        return False
    return _has_dicom_preamble(stream)


def _series_key(dataset: Any) -> str:
    uid = str(getattr(dataset, "SeriesInstanceUID", ""))
    if uid:
        return f"uid:{uid}"
    return "fallback:" + ":".join(
        (
            str(getattr(dataset, "SeriesNumber", "")),
            str(getattr(dataset, "Rows", "")),
            str(getattr(dataset, "Columns", "")),
        )
    )


def _object_key(dataset: Any, source: DicomSource) -> str:
    uid = str(getattr(dataset, "SOPInstanceUID", ""))
    if uid:
        return f"uid:{uid}"
    return f"source:{source.container}:{source.member or ''}"


def scan_case_dicoms(case_dir: Path) -> DicomInventory:
    """Scan direct and first-level ZIP members, deduplicating CT by SOP UID."""

    modality_counts: dict[str, int] = {}
    grouped: dict[str, dict[str, DicomSource]] = {}
    candidate_count = 0
    ct_object_count = 0
    unreadable_archives = 0
    unreadable_members = 0
    archives: list[Path] = []

    def register(source: DicomSource, dataset: Any) -> None:
        nonlocal candidate_count, ct_object_count
        candidate_count += 1
        modality = str(getattr(dataset, "Modality", "unknown"))
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
        if modality != "CT":
            return
        ct_object_count += 1
        grouped.setdefault(_series_key(dataset), {}).setdefault(
            _object_key(dataset, source), source
        )

    for root, _, names in os.walk(case_dir):
        for name in names:
            path = Path(root) / name
            if path.suffix.lower() == ".zip":
                archives.append(path)
                continue
            if not _direct_candidate(path):
                continue
            try:
                dataset = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=False,
                    specific_tags=HEADER_TAGS,
                )
            except Exception:
                continue
            register(DicomSource(path), dataset)

    for archive_path in archives:
        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile):
            unreadable_archives += 1
            continue
        with archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                try:
                    with archive.open(info) as stream:
                        if not _archive_member_candidate(info.filename, stream):
                            continue
                        dataset = pydicom.dcmread(
                            stream,
                            stop_before_pixels=True,
                            force=False,
                            specific_tags=HEADER_TAGS,
                        )
                except Exception:
                    unreadable_members += 1
                    continue
                register(DicomSource(archive_path, info.filename), dataset)

    series = [
        list(objects.values())
        for objects in grouped.values()
        if objects
    ]
    series.sort(key=len, reverse=True)
    return DicomInventory(
        candidate_object_count=candidate_count,
        modality_counts=dict(sorted(modality_counts.items())),
        ct_object_count=ct_object_count,
        ct_unique_object_count=sum(len(paths) for paths in series),
        ct_series=series,
        unreadable_archive_count=unreadable_archives,
        unreadable_member_count=unreadable_members,
    )


@contextmanager
def open_dicom_reader():
    """Yield a reader that reuses open ZIP containers within one operation."""

    with ExitStack() as stack:
        archives: dict[Path, zipfile.ZipFile] = {}

        def read(
            source: DicomSource,
            *,
            stop_before_pixels: bool = False,
            specific_tags: list[str] | None = None,
        ) -> Any:
            if source.member is None:
                return pydicom.dcmread(
                    source.container,
                    force=False,
                    stop_before_pixels=stop_before_pixels,
                    specific_tags=specific_tags,
                )
            archive = archives.get(source.container)
            if archive is None:
                archive = stack.enter_context(zipfile.ZipFile(source.container))
                archives[source.container] = archive
            payload = archive.read(source.member)
            return pydicom.dcmread(
                io.BytesIO(payload),
                force=False,
                stop_before_pixels=stop_before_pixels,
                specific_tags=specific_tags,
            )

        yield read


def load_datasets(sources: list[DicomSource]) -> list[Any]:
    """Load complete datasets while opening each ZIP container only once."""

    with open_dicom_reader() as read:
        return [read(source) for source in sources]
