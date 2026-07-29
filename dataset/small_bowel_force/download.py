"""Resumable downloader for Zenodo record 19370452."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RECORD_ID = 19370452
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SAMPLE = ROOT / "sample" / "1.zip"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata() -> dict:
    with urllib.request.urlopen(API_URL, timeout=60) as response:
        return json.load(response)


def download_one(file_info: dict) -> dict:
    key = file_info["key"]
    target = RAW / key
    expected = file_info["checksum"].split(":", 1)[1].lower()
    size = int(file_info["size"])
    if target.exists() and target.stat().st_size == size and md5(target) == expected:
        return {"file": key, "size": size, "md5": expected, "status": "verified"}

    command = [
        "curl.exe",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-delay",
        "3",
        "--continue-at",
        "-",
        "--output",
        str(target),
        file_info["links"]["self"],
    ]
    subprocess.run(command, check=True)
    actual = md5(target)
    if actual != expected:
        raise ValueError(f"{key}: MD5 {actual} != {expected}")
    return {"file": key, "size": size, "md5": actual, "status": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    record = metadata()
    files = sorted(record["files"], key=lambda item: int(Path(item["key"]).stem))
    first = RAW / "1.zip"
    if SAMPLE.exists() and not first.exists():
        try:
            os.link(SAMPLE, first)
        except OSError:
            import shutil

            shutil.copy2(SAMPLE, first)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, info): info["key"] for info in files}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[{len(rows):02d}/{len(files)}] {row['file']} "
                f"{row['status']} ({row['size'] / 1024**2:.1f} MiB)",
                flush=True,
            )

    rows.sort(key=lambda row: int(Path(row["file"]).stem))
    manifest = {
        "record_id": RECORD_ID,
        "doi": record["doi"],
        "license": record["metadata"]["license"]["id"],
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
    }
    (ROOT / "checksums.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("FORCE_DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
