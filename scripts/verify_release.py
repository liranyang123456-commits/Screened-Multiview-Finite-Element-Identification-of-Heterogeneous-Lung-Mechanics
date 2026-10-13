
"""Verify that this release contains no restricted/raw artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".dcm", ".dicom", ".mp4", ".zip", ".pt", ".pth", ".joblib", ".npz"
}
FORBIDDEN_TEXT = (
    "\x45:\\Diff_Rending_Re_3D",
    "\x45:\\\\Diff_Rending_Re_3D",   # JSON/regex-escaped form
    "\x45:\\",               # any absolute path of the development machine
    "\x45:/",                # forward-slash form of the same path
    "\x46:\\",               # private raw-data drive root
    "\u7701\u533b",          # institution token
    "ION\u60a3\u8005\u6574\u7406",  # private source folder name
)


def iter_files():
    """(relative path, OS path) of every file, including Windows extended-length names."""
    base = str(ROOT.resolve())
    if os.name == "nt" and not base.startswith("\\\\?\\"):
        base = "\\\\?\\" + base
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [name for name in dirnames if name not in {".git", "__pycache__"}]
        for name in filenames:
            full = os.path.join(dirpath, name)
            yield Path(os.path.relpath(full, base)), full


def main() -> None:
    failures: list[str] = []
    json_count = 0
    for relative, full in iter_files():
        suffix = relative.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact: {relative}")
        if os.path.getsize(full) > 25 * 1024 * 1024:
            failures.append(f"oversized file: {relative}")
        if suffix == ".json":
            with open(full, encoding="utf-8") as handle:
                json.loads(handle.read())
            json_count += 1
        if suffix in {".py", ".md", ".txt", ".tex", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".ps1", ".cff"}:
            with open(full, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
            for token in FORBIDDEN_TEXT:
                if token in text:
                    failures.append(f"private token in {relative}: {token}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"Release verification passed: {json_count} JSON files; "
        "no raw/checkpoint artifacts, no development-machine or private paths."
    )


if __name__ == "__main__":
    main()
