
"""Verify that this release contains no restricted/raw artifacts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".dcm", ".dicom", ".mp4", ".zip", ".pt", ".pth", ".joblib", ".npz"
}
FORBIDDEN_TEXT = (
    "E:\\Diff_Rending_Re_3D",
)


def main() -> None:
    failures: list[str] = []
    json_count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact: {path.relative_to(ROOT)}")
        if path.stat().st_size > 25 * 1024 * 1024:
            failures.append(f"oversized file: {path.relative_to(ROOT)}")
        if path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            json_count += 1
        if path.suffix.lower() in {".py", ".md", ".txt", ".tex", ".json", ".yml", ".yaml", ".toml"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in FORBIDDEN_TEXT:
                if token in text:
                    failures.append(
                        f"private token in {path.relative_to(ROOT)}: {token}"
                    )
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"Release verification passed: {json_count} JSON files; "
        "no raw/checkpoint artifacts or known private paths."
    )


if __name__ == "__main__":
    main()
