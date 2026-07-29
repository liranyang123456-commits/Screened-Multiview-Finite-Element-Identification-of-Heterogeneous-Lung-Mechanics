"""Recording-level cross-validation splits without temporal leakage."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def recording_id(geometry_id: int, camera_id: int) -> int:
    if geometry_id not in range(1, 6) or camera_id not in range(1, 11):
        raise ValueError("geometry_id must be 1..5 and camera_id must be 1..10")
    return (geometry_id - 1) * 10 + camera_id


def split_recordings(protocol: str, fold: int) -> dict[str, list[int]]:
    """Return 60/20/20 recording-level train/val/test IDs.

    geometry: one complete geometry is test, the next is validation.
    camera: within every geometry, two camera recordings are test, the next
    pair is validation, and the remaining six are training.
    """
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    if protocol == "geometry":
        test_geometry = fold + 1
        val_geometry = ((fold + 1) % 5) + 1
        split = {
            "train": [
                recording_id(g, c)
                for g in range(1, 6)
                if g not in (test_geometry, val_geometry)
                for c in range(1, 11)
            ],
            "val": [recording_id(val_geometry, c) for c in range(1, 11)],
            "test": [recording_id(test_geometry, c) for c in range(1, 11)],
        }
    elif protocol == "camera":
        test_pair = {(2 * fold) % 10 + 1, (2 * fold + 1) % 10 + 1}
        val_pair = {(2 * fold + 2) % 10 + 1, (2 * fold + 3) % 10 + 1}
        split = {
            name: [
                recording_id(g, c)
                for g in range(1, 6)
                for c in range(1, 11)
                if (
                    (name == "test" and c in test_pair)
                    or (name == "val" and c in val_pair)
                    or (name == "train" and c not in test_pair | val_pair)
                )
            ]
            for name in ("train", "val", "test")
        }
    else:
        raise ValueError("protocol must be 'geometry' or 'camera'")

    sets = {name: set(ids) for name, ids in split.items()}
    if any(sets[a] & sets[b] for a in sets for b in sets if a < b):
        raise AssertionError("recording leakage between splits")
    if set.union(*sets.values()) != set(range(1, 51)):
        raise AssertionError("splits do not cover recordings 1..50")
    if tuple(map(len, (split["train"], split["val"], split["test"]))) != (30, 10, 10):
        raise AssertionError("expected a 60/20/20 recording split")
    return split


def make_all_splits(path: Path = ROOT / "splits.json") -> dict:
    payload = {
        protocol: {
            f"fold_{fold}": split_recordings(protocol, fold) for fold in range(5)
        }
        for protocol in ("geometry", "camera")
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    make_all_splits()
    print("Wrote leak-free 5-fold geometry and camera protocols")
