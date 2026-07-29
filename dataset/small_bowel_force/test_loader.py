"""Unit checks for the public video-force adapter."""
from __future__ import annotations

from dataset.small_bowel_force.loader import SmallBowelForceDataset, inspect_recording
from dataset.small_bowel_force.splits import split_recordings


def test_recording_splits_are_complete_and_disjoint():
    expected = set(range(1, 51))
    for protocol in ("geometry", "camera"):
        for fold in range(5):
            split = split_recordings(protocol, fold)
            train, val, test = map(set, (split["train"], split["val"], split["test"]))
            assert len(train) == 30 and len(val) == 10 and len(test) == 10
            assert not train & val
            assert not train & test
            assert not val & test
            assert train | val | test == expected


def test_sample_recording_and_window():
    metadata = inspect_recording(1)
    assert metadata["frames"] == 359
    assert metadata["labels"] == 133
    dataset = SmallBowelForceDataset([1], window=10)
    sample = dataset[0]
    assert tuple(sample["video"].shape) == (3, 10, 112, 112)
    assert sample["force"].ndim == 0
