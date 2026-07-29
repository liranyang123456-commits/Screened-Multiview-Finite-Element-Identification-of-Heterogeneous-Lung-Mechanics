from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from lung_inverse_rendering.test_lung_ai import _write_patient


ROOT = Path(__file__).resolve().parents[1]


def test_tiny_training_and_checkpoint_resume(tmp_path: Path) -> None:
    rows = [
        _write_patient(tmp_path, index, ratio=1.0 + 0.4 * index, extra_node=True)
        for index in range(5)
    ]
    for index, row in enumerate(rows):
        row["split"] = "train" if index < 3 else "val" if index == 3 else "test"
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": "tiny", "patients": rows}), encoding="utf-8"
    )
    results = tmp_path / "results"
    base = [
        sys.executable,
        str(ROOT / "experiments" / "train_lung_mesh_gnn.py"),
        "--dataset",
        str(tmp_path),
        "--results",
        str(results),
        "--model",
        "gnn",
        "--batch-size",
        "2",
        "--hidden-dim",
        "16",
        "--layers",
        "1",
        "--patience",
        "5",
        "--no-rotation-augmentation",
    ]
    subprocess.run([*base, "--epochs", "2"], check=True, capture_output=True)
    assert (results / "best_gnn.pt").exists()
    assert json.loads(
        (results / "metrics_gnn.json").read_text(encoding="utf-8")
    )["test"]["patient_count"] == 1
    subprocess.run(
        [
            *base,
            "--epochs",
            "3",
            "--resume",
            str(results / "last_gnn.pt"),
        ],
        check=True,
        capture_output=True,
    )
    state = json.loads(
        (results / "metrics_gnn.json").read_text(encoding="utf-8")
    )
    assert max(row["epoch"] for row in state["history"]) == 2


def test_multiview_temporal_training_and_frozen_evaluation(tmp_path: Path) -> None:
    rows = [
        _write_patient(
            tmp_path,
            index,
            ratio=1.0 + 0.5 * index,
            extra_node=True,
            multiview=True,
        )
        for index in range(3)
    ]
    for index, row in enumerate(rows):
        row["split"] = ("train", "val", "test")[index]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": "temporal-tiny", "patients": rows}),
        encoding="utf-8",
    )
    results = tmp_path / "temporal_results"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "train_lung_mesh_gnn.py"),
            "--dataset",
            str(tmp_path),
            "--results",
            str(results),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "16",
            "--layers",
            "1",
            "--no-rotation-augmentation",
        ],
        check=True,
        capture_output=True,
    )
    checkpoint = results / "best_gnn.pt"
    state = __import__("torch").load(checkpoint, map_location="cpu", weights_only=True)
    assert state["config"]["dynamic_dim"] == 14
    output = tmp_path / "frozen_temporal.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluation" / "evaluate_lung_mesh_gnn.py"),
            "--dataset",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(output),
            "--batch-size",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    frozen = json.loads(output.read_text(encoding="utf-8"))
    assert frozen["metrics"]["patient_count"] == 1
    assert frozen["metrics"]["node_sdf_mae_mean"] is not None
