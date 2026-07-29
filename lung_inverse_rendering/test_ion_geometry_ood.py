"""Synthetic-only tests for ION geometry-domain OOD validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_lung_geometry_ood import (  # noqa: E402
    evaluate_geometry,
    surface_mesh_qc,
)
from experiments.train_lung_mesh_gnn import build_model  # noqa: E402
from lung_inverse_rendering.ct_loader import lung_air_mask, mask_to_surface  # noqa: E402
from lung_inverse_rendering.prepare_ion_ct_meshes import (  # noqa: E402
    _bilateral_lung_score,
    opaque_geometry_id,
    prepare,
    select_ct_candidates,
)


def _synthetic_hu() -> np.ndarray:
    z, y, x = np.mgrid[:36, :36, :36]
    volume = np.zeros((36, 36, 36), dtype=np.float32)
    left = (x - 11) ** 2 + (y - 18) ** 2 + (z - 18) ** 2 < 8**2
    right = (x - 25) ** 2 + (y - 18) ** 2 + (z - 18) ** 2 < 8**2
    volume[left | right] = -750.0
    return volume


def _synthetic_surface() -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, 21)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.column_stack(
        (xx.ravel(), yy.ravel(), 0.08 * (np.square(xx) + np.square(yy)).ravel())
    )
    faces = []
    for row in range(20):
        for column in range(20):
            first = row * 21 + column
            faces.extend(
                ([first, first + 1, first + 22], [first, first + 22, first + 21])
            )
    return vertices, np.asarray(faces, dtype=np.int64)


def _audit_manifest() -> dict:
    return {
        "case_records": [
            {
                "case_id": "case_001",
                "dicom": {"candidate_files": 20, "modalities": {"CT": 10}},
            },
            {
                "case_id": "case_002",
                "dicom": {"candidate_files": 4, "modalities": {"MR": 4}},
            },
        ]
    }


def test_synthetic_hu_extracts_valid_surface() -> None:
    mask = lung_air_mask(_synthetic_hu())
    vertices, faces = mask_to_surface(mask, (1.0, 1.0, 1.0))
    qc = surface_mesh_qc(vertices, faces)
    assert mask.any()
    assert qc["valid"]
    assert qc["vertex_count"] > 100


def test_bilateral_lung_score_rejects_single_air_cavity() -> None:
    bilateral = np.zeros((3, 64, 64), dtype=bool)
    bilateral[:, 16:48, 8:26] = True
    bilateral[:, 16:48, 38:56] = True
    single = np.zeros_like(bilateral)
    single[:, 20:44, 18:46] = True
    assert _bilateral_lung_score(bilateral) > 0.01
    assert _bilateral_lung_score(single) == 0.0


def test_audit_selection_and_dry_run_are_private(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("lung_inverse_rendering.prepare_ion_ct_meshes.ROOT", tmp_path)
    audit = _audit_manifest()
    selected = select_ct_candidates(audit)
    assert selected == [
        {
            "_case_id": "case_001",
            "geometry_id": opaque_geometry_id("case_001"),
        }
    ]
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    output = tmp_path / "deidentified_output"
    result = prepare(
        audit_path,
        output,
        export=False,
        source_root=tmp_path / "source_must_not_be_accessed",
    )
    assert result["mode"] == "dry_run"
    assert not result["source_accessed"]
    assert not result["pixels_read"]
    serialized = (output / "privacy_manifest.json").read_text(encoding="utf-8")
    assert "case_001" not in serialized
    assert "SeriesInstanceUID" not in serialized


def test_explicit_export_uses_only_opaque_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("lung_inverse_rendering.prepare_ion_ct_meshes.ROOT", tmp_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(_audit_manifest()), encoding="utf-8")
    source = tmp_path / "synthetic_source"
    (source / "synthetic_case").mkdir(parents=True)
    output = tmp_path / "deidentified_output"

    def fake_export(case_dir: Path, destination: Path, geometry_id: str) -> None:
        assert case_dir.name == "synthetic_case"
        vertices, faces = _synthetic_surface()
        np.savez_compressed(
            destination,
            vertices=vertices,
            faces=faces,
            geometry_id=np.asarray(geometry_id),
        )

    monkeypatch.setattr(
        "lung_inverse_rendering.prepare_ion_ct_meshes._export_case_mesh",
        fake_export,
    )
    result = prepare(audit_path, output, export=True, source_root=source)
    assert len(result["exported_geometry_ids"]) == 1
    assert not result["failures"]
    assert (output / f"{result['exported_geometry_ids'][0]}.npz").exists()


def test_surrogate_fem_and_model_stability_have_no_gt_errors(tmp_path: Path) -> None:
    config = {
        "model": "gnn",
        "input_dim": 29,
        "hidden_dim": 16,
        "layers": 2,
        "dropout": 0.0,
    }
    model = build_model(
        config["model"],
        config["input_dim"],
        config["hidden_dim"],
        config["layers"],
        config["dropout"],
    )
    checkpoint = tmp_path / "synthetic_checkpoint.pt"
    torch.save({"config": config, "model": model.state_dict()}, checkpoint)
    result = evaluate_geometry(checkpoint=checkpoint, seed=17)
    assert result["geometry_source"] == "synthetic_ct_surrogate"
    assert result["fem_mesh_qc"]["construction_success"]
    assert result["fem_mesh_qc"]["minimum_deformation_jacobian"] > 0.999
    assert result["model_output_stability"]["finite_outputs"]
    assert result["model_output_stability"]["within_declared_output_ranges"]
    text = json.dumps(result)
    assert "relative_error" not in text
    assert "E_background_true" not in text
    assert "inclusion_ratio_true" not in text


def test_synthetic_surface_mesh_geometry_evaluation(tmp_path: Path) -> None:
    vertices, faces = _synthetic_surface()
    mesh_path = tmp_path / "synthetic_mesh.npz"
    np.savez_compressed(mesh_path, vertices=vertices, faces=faces)
    result = evaluate_geometry(
        mesh_path=mesh_path,
        geometry_id="geom_synthetic_test",
    )
    assert result["surface_mesh_qc"]["valid"]
    assert result["fem_mesh_qc"]["construction_success"]
    assert result["ground_truth_material_metrics_reported"] is False
