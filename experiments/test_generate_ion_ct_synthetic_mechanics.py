"""Protocol tests for de-identified CT synthetic-mechanics generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import experiments.generate_ion_ct_synthetic_mechanics as generator
from dataset.sim_lung_graph import SimLungGraphDataset
from experiments.generate_ion_ct_synthetic_mechanics import (
    discover_geometry_meshes,
    generation_config,
    manifest_payload,
    scenario_spec,
)
from lung_inverse_rendering.evaluate_sim_lung_v2 import load_patient


def test_geometry_discovery_limit_is_deterministic(tmp_path: Path) -> None:
    for geometry_id in ("geom_cc", "geom_aa", "geom_bb"):
        np.savez(tmp_path / f"{geometry_id}.npz", vertices=np.empty((0, 3)))
    selected = discover_geometry_meshes(tmp_path, geometry_limit=2)
    assert [path.stem for path in selected] == ["geom_aa", "geom_bb"]


def test_scenario_protocol_is_private_test_only_and_synthetic() -> None:
    geometry_id = "geom_ab578f1bb0ee02be2ccd"
    spec = scenario_spec(7, 20, geometry_id=geometry_id)
    config = generation_config(
        geometry_ids=[geometry_id],
        scenarios_per_geometry=20,
        resolution=48,
        motion_noise_std=2.5e-4,
        save_images=False,
        force_prior_fraction=0.05,
    )
    assert spec["patient_id"] == f"{geometry_id}_scenario_007"
    assert spec["scenario_id"] == "scenario_007"
    assert spec["scenario_template_index"] == 7
    assert spec["split"] == "test"
    assert spec["material_source"] == "synthetic"
    assert spec["mechanics_source"] == "synthetic"
    assert config["load_count"] == 4
    assert config["frame_count"] == 7
    assert config["num_views"] == 3


def _minimal_multiview_experiment(patient_id: str, name: str) -> dict:
    frames, views, nodes = 7, 3, 4
    forces = torch.zeros((frames, 3 * nodes), dtype=torch.float64)
    forces[2, 2] = 1.0
    return {
        "schema_version": generator.SCHEMA_VERSION,
        "patient_id": patient_id,
        "experiment": name,
        "nodes": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        "elems": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "surface_tris": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "surface_node_ids": torch.arange(nodes),
        "fixed": torch.tensor([0, 1, 2], dtype=torch.long),
        "forces": forces,
        "forces_measured": forces.clone(),
        "poses": torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1),
        "poses_multiview": torch.eye(4, dtype=torch.float64).repeat(
            frames, views, 1, 1
        ),
        "image_uv_rest_multiview_seq": torch.zeros(
            frames, views, nodes, 2, dtype=torch.float64
        ),
        "image_uv_deformed_multiview_seq": torch.zeros(
            frames, views, nodes, 2, dtype=torch.float64
        ),
        "image_depth_rest_multiview_seq": torch.ones(
            frames, views, nodes, dtype=torch.float64
        ),
        "image_depth_deformed_multiview_seq": torch.ones(
            frames, views, nodes, dtype=torch.float64
        ),
        "image_occlusion_confidence_multiview_seq": torch.ones(
            frames, views, nodes, dtype=torch.float64
        ),
        "render_intrinsics": {"focal_px": 200.0, "height_px": 48, "width_px": 48},
        "inclusion_ratio": 1.5,
        "E_background": 5_000.0,
        "inclusion_center": torch.tensor([0.5, 0.5, 0.5]),
        "inclusion_radius": 0.2,
    }


def test_manifest_is_readable_by_graph_dataset_and_evaluator(tmp_path: Path) -> None:
    spec = scenario_spec(0, 5, geometry_id="geom_private")
    experiments = []
    for name in ("press_left", "press_right", "shear_x", "shear_y"):
        relative_path = Path(spec["patient_id"]) / name / "gt.pt"
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True)
        torch.save(_minimal_multiview_experiment(spec["patient_id"], name), destination)
        experiments.append(
            {"name": name, "relative_path": str(relative_path), "minimum_jacobian": 1.0}
        )
    row = {**spec, "experiments": experiments}
    config = generation_config(
        geometry_ids=["geom_private"],
        scenarios_per_geometry=5,
        resolution=48,
        motion_noise_std=2.5e-4,
        save_images=False,
        force_prior_fraction=0.05,
    )
    manifest = manifest_payload(
        {spec["patient_id"]: row},
        total_scenarios=5,
        config=config,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    graph = SimLungGraphDataset(tmp_path, split="test")[0]
    scene, loaded = load_patient(tmp_path, row)
    assert graph["dynamic_seq"].shape == (4, 3, 7, 4, 14)
    assert len(loaded) == 4
    assert scene["Nn"] == 4
    serialized = json.dumps(manifest)
    assert "case_id" not in serialized
    assert "C:\\" not in serialized
    assert "E:\\" not in serialized


def test_range_resume_merges_completed_scenarios(
    tmp_path: Path, monkeypatch
) -> None:
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    np.savez(mesh_dir / "geom_private.npz", vertices=np.empty((0, 3)))
    scene = {
        "nodes": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.2],
            ],
            dtype=torch.float64,
        ),
        "elems": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "surface_tris": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "fixed": torch.tensor([0, 1, 2], dtype=torch.long),
        "Nn": 4,
        "lx": 1.0,
        "ly": 1.0,
        "nu_true": torch.tensor(0.45, dtype=torch.float64),
    }
    monkeypatch.setattr(generator, "build_scene_from_ct_mesh", lambda *args, **kwargs: scene)

    def fake_generate(
        out_root: Path,
        patient: dict,
        _scene: dict,
        *_args,
        experiment: dict | None = None,
        experiment_index: int,
        **_kwargs,
    ) -> dict:
        current = experiment or generator.EXPERIMENTS[experiment_index]
        relative = Path(patient["patient_id"]) / current["name"] / "gt.pt"
        destination = out_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"patient_id": patient["patient_id"]}, destination)
        return {"name": current["name"], "relative_path": str(relative)}

    monkeypatch.setattr(generator, "generate_experiment", fake_generate)
    output = tmp_path / "dataset"

    def arguments(start: int, end: int, *, resume: bool) -> argparse.Namespace:
        return argparse.Namespace(
            mesh_dir=mesh_dir,
            out=output,
            geometry_limit=None,
            scenarios_per_geometry=2,
            scenario_start=start,
            scenario_end=end,
            resolution=48,
            motion_noise_std=2.5e-4,
            force_prior_fraction=0.05,
            no_images=True,
            resume=resume,
            overwrite=False,
        )

    generator.generate_dataset(arguments(0, 1, resume=False))
    generator.generate_dataset(arguments(1, 2, resume=True))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_patient_count"] == 2
    assert [row["patient_id"] for row in manifest["patients"]] == [
        "geom_private_scenario_000",
        "geom_private_scenario_001",
    ]
    assert all(len(row["experiments"]) == 4 for row in manifest["patients"])
