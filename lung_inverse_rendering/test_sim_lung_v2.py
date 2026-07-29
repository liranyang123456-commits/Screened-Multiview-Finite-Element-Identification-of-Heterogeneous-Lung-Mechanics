"""Tests for patient consistency and measured-input dataset semantics."""
from __future__ import annotations

import torch

from lung_inverse_rendering.augment_sim_lung_v2_tracks import track_fields
from lung_inverse_rendering.generate_sim_lung_v2 import (
    EXPERIMENTS,
    build_patient,
    force_sequence,
    measured_force_fields,
    multiview_projection_fields,
    patient_spec,
)
from lung_inverse_rendering.qc_sim_lung_ai import validate_experiment_protocol
from lung_inverse_rendering.viscoelastic_extension import relaxation_sequence
from rendering.gaussian_pbr import project_with_foreground_confidence
from simulator.scene import make_camera_poses, make_multiview_camera_poses


def test_one_patient_keeps_geometry_material_and_boundary() -> None:
    spec = patient_spec(3, 10)
    first_scene, first_E, first_center, first_radius = build_patient(spec)
    second_scene, second_E, second_center, second_radius = build_patient(spec)
    assert torch.equal(first_scene["nodes"], second_scene["nodes"])
    assert torch.equal(first_scene["fixed"], second_scene["fixed"])
    assert torch.equal(first_E, second_E)
    assert torch.equal(first_center, second_center)
    assert first_radius == second_radius


def test_different_patients_have_different_geometry() -> None:
    first, _, _, _ = build_patient(patient_spec(0, 10))
    second, _, _, _ = build_patient(patient_spec(1, 10))
    assert not torch.equal(first["nodes"], second["nodes"])


def test_build_patient_accepts_prebuilt_deidentified_ct_scene() -> None:
    spec = patient_spec(0, 10)
    source_scene, _, _, _ = build_patient(spec)
    source_scene["geometry_source"] = "approved_deidentified_ct_mesh"
    scene, E_nodes, center, radius = build_patient(
        spec, ct_mesh_scene=source_scene
    )
    assert scene is source_scene
    assert torch.equal(scene["nodes"], source_scene["nodes"])
    assert E_nodes.shape == (scene["Nn"],)
    assert center.shape == (3,)
    assert radius > 0


def test_ai_material_sampling_is_deterministic_and_continuous() -> None:
    first = patient_spec(7, 40, train_count=24, val_count=8, randomize_materials=True)
    repeated = patient_spec(
        7, 40, train_count=24, val_count=8, randomize_materials=True
    )
    neighbor = patient_spec(
        8, 40, train_count=24, val_count=8, randomize_materials=True
    )
    assert first == repeated
    assert 2200.0 <= first["E_background"] <= 8000.0
    assert first["E_background"] != neighbor["E_background"]
    assert first["inclusion_ratio"] == 1.0 or 1.3 <= first["inclusion_ratio"] <= 4.0


def test_interleaved_split_preserves_stage_ratios() -> None:
    splits = [
        patient_spec(index, 100, interleaved_split=True)["split"]
        for index in range(100)
    ]
    assert splits.count("train") == 60
    assert splits.count("val") == 20
    assert splits.count("test") == 20


def test_force_is_surface_only_and_resultant_is_recorded() -> None:
    scene, _, _, _ = build_patient(patient_spec(0, 10))
    forces, log = force_sequence(scene, EXPERIMENTS[0], max_force=10.0)
    surface = torch.unique(scene["surface_tris"].reshape(-1))
    nonzero_nodes = torch.where(
        torch.linalg.vector_norm(forces[2].view(-1, 3), dim=1) > 0
    )[0]
    assert torch.isin(nonzero_nodes, surface).all()
    assert abs(log[2]["resultant_force_N"] - 10.0) < 1e-8
    assert log[0]["resultant_force_N"] == 0.0


def test_viscoelastic_relaxation_lags_and_rebounds() -> None:
    equilibrium = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float64)
    motion = relaxation_sequence(equilibrium, tau_seconds=0.5, dt_seconds=0.1)
    assert 0.0 < motion[1, 0] < 1.0
    assert motion[2, 0] > motion[1, 0]
    assert 0.0 < motion[3, 0] < motion[2, 0]


def test_image_tracks_persist_intrinsics_and_visibility() -> None:
    scene, _, _, _ = build_patient(patient_spec(0, 10))
    surface_nodes = torch.unique(scene["surface_tris"].reshape(-1))
    poses = make_camera_poses(
        T=2,
        radius=1.55,
        height=1.0,
        look_at=tuple(float(value) for value in scene["nodes"].mean(dim=0)),
    )
    gt = {
        "nodes": scene["nodes"],
        "surface_tris": scene["surface_tris"],
        "surface_node_ids": surface_nodes,
        "u_seq": torch.zeros(2, 3 * scene["Nn"], dtype=scene["nodes"].dtype),
        "poses": poses,
    }
    fields = track_fields(gt, resolution=40)
    assert fields["image_uv_rest_seq"].shape == (2, len(surface_nodes), 2)
    assert torch.equal(
        fields["image_uv_rest_seq"], fields["image_uv_deformed_seq"]
    )
    assert fields["image_visibility_seq"].dtype == torch.bool
    assert (
        fields["image_gaussian_uv_rest_seq"].shape[1]
        > fields["image_uv_rest_seq"].shape[1]
    )
    assert fields["image_gaussian_visibility_seq"].dtype == torch.bool
    assert fields["render_intrinsics"]["width_px"] == 40


def test_multiview_camera_rings_are_synchronized_and_legacy_compatible() -> None:
    kwargs = {
        "T": 7,
        "radius": 1.6,
        "height": 1.2,
        "look_at": (0.5, 0.5, 0.0),
    }
    poses_multiview = make_multiview_camera_poses(**kwargs)
    legacy = make_camera_poses(**kwargs)
    assert poses_multiview.shape == (7, 3, 4, 4)
    assert torch.equal(poses_multiview[:, 1], legacy)
    assert not torch.equal(poses_multiview[:, 0], poses_multiview[:, 2])


def test_foreground_confidence_distinguishes_same_pixel_occlusion() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float64
    )
    pose = torch.eye(4, dtype=torch.float64)
    _, depth, in_frame, zbuffer, confidence = project_with_foreground_confidence(
        points, pose, focal=10.0, H=20, W=20, depth_softness=0.1
    )
    assert in_frame.all()
    assert torch.equal(depth, torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert torch.equal(zbuffer, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert confidence[0] == 1.0
    assert 0.0 < confidence[1] < 1e-3


def test_multiview_protocol_shapes_force_seed_and_qc() -> None:
    T, N, resolution = 7, 4, 32
    rest = torch.tensor(
        [
            [-0.10, -0.10, 1.0],
            [0.10, -0.10, 1.0],
            [-0.10, 0.10, 1.0],
            [0.10, 0.10, 1.0],
        ],
        dtype=torch.float64,
    )
    deformed = rest.unsqueeze(0).repeat(T, 1, 1)
    poses = torch.eye(4, dtype=torch.float64).repeat(T, 3, 1, 1)
    fields = multiview_projection_fields(
        rest, deformed, poses, focal=20.0, resolution=resolution
    )
    forces = torch.zeros(T, 3 * N, dtype=torch.float64)
    forces[:, 2] = torch.tensor(
        [0.0, 0.45, 1.0, 1.0, 0.65, 0.25, 0.0], dtype=torch.float64
    )
    measured = measured_force_fields(
        forces, true_force_scale_N=1.0, seed=90210, prior_fraction=0.05
    )
    repeated = measured_force_fields(
        forces, true_force_scale_N=1.0, seed=90210, prior_fraction=0.05
    )
    gaussian_fields = {
        key.replace("image_", "image_gaussian_", 1): value.clone()
        for key, value in fields.items()
    }
    assert torch.equal(measured["forces_measured"], repeated["forces_measured"])
    assert fields["image_uv_rest_multiview_seq"].shape == (T, 3, N, 2)
    assert fields["image_depth_deformed_multiview_seq"].shape == (T, 3, N)
    assert fields["image_foreground_confidence_deformed_multiview_seq"].shape == (
        T,
        3,
        N,
    )
    gt = {
        "schema_version": "sim_lung_v2_multiview_v1",
        "poses": poses[:, 1],
        "poses_multiview": poses,
        "num_views": 3,
        "camera_frame_index_multiview": torch.arange(T)[:, None].repeat(1, 3),
        "intrinsics_multiview": torch.eye(3, dtype=torch.float64).repeat(3, 1, 1),
        "surface_node_ids": torch.arange(N),
        "image_gaussian_host_tri": torch.arange(N),
        "forces": forces,
        "u_seq": torch.zeros(T, 3 * N, dtype=torch.float64),
        "motion_noise_seed": 123,
        "geometry_seed": 456,
        **fields,
        **gaussian_fields,
        **measured,
    }
    assert validate_experiment_protocol(gt)


def test_qc_accepts_legacy_single_view_schema() -> None:
    gt = {"schema_version": "sim_lung_v2"}
    assert validate_experiment_protocol(gt) is False
