"""Tests for the dependency-free lung graph learning core."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import (  # noqa: E402
    SimLungGraphDataset,
    collate_lung_graphs,
    tetrahedra_to_edge_index,
)
from models.lung_mesh_material_gnn import (  # noqa: E402
    GlobalFeatureMLP,
    MeshMaterialGNN,
    decode_predictions,
)
from experiments.train_lung_mesh_gnn import (  # noqa: E402
    material_loss,
    random_yaw_augmentation,
)


def _write_patient(
    root: Path,
    index: int,
    ratio: float,
    extra_node: bool,
    multiview: bool = False,
) -> dict:
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=torch.float64,
    )
    elems = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    if extra_node:
        elems = torch.tensor([[0, 1, 2, 4], [0, 2, 3, 4]], dtype=torch.long)
    else:
        nodes = nodes[:4]
    surface_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    patient_id = f"patient_{index}"
    experiments = []
    for experiment_index in range(4):
        force = torch.zeros((3, nodes.numel()), dtype=torch.float64)
        force[1, 3:6] = torch.tensor([0.0, 0.0, 2.0 + experiment_index])
        motion = torch.zeros((3, len(surface_ids), 3), dtype=torch.float64)
        motion[1, :, 2] = 0.01 * (experiment_index + 1)
        data = {
            "nodes": nodes,
            "elems": elems,
            "surface_node_ids": surface_ids,
            "surface_tris": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "fixed": torch.tensor([0, 1, 2], dtype=torch.long),
            "forces": force,
            "surface_motion_true": motion,
            "surface_motion_observed": motion + 0.001,
            "E_background": 4_000.0,
            "inclusion_ratio": ratio,
            "inclusion_center": torch.tensor([0.5, 0.5, 0.5]),
            "inclusion_radius": 0.3,
        }
        if multiview:
            view_offsets = torch.tensor(
                [[0.0, 0.0], [0.2, -0.1], [-0.15, 0.1]], dtype=torch.float32
            )
            rest_uv = (
                20.0 * nodes.to(torch.float32)[None, None, :, :2]
                + view_offsets[None, :, None, :]
            ).expand(3, -1, -1, -1).clone()
            time = torch.arange(3, dtype=torch.float32)[:, None, None, None]
            view_scale = torch.tensor([1.0, 0.8, 1.2])[None, :, None, None]
            deformed_uv = rest_uv + time * view_scale * torch.tensor([0.3, -0.2])
            depth_rest = nodes[:, 2].to(torch.float32)[None, None].expand(
                3, 3, -1
            ).clone()
            confidence = torch.ones((3, 3, len(nodes)), dtype=torch.float32)
            poses = torch.eye(4).repeat(3, 1, 1)
            poses[:, 0, 3] = torch.tensor([-1.0, 0.0, 1.0])
            data.update(
                {
                    "image_uv_rest_multiview_seq": rest_uv,
                    "image_uv_deformed_multiview_seq": deformed_uv,
                    "image_depth_rest_multiview_seq": depth_rest,
                    "image_depth_deformed_multiview_seq": depth_rest
                    + 0.01 * time.squeeze(-1),
                    "image_occlusion_confidence_multiview_seq": confidence,
                    "poses_multiview": poses,
                    "render_intrinsics": {"focal_px": 50.0},
                }
            )
        relative = Path(patient_id) / f"experiment_{experiment_index}" / "gt.pt"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, root / relative)
        experiments.append(
            {
                "name": f"experiment_{experiment_index}",
                "relative_path": str(relative),
            }
        )
    return {
        "patient_id": patient_id,
        "split": "train",
        "E_background": 4_000.0,
        "inclusion_ratio": ratio,
        "inclusion_center_fraction": [0.5, 0.5, 0.5],
        "inclusion_radius_fraction": 0.3,
        "experiments": experiments,
    }


def _make_dataset(tmp_path: Path) -> SimLungGraphDataset:
    manifest = {
        "version": "test",
        "patients": [
            _write_patient(tmp_path, 0, ratio=1.0, extra_node=False),
            _write_patient(tmp_path, 1, ratio=2.0, extra_node=True),
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return SimLungGraphDataset(tmp_path, split="train")


def _make_multiview_dataset(tmp_path: Path, **kwargs: object) -> SimLungGraphDataset:
    manifest = {
        "version": "multiview-test",
        "patients": [
            _write_patient(
                tmp_path, 0, ratio=2.0, extra_node=True, multiview=True
            )
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return SimLungGraphDataset(tmp_path, split="train", **kwargs)


def test_tetra_edges_are_unique_and_bidirectional() -> None:
    edge_index = tetrahedra_to_edge_index(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    )
    edges = {tuple(edge) for edge in edge_index.t().tolist()}
    assert edge_index.shape == (2, 12)
    assert len(edges) == edge_index.shape[1]
    assert all((target, source) in edges for source, target in edges)


def test_dataset_shapes_limits_and_homogeneous_region_semantics(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    graph = dataset[0]
    assert graph["x"].dtype == torch.float32
    assert graph["x"].shape == (4, 5 + 4 * 6)
    assert graph["peak_frames"].tolist() == [1, 1, 1, 1]
    assert not graph["labels"]["region_mask"].any()
    assert not bool(graph["labels"]["heterogeneous"])
    assert torch.allclose(
        graph["labels"]["E_nodes_normalized_log"],
        torch.full_like(
            graph["labels"]["E_nodes_normalized_log"],
            (math.log(4_000.0) - math.log(1_000.0))
            / (math.log(75_000.0) - math.log(1_000.0)),
        ),
    )
    # Node 3 is internal: its four observed-motion feature triplets are zero.
    for start in range(5 + 3, graph["x"].shape[1], 6):
        assert torch.equal(graph["x"][3, start : start + 3], torch.zeros(3))

    limited = SimLungGraphDataset(
        tmp_path, observation_key="oracle", experiments_limit=2
    )[0]
    assert limited["x"].shape[1] == 5 + 2 * 6
    assert limited["experiment_names"] == ["experiment_0", "experiment_1"]


def test_variable_graph_batch_models_and_backward(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    batch = collate_lung_graphs([dataset[0], dataset[1]])
    assert [graph["x"].shape[0] for graph in batch] == [4, 5]

    model = MeshMaterialGNN(input_dim=29, hidden_dim=24, num_layers=2)
    output = model(batch)
    assert output["log_E_background_mean"].shape == (2, 1)
    assert output["center_fraction_mean"].shape == (2, 3)
    assert output["heterogeneity_logit"].shape == (2, 1)
    assert [value.shape for value in output["node_log_E"]] == [(4,), (5,)]
    decoded = decode_predictions(output)
    assert torch.all((decoded["inclusion_ratio"] >= 1.0) & (decoded["inclusion_ratio"] <= 5.0))
    loss = (
        output["log_E_background_mean"].sum()
        + output["heterogeneity_logit"].sum()
        + sum(value.sum() for value in output["node_log_E"])
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    baseline = GlobalFeatureMLP(input_dim=29, hidden_dim=16)
    baseline_output = baseline(batch)
    baseline_loss = baseline_output["radius_fraction_mean"].sum()
    baseline_loss.backward()
    assert baseline_output["center_fraction_logvar"].shape == (2, 3)


def test_pre_sdf_checkpoint_state_is_backward_compatible() -> None:
    for model in (
        MeshMaterialGNN(input_dim=29, hidden_dim=16, num_layers=1),
        GlobalFeatureMLP(input_dim=29, hidden_dim=16),
    ):
        legacy_state = {
            name: value
            for name, value in model.state_dict().items()
            if "node_sdf_head" not in name and "partition_head" not in name
        }
        restored = type(model)(input_dim=29, hidden_dim=16, num_layers=1)
        restored.load_state_dict(legacy_state)


def test_yaw_augmentation_preserves_vector_norms_and_masks(tmp_path: Path) -> None:
    graph = _make_dataset(tmp_path)[0]
    torch.manual_seed(7)
    augmented = random_yaw_augmentation(graph)
    assert torch.equal(graph["x"][:, 3:5], augmented["x"][:, 3:5])
    for start in range(5, graph["x"].shape[1], 6):
        assert torch.allclose(
            graph["x"][:, start : start + 3].norm(dim=1),
            augmented["x"][:, start : start + 3].norm(dim=1),
            atol=1e-6,
        )
    assert torch.allclose(
        (graph["labels"]["center_fraction"][:2] - 0.5).norm(),
        (augmented["labels"]["center_fraction"][:2] - 0.5).norm(),
    )


def test_multiview_sequence_shapes_masks_and_ablations(tmp_path: Path) -> None:
    graph = _make_multiview_dataset(tmp_path)[0]
    assert graph["x"].shape == (5, 5)
    assert graph["static_x"].shape == (5, 5)
    assert graph["dynamic_seq"].shape == (4, 3, 7, 5, 14)
    assert graph["temporal_frame_indices"].shape == (4, 7)
    assert torch.all((graph["dynamic_seq"][..., 3] >= 0.0))
    assert torch.all((graph["dynamic_seq"][..., 4] <= 1.0))

    single = _make_multiview_dataset(tmp_path, single_view=True)[0]
    assert single["dynamic_seq"].shape[1] == 1
    ablated = _make_multiview_dataset(
        tmp_path, no_depth=True, no_confidence=True, peak_only=True
    )[0]
    assert not ablated["dynamic_seq"][..., 2].any()
    assert torch.equal(
        ablated["dynamic_seq"][:, :, 0],
        ablated["dynamic_seq"][:, :, -1],
    )


def test_temporal_model_permutation_missing_view_sdf_and_backward(
    tmp_path: Path,
) -> None:
    graph = _make_multiview_dataset(tmp_path)[0]
    model = MeshMaterialGNN(
        input_dim=5, dynamic_dim=14, hidden_dim=24, num_layers=2
    ).eval()
    with torch.no_grad():
        reference = model(graph)
        permuted = {**graph, "dynamic_seq": graph["dynamic_seq"][:, [2, 0, 1]]}
        permutation_output = model(permuted)
        assert torch.allclose(
            reference["node_sdf_mean"],
            permutation_output["node_sdf_mean"],
            atol=2e-6,
        )

        masked_dynamic = graph["dynamic_seq"].clone()
        masked_dynamic[:, 2, ..., 3:5] = 0.0
        masked_dynamic[:, 2, ..., :3] = 1e4
        masked = model({**graph, "dynamic_seq": masked_dynamic})
        removed = model({**graph, "dynamic_seq": graph["dynamic_seq"][:, :2]})
        assert torch.allclose(
            masked["node_log_E"], removed["node_log_E"], atol=2e-6
        )
        assert torch.isfinite(reference["node_sdf_mean"]).all()
        assert torch.isfinite(reference["node_sdf_logvar"]).all()
        assert reference["partition_logits"].shape == (5,)
        assert reference["center_fraction_mean"].shape == (1, 3)
        assert reference["radius_fraction_mean"].shape == (1, 1)

    model.train()
    output = model([graph])
    loss, terms = material_loss(output, [graph])
    loss.backward()
    assert all(math.isfinite(value) for value in terms.values())
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_image_mode_yaw_is_not_applied_in_3d(tmp_path: Path) -> None:
    graph = _make_multiview_dataset(tmp_path)[0]
    augmented = random_yaw_augmentation(graph)
    assert augmented is graph
