"""Pure-PyTorch graph dataset for the patient-level lung benchmark."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def tetrahedra_to_edge_index(elems: torch.Tensor) -> torch.Tensor:
    """Return unique directed edges for every undirected tetrahedral edge."""
    elems = torch.as_tensor(elems, dtype=torch.long)
    if elems.ndim != 2 or elems.shape[1] != 4:
        raise ValueError("elems must have shape (num_tetrahedra, 4)")
    pairs = elems[:, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    pairs = pairs.reshape(-1, 2)
    undirected = torch.sort(pairs, dim=1).values
    undirected = torch.unique(undirected, dim=0)
    directed = torch.cat((undirected, undirected.flip(1)), dim=0)
    return directed.t().contiguous()


# Short alias useful in training scripts.
build_edge_index = tetrahedra_to_edge_index


def collate_lung_graphs(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep variable-size patient graphs as a list batch."""
    return graphs


# Conventional name accepted directly by torch.utils.data.DataLoader.
graph_collate_fn = collate_lung_graphs


class SimLungGraphDataset(Dataset):
    """Load one graph per patient and concatenate its load experiments.

    Node features are normalized position, fixed/surface masks, and, for each
    experiment, the peak-frame nodal force and observed displacement. Motion
    outside the observed surface is represented by exact zeros.
    """

    OBSERVATION_ALIASES = {
        "oracle": "surface_motion_true",
        "noisy": "surface_motion_observed",
        "image_tracks": "image_tracks",
    }

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        observation_key: str = "surface_motion_observed",
        experiments_limit: int | None = None,
        track_noise_px: float = 0.0,
        seed: int = 2026,
        single_view: bool = False,
        no_depth: bool = False,
        no_confidence: bool = False,
        peak_only: bool = False,
        sequence_length: int = 7,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root if self.root.is_file() else self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.observation_key = self.OBSERVATION_ALIASES.get(
            observation_key, observation_key
        )
        if experiments_limit is not None and experiments_limit <= 0:
            raise ValueError("experiments_limit must be positive")
        self.experiments_limit = experiments_limit
        self.track_noise_px = track_noise_px
        self.seed = seed
        if sequence_length != 7:
            raise ValueError("The temporal MeshGNN requires exactly seven frames")
        self.single_view = single_view
        self.no_depth = no_depth
        self.no_confidence = no_confidence
        self.peak_only = peak_only
        self.sequence_length = sequence_length
        patients = self.manifest.get("patients", [])
        self.patients = [
            patient for patient in patients if split is None or patient.get("split") == split
        ]

    def __len__(self) -> int:
        return len(self.patients)

    @staticmethod
    def _peak_frame(forces: torch.Tensor, node_count: int) -> int:
        nodal = forces.reshape(forces.shape[0], node_count, 3)
        resultant = nodal.sum(dim=1).norm(dim=1)
        if float(resultant.max()) <= 0.0:
            resultant = nodal.square().sum(dim=(1, 2)).sqrt()
        return int(resultant.argmax())

    @staticmethod
    def _check_consistency(reference: dict[str, Any], current: dict[str, Any]) -> None:
        for key in ("nodes", "elems", "surface_node_ids", "fixed"):
            if not torch.equal(reference[key], current[key]):
                raise ValueError(f"Patient graph changes between experiments: {key}")

    @staticmethod
    def _seven_frame_indices(frame_count: int, peak: int) -> torch.Tensor:
        if frame_count <= 0:
            raise ValueError("Temporal observations must contain at least one frame")
        offsets = torch.arange(-3, 4)
        return (offsets + peak).clamp(0, frame_count - 1).to(torch.long)

    @staticmethod
    def _pose_features(
        poses: Any, indices: torch.Tensor, views: int
    ) -> torch.Tensor:
        """Encode camera translation and optical axis as six pose channels."""
        pose = torch.as_tensor(poses, dtype=torch.float32)
        if pose.ndim == 3:
            pose = pose.unsqueeze(0).expand(len(indices), -1, -1, -1)
        elif pose.ndim == 2:
            pose = pose.view(1, 1, *pose.shape).expand(len(indices), views, -1, -1)
        elif pose.ndim != 4:
            return torch.zeros((len(indices), views, 6), dtype=torch.float32)
        else:
            pose = pose[indices.clamp_max(pose.shape[0] - 1)]
        if pose.shape[1] < views:
            padding = pose[:, -1:].expand(-1, views - pose.shape[1], -1, -1)
            pose = torch.cat((pose, padding), dim=1)
        pose = pose[:, :views]
        translation = pose[..., :3, 3] if pose.shape[-1] >= 4 else pose.new_zeros((*pose.shape[:2], 3))
        optical_axis = pose[..., :3, 2]
        translation = translation / translation.norm(dim=-1, keepdim=True).clamp_min(1.0)
        optical_axis = optical_axis / optical_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat((translation, optical_axis), dim=-1)

    def _multiview_dynamic(
        self,
        experiment: dict[str, Any],
        forces: torch.Tensor,
        peak: int,
        node_count: int,
        surface_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rest = experiment["image_uv_rest_multiview_seq"].to(torch.float32)
        deformed = experiment["image_uv_deformed_multiview_seq"].to(torch.float32)
        if rest.ndim != 4 or rest.shape[-1] != 2:
            raise ValueError("multiview UV fields must have shape (T,V,N,2)")
        indices = self._seven_frame_indices(rest.shape[0], peak)
        rest, deformed = rest[indices], deformed[indices]
        views = min(3, rest.shape[1])
        if self.single_view:
            views = 1
        rest, deformed = rest[:, :views], deformed[:, :views]
        focal = float(experiment.get("render_intrinsics", {}).get("focal_px", 100.0))
        flow = 100.0 * (deformed - rest) / max(focal, 1e-6)

        depth_delta = torch.zeros((*flow.shape[:-1], 1), dtype=torch.float32)
        depth_keys = (
            "image_depth_rest_multiview_seq",
            "image_depth_deformed_multiview_seq",
        )
        if not self.no_depth and all(key in experiment for key in depth_keys):
            depth_rest = experiment[depth_keys[0]].to(torch.float32)[indices, :views]
            depth_deformed = experiment[depth_keys[1]].to(torch.float32)[indices, :views]
            if depth_rest.ndim == 3:
                depth_rest = depth_rest.unsqueeze(-1)
                depth_deformed = depth_deformed.unsqueeze(-1)
            depth_scale = depth_rest.abs().median().clamp_min(1e-6)
            depth_delta = (depth_deformed - depth_rest) / depth_scale

        confidence = torch.ones((*flow.shape[:-1], 1), dtype=torch.float32)
        confidence_key = "image_occlusion_confidence_multiview_seq"
        if not self.no_confidence and confidence_key in experiment:
            confidence = experiment[confidence_key].to(torch.float32)[indices, :views]
            if confidence.ndim == 3:
                confidence = confidence.unsqueeze(-1)
            confidence = confidence.clamp(0.0, 1.0)
        finite = torch.isfinite(flow).all(dim=-1, keepdim=True)
        if not self.no_depth:
            finite = finite & torch.isfinite(depth_delta)
        visibility = finite & (confidence > 0.0)
        flow = torch.nan_to_num(flow) * visibility
        depth_delta = torch.nan_to_num(depth_delta) * visibility
        if self.no_confidence:
            confidence = torch.ones_like(confidence)
        confidence = confidence * visibility
        if flow.shape[-2] != node_count:
            if flow.shape[-2] != len(surface_ids):
                raise ValueError(
                    "Multiview track count must match all nodes or surface_node_ids"
                )
            full_shape = (*flow.shape[:-2], node_count)
            flow_full = torch.zeros((*full_shape, 2), dtype=flow.dtype)
            depth_full = torch.zeros((*full_shape, 1), dtype=depth_delta.dtype)
            confidence_full = torch.zeros((*full_shape, 1), dtype=confidence.dtype)
            visibility_full = torch.zeros((*full_shape, 1), dtype=torch.bool)
            flow_full[..., surface_ids, :] = flow
            depth_full[..., surface_ids, :] = depth_delta
            confidence_full[..., surface_ids, :] = confidence
            visibility_full[..., surface_ids, :] = visibility
            flow, depth_delta = flow_full, depth_full
            confidence, visibility = confidence_full, visibility_full

        frame_force = forces.reshape(forces.shape[0], node_count, 3)[indices]
        force_feature = torch.sign(frame_force) * torch.log1p(frame_force.abs())
        force_feature = force_feature[:, None].expand(-1, views, -1, -1)
        poses = self._pose_features(
            experiment.get("poses_multiview", torch.eye(4).repeat(views, 1, 1)),
            indices,
            views,
        )
        pose_feature = poses[:, :, None].expand(-1, -1, node_count, -1)
        dynamic = torch.cat(
            (
                flow,
                depth_delta,
                confidence,
                visibility.to(torch.float32),
                force_feature,
                pose_feature,
            ),
            dim=-1,
        )
        if self.peak_only:
            peak_dynamic = dynamic[3:4]
            dynamic = peak_dynamic.expand(self.sequence_length, -1, -1, -1).clone()
        # Return view-major order: (V,7,N,C).
        return dynamic.permute(1, 0, 2, 3).contiguous(), indices

    def __getitem__(self, index: int) -> dict[str, Any]:
        patient = self.patients[index]
        experiment_rows = patient.get("experiments", [])
        if self.experiments_limit is not None:
            experiment_rows = experiment_rows[: self.experiments_limit]
        if not experiment_rows:
            raise ValueError(f"Patient {patient.get('patient_id')} has no experiments")

        experiments = [
            torch.load(
                self.root / row["relative_path"],
                map_location="cpu",
                weights_only=False,
            )
            for row in experiment_rows
        ]
        reference = experiments[0]
        for experiment in experiments[1:]:
            self._check_consistency(reference, experiment)

        nodes = reference["nodes"].to(torch.float32)
        node_count = nodes.shape[0]
        minimum = nodes.amin(dim=0)
        extent = (nodes.amax(dim=0) - minimum).clamp_min(torch.finfo(nodes.dtype).eps)
        coordinates = (nodes - minimum) / extent

        fixed_mask = torch.zeros(node_count, dtype=torch.float32)
        fixed_nodes = torch.unique(reference["fixed"].to(torch.long) // 3)
        fixed_mask[fixed_nodes] = 1.0
        surface_ids = reference["surface_node_ids"].to(torch.long)
        surface_mask = torch.zeros(node_count, dtype=torch.float32)
        surface_mask[surface_ids] = 1.0

        static_x = torch.cat(
            (coordinates, fixed_mask[:, None], surface_mask[:, None]), dim=1
        ).to(torch.float32)
        features = [coordinates, fixed_mask[:, None], surface_mask[:, None]]
        peak_frames: list[int] = []
        peak_forces: list[torch.Tensor] = []
        peak_surface_observations: list[torch.Tensor] = []
        dynamic_rows: list[torch.Tensor] = []
        temporal_indices: list[torch.Tensor] = []
        has_multiview = all(
            "image_uv_rest_multiview_seq" in experiment
            and "image_uv_deformed_multiview_seq" in experiment
            for experiment in experiments
        )
        for experiment in experiments:
            force_key = (
                "measured_forces"
                if "measured_forces" in experiment
                else "forces_measured"
                if "forces_measured" in experiment
                else "forces"
            )
            forces = experiment[force_key].to(torch.float32)
            peak = self._peak_frame(forces, node_count)
            peak_frames.append(peak)
            if has_multiview:
                dynamic, frame_indices = self._multiview_dynamic(
                    experiment, forces, peak, node_count, surface_ids
                )
                dynamic_rows.append(dynamic)
                temporal_indices.append(frame_indices)
                peak_forces.append(forces[peak].to(torch.float64))
                continue
            nodal_force = forces[peak].reshape(node_count, 3)
            force_feature = torch.sign(nodal_force) * torch.log1p(nodal_force.abs())
            if self.observation_key == "image_tracks":
                frame_force = forces.reshape(forces.shape[0], node_count, 3)
                frame_magnitude = frame_force.sum(dim=1).norm(dim=1)
                hold_frames = torch.where(
                    frame_magnitude >= 0.95 * frame_magnitude.max()
                )[0].tolist()
                flow_rows, visibility_rows = [], []
                for frame_index in hold_frames:
                    rest_uv = experiment["image_uv_rest_seq"][frame_index].to(
                        torch.float32
                    )
                    deformed_uv = experiment["image_uv_deformed_seq"][
                        frame_index
                    ].to(torch.float32)
                    visibility_rows.append(
                        experiment["image_visibility_seq"][frame_index]
                    )
                    generator = torch.Generator().manual_seed(
                        self.seed
                        + 1009 * index
                        + 31 * len(peak_frames)
                        + frame_index
                    )
                    noise = self.track_noise_px * torch.randn(
                        deformed_uv.shape,
                        generator=generator,
                        dtype=deformed_uv.dtype,
                    )
                    flow_rows.append(deformed_uv + noise - rest_uv)
                visibility = torch.stack(visibility_rows).all(dim=0).to(
                    torch.float32
                )
                focal = float(experiment["render_intrinsics"]["focal_px"])
                flow = 100.0 * torch.stack(flow_rows).mean(dim=0) / focal
                observed_surface_motion = torch.cat(
                    (flow, visibility[:, None]), dim=1
                )
            else:
                observed_surface_motion = experiment[self.observation_key][peak].to(
                    torch.float32
                )
                observed_surface_motion = (
                    100.0 * observed_surface_motion / extent[None, :]
                )
            if observed_surface_motion.shape != (surface_ids.numel(), 3):
                raise ValueError(
                    f"{self.observation_key} must contain surface-node 3D motion"
                )
            motion = torch.zeros((node_count, 3), dtype=torch.float32)
            motion[surface_ids] = observed_surface_motion
            features.extend((force_feature, motion))
            peak_forces.append(forces[peak].to(torch.float64))
            if self.observation_key != "image_tracks":
                peak_surface_observations.append(
                    experiment[self.observation_key][peak].to(torch.float64)
                )

        ratio = float(patient.get("inclusion_ratio", reference["inclusion_ratio"]))
        background = float(patient.get("E_background", reference["E_background"]))
        center_fraction = torch.as_tensor(
            patient.get("inclusion_center_fraction"), dtype=torch.float32
        )
        if center_fraction.shape != (3,):
            center = reference["inclusion_center"].to(torch.float32)
            center_fraction = (center - minimum) / extent
        radius_fraction = float(
            patient.get(
                "inclusion_radius_fraction",
                float(reference["inclusion_radius"]) / float(extent[:2].min()),
            )
        )
        heterogeneous = ratio > 1.05
        physical_center = minimum + center_fraction * extent
        physical_radius = radius_fraction * float(extent[:2].min())
        distance = torch.linalg.vector_norm(nodes - physical_center, dim=1)
        # A homogeneous sample has no identifiable inclusion region even when
        # the generator persisted otherwise meaningful geometric parameters.
        region_mask = (distance <= physical_radius) & heterogeneous
        node_sdf = (
            distance / float(extent[:2].min()) - radius_fraction
        ).to(torch.float32)
        E_nodes = torch.full((node_count,), background, dtype=torch.float32)
        E_nodes[region_mask] = background * ratio
        log_E_nodes = E_nodes.log()
        log_E_min, log_E_max = math.log(1_000.0), math.log(75_000.0)
        E_nodes_normalized_log = (
            (log_E_nodes - log_E_min) / (log_E_max - log_E_min)
        ).clamp(0.0, 1.0)

        labels = {
            "log_E_background": torch.tensor(math.log(background), dtype=torch.float32),
            "log_ratio": torch.tensor(math.log(ratio), dtype=torch.float32),
            "center_fraction": center_fraction.to(torch.float32),
            "radius_fraction": torch.tensor(radius_fraction, dtype=torch.float32),
            "heterogeneous": torch.tensor(float(heterogeneous), dtype=torch.float32),
            "E_nodes_normalized_log": E_nodes_normalized_log,
            "E_nodes_log_normalized": E_nodes_normalized_log,
            "node_log_E": log_E_nodes,
            "region_mask": region_mask,
            "node_sdf": node_sdf,
            "partition": region_mask.to(torch.float32),
            "geometry_mask": torch.tensor(
                float(heterogeneous), dtype=torch.float32
            ),
        }
        dynamic_seq = (
            torch.stack(dynamic_rows).to(torch.float32) if dynamic_rows else None
        )
        x = static_x if dynamic_seq is not None else torch.cat(features, dim=1).to(torch.float32)
        return {
            "patient_id": patient["patient_id"],
            "split": patient.get("split"),
            "x": x,
            "static_x": static_x,
            "dynamic_seq": dynamic_seq,
            "dynamic_layout": {
                "flow": (0, 2),
                "depth_delta": (2, 3),
                "confidence": (3, 4),
                "visibility": (4, 5),
                "force": (5, 8),
                "pose": (8, 14),
            },
            "observation_mode": (
                "multiview_sequence"
                if dynamic_seq is not None
                else self.observation_key
            ),
            "pos": coordinates,
            "edge_index": tetrahedra_to_edge_index(reference["elems"]),
            "labels": labels,
            "y": labels,
            "peak_frames": torch.tensor(peak_frames, dtype=torch.long),
            "temporal_frame_indices": (
                torch.stack(temporal_indices)
                if temporal_indices
                else None
            ),
            "experiment_names": [row["name"] for row in experiment_rows],
            "physics": {
                "nodes": reference["nodes"].to(torch.float64),
                "elems": reference["elems"],
                "fixed": reference["fixed"],
                "surface_node_ids": surface_ids,
                "nu": torch.tensor(
                    float(patient.get("nu", 0.45)), dtype=torch.float64
                ),
                "forces": torch.stack(peak_forces),
                "surface_observations": (
                    torch.stack(peak_surface_observations)
                    if peak_surface_observations
                    else None
                ),
            },
        }


# Backwards-friendly concise name.
LungGraphDataset = SimLungGraphDataset
