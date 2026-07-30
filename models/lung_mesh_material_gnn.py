"""Pure-PyTorch graph models for lung material-field inference."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


LOG_E_BOUNDS = (math.log(1_000.0), math.log(15_000.0))
LOG_RATIO_BOUNDS = (math.log(1.0), math.log(5.0))
RADIUS_BOUNDS = (0.05, 0.40)
LOGVAR_BOUNDS = (-10.0, 5.0)
NODE_LOG_E_BOUNDS = (math.log(1_000.0), math.log(75_000.0))


def _bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def _masked_softmax(
    scores: torch.Tensor, mask: torch.Tensor, dim: int
) -> torch.Tensor:
    mask = mask.to(torch.bool)
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim) * mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class IndexAddMessageLayer(nn.Module):
    """Residual message passing without torch-geometric dependencies."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index.to(torch.long)
        messages = self.message(torch.cat((h[source], h[target]), dim=-1)).to(
            h.dtype
        )
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, target, messages)
        degree = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=h.dtype))
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        update = self.update(torch.cat((h, aggregated), dim=-1))
        return self.norm(h + update)


class MultiViewTemporalEncoder(nn.Module):
    """Permutation-invariant load/view fusion with seven-frame attention."""

    def __init__(self, dynamic_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        heads = next(value for value in (4, 2, 1) if hidden_dim % value == 0)
        self.dynamic_dim = dynamic_dim
        self.token_projection = nn.Linear(dynamic_dim, hidden_dim)
        self.temporal_position = nn.Parameter(torch.zeros(7, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.view_score = nn.Linear(hidden_dim, 1)
        self.load_score = nn.Linear(hidden_dim, 1)
        self.summary_projection = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.view_summary_projection = nn.Sequential(
            nn.Linear(60, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_summary_projection = nn.Sequential(
            nn.Linear(24, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, dynamic: torch.Tensor) -> torch.Tensor:
        if dynamic.ndim != 5 or dynamic.shape[2] != 7:
            raise ValueError("dynamic_seq must have shape (loads,views,7,nodes,channels)")
        if dynamic.shape[-1] != self.dynamic_dim:
            raise ValueError(f"dynamic_seq requires {self.dynamic_dim} channels")
        loads, views, frames, nodes, _ = dynamic.shape
        tokens = dynamic.permute(3, 0, 1, 2, 4).reshape(
            nodes * loads * views, frames, self.dynamic_dim
        )
        confidence = tokens[..., 3].clamp(0.0, 1.0)
        visibility = tokens[..., 4] > 0.5
        valid = visibility & (confidence > 0.0)
        safe_valid = valid.clone()
        all_missing = ~safe_valid.any(dim=1)
        safe_valid[all_missing, 0] = True
        h = self.token_projection(tokens) + self.temporal_position[None]
        h = self.temporal(h, src_key_padding_mask=~safe_valid)
        temporal_weights = confidence * valid.to(confidence.dtype)
        temporal_weights[all_missing, 0] = 1.0
        temporal = (h * temporal_weights.unsqueeze(-1)).sum(dim=1)
        temporal = temporal / temporal_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        temporal = temporal.reshape(nodes, loads, views, -1)

        view_confidence = confidence.reshape(nodes, loads, views, frames).amax(dim=-1)
        view_valid = valid.reshape(nodes, loads, views, frames).any(dim=-1)
        view_scores = self.view_score(temporal).squeeze(-1)
        view_scores = view_scores + torch.log(view_confidence.clamp_min(1e-6))
        view_weights = _masked_softmax(view_scores, view_valid, dim=2)
        per_load = (temporal * view_weights.unsqueeze(-1)).sum(dim=2)

        load_valid = view_valid.any(dim=2)
        load_scores = self.load_score(per_load).squeeze(-1)
        load_weights = _masked_softmax(load_scores, load_valid, dim=1)
        fused = (per_load * load_weights.unsqueeze(-1)).sum(dim=1)
        flow_magnitude = torch.linalg.vector_norm(dynamic[..., :3], dim=-1)
        decoded_force = torch.sign(dynamic[..., 5:8]) * torch.expm1(
            dynamic[..., 5:8].abs()
        )
        valid_float = (dynamic[..., 4] > 0.5).to(dynamic.dtype)
        flow_mean = (flow_magnitude * valid_float).sum(dim=(1, 2, 3))
        flow_mean = flow_mean / valid_float.sum(dim=(1, 2, 3)).clamp_min(1.0)
        masked_flow = flow_magnitude.masked_fill(valid_float <= 0, 0.0)
        flow_max = masked_flow.amax(dim=(1, 2, 3))
        force_resultant = decoded_force.sum(dim=3)
        force_max = torch.linalg.vector_norm(force_resultant, dim=-1).amax(dim=(1, 2))
        compliance = flow_mean / force_max.clamp_min(1e-6)
        per_load_summary = torch.stack(
            (flow_mean, flow_max, force_max, compliance), dim=-1
        )
        summary = torch.cat(
            (per_load_summary.mean(dim=0), per_load_summary.amax(dim=0)), dim=0
        )
        summary_embedding = self.summary_projection(
            torch.sign(summary) * torch.log1p(summary.abs())
        )
        # Preserve load/view-specific deformation signatures. Views are sorted
        # by camera azimuth so this remains invariant to input view ordering.
        camera_xy = dynamic[..., 8:10].mean(dim=(0, 2, 3))
        view_active = valid_float.any(dim=(0, 2, 3))
        fallback_key = (
            (flow_magnitude * valid_float).sum(dim=(0, 2, 3))
            / valid_float.sum(dim=(0, 2, 3)).clamp_min(1.0)
        )
        camera_radius = torch.linalg.vector_norm(camera_xy, dim=1)
        view_key = torch.where(
            camera_radius > 1e-6,
            torch.atan2(camera_xy[:, 1], camera_xy[:, 0]),
            fallback_key,
        )
        view_key = view_key.masked_fill(~view_active, float("inf"))
        view_order = torch.argsort(view_key)
        canonical_flow = flow_magnitude[:, view_order]
        canonical_valid = valid_float[:, view_order]
        flat_flow = canonical_flow.flatten(2)
        flat_valid = canonical_valid.flatten(2)
        count = flat_valid.sum(dim=-1).clamp_min(1.0)
        mean = (flat_flow * flat_valid).sum(dim=-1) / count
        variance = (
            (flat_flow - mean[..., None]).square() * flat_valid
        ).sum(dim=-1) / count
        maximum = flat_flow.masked_fill(flat_valid <= 0, 0.0).amax(dim=-1)
        quantile_input = flat_flow.masked_fill(flat_valid <= 0, float("nan"))
        median = torch.nanquantile(quantile_input, 0.5, dim=-1).nan_to_num()
        upper = torch.nanquantile(quantile_input, 0.9, dim=-1).nan_to_num()
        view_summary = torch.stack(
            (mean, variance.sqrt(), median, upper, maximum), dim=-1
        )
        padded = dynamic.new_zeros((4, 3, 5))
        padded[: min(loads, 4), : min(views, 3)] = view_summary[:4, :3]
        view_embedding = self.view_summary_projection(
            torch.log1p(padded.flatten())
        )
        node_flow = flow_magnitude[:, view_order]
        node_valid = valid_float[:, view_order]
        node_count_valid = node_valid.sum(dim=2).clamp_min(1.0)
        node_mean = (node_flow * node_valid).sum(dim=2) / node_count_valid
        node_maximum = node_flow.masked_fill(node_valid <= 0, 0.0).amax(dim=2)
        node_statistics = torch.stack((node_mean, node_maximum), dim=-1)
        node_padded = dynamic.new_zeros((4, 3, nodes, 2))
        node_padded[: min(loads, 4), : min(views, 3)] = node_statistics[:4, :3]
        node_features = node_padded.permute(2, 0, 1, 3).reshape(nodes, 24)
        node_embedding = self.node_summary_projection(torch.log1p(node_features))
        return self.output_norm(
            fused + node_embedding + summary_embedding[None] + view_embedding[None]
        )


class _MaterialHeads(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                "log_E_background": nn.Linear(hidden_dim, 2),
                "log_ratio": nn.Linear(hidden_dim, 2),
                "center_fraction": nn.Linear(hidden_dim, 6),
                "radius_fraction": nn.Linear(hidden_dim, 2),
            }
        )
        self.heterogeneity = nn.Linear(hidden_dim, 1)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        bounds = {
            "log_E_background": LOG_E_BOUNDS,
            "log_ratio": LOG_RATIO_BOUNDS,
            "center_fraction": (0.0, 1.0),
            "radius_fraction": RADIUS_BOUNDS,
        }
        dimensions = {
            "log_E_background": 1,
            "log_ratio": 1,
            "center_fraction": 3,
            "radius_fraction": 1,
        }
        for name, head in self.heads.items():
            dimension = dimensions[name]
            raw = head(pooled)
            lower, upper = bounds[name]
            output[f"{name}_mean"] = _bounded(raw[:, :dimension], lower, upper)
            output[f"{name}_logvar"] = _bounded(
                raw[:, dimension:], *LOGVAR_BOUNDS
            )
        output["heterogeneity_logit"] = self.heterogeneity(pooled)
        return output


def _merge_graph_outputs(
    outputs: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    merged: dict[str, torch.Tensor | list[torch.Tensor]] = {}
    for key in outputs[0]:
        values = [output[key] for output in outputs]
        if key.startswith("node_") or key == "partition_logits":
            merged[key] = values
        else:
            merged[key] = torch.cat(values, dim=0)
    return merged


class MeshMaterialGNN(nn.Module):
    """Residual mesh GNN with global probabilistic and node-field heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        dynamic_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        self.input_dim = input_dim
        self.dynamic_dim = dynamic_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            IndexAddMessageLayer(hidden_dim) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.temporal_encoder = (
            MultiViewTemporalEncoder(dynamic_dim, hidden_dim, dropout)
            if dynamic_dim is not None
            else None
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.material_heads = _MaterialHeads(hidden_dim)
        self.node_head = nn.Linear(hidden_dim, 2)
        self.node_sdf_head = nn.Linear(hidden_dim, 2)
        self.partition_head = nn.Linear(hidden_dim, 1)
        self.ratio_refinement = nn.Linear(3 * hidden_dim, 2)

    def _forward_one(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        dynamic_seq: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must have shape (nodes, {self.input_dim})")
        model_dtype = self.encoder[0].weight.dtype
        h = self.encoder(x.to(dtype=model_dtype))
        if dynamic_seq is not None:
            if self.temporal_encoder is None:
                raise ValueError("Model was not configured with dynamic_dim")
            h = h + self.temporal_encoder(dynamic_seq.to(dtype=model_dtype))
        for layer in self.layers:
            h = self.dropout(layer(h, edge_index))
        pooled = self.pool_projection(torch.cat((h.mean(dim=0), h.amax(dim=0)))).unsqueeze(0)
        output = self.material_heads(pooled)
        node_raw = self.node_head(h)
        output["node_log_E"] = _bounded(
            node_raw[:, 0], *NODE_LOG_E_BOUNDS
        )
        output["node_log_E_logvar"] = _bounded(
            node_raw[:, 1], *LOGVAR_BOUNDS
        )
        sdf_raw = self.node_sdf_head(h)
        output["node_sdf_mean"] = 2.0 * torch.tanh(sdf_raw[:, 0])
        output["node_sdf_logvar"] = _bounded(sdf_raw[:, 1], *LOGVAR_BOUNDS)
        output["partition_logits"] = self.partition_head(h).squeeze(-1)
        if pos is not None:
            pos = pos.to(dtype=h.dtype)
            center_global = output["center_fraction_mean"].squeeze(0)
            radius_global = output["radius_fraction_mean"].squeeze()
            distance_global = torch.linalg.vector_norm(
                pos - center_global[None], dim=1
            )
            region_weight = torch.sigmoid(
                (radius_global - distance_global) / 0.05
            )
            inside = (h * region_weight[:, None]).sum(dim=0) / region_weight.sum().clamp_min(
                1e-6
            )
            outside_weight = 1.0 - region_weight
            outside = (h * outside_weight[:, None]).sum(dim=0) / outside_weight.sum().clamp_min(
                1e-6
            )
            ratio_raw = self.ratio_refinement(
                torch.cat((pooled.squeeze(0), inside, outside), dim=0)
            )
            output["log_ratio_mean"] = _bounded(
                ratio_raw[:1].reshape(1, 1), *LOG_RATIO_BOUNDS
            )
            output["log_ratio_logvar"] = _bounded(
                ratio_raw[1:].reshape(1, 1), *LOGVAR_BOUNDS
            )
            occupancy = torch.sigmoid(-output["node_sdf_mean"] / 0.05)
            normalizer = occupancy.sum().clamp_min(1e-6)
            center = (pos * occupancy[:, None]).sum(dim=0) / normalizer
            radius = (
                ((pos - center).square().sum(dim=1) * occupancy).sum()
                / normalizer
            ).clamp_min(1e-8).sqrt() * math.sqrt(5.0 / 3.0)
            # Keep the independently supervised global region heads intact.
            # Overwriting them here disconnects their losses from the model.
            output["sdf_center_fraction"] = center.unsqueeze(0)
            output["sdf_radius_fraction"] = radius.reshape(1, 1).clamp(
                *RADIUS_BOUNDS
            )
        return output

    def forward(
        self,
        graph: dict[str, Any] | list[dict[str, Any]] | torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if isinstance(graph, list):
            if not graph:
                raise ValueError("Graph batch must not be empty")
            return _merge_graph_outputs(
                [
                    self._forward_one(
                        item.get("static_x", item["x"])
                        if item.get("dynamic_seq") is not None
                        else item["x"],
                        item["edge_index"],
                        item.get("dynamic_seq"),
                        item.get("pos"),
                    )
                    for item in graph
                ]
            )
        if isinstance(graph, dict):
            return self._forward_one(
                graph.get("static_x", graph["x"])
                if graph.get("dynamic_seq") is not None
                else graph["x"],
                graph["edge_index"],
                graph.get("dynamic_seq"),
                graph.get("pos"),
            )
        if edge_index is None:
            raise ValueError("edge_index is required when passing x directly")
        return self._forward_one(graph, edge_index)

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False
    ) -> Any:
        # Frozen peak-frame checkpoints predate temporal/SDF heads.
        legacy = (
            "node_sdf_head.weight" not in state_dict
            or "ratio_refinement.weight" not in state_dict
            or (
                self.temporal_encoder is not None
                and (
                    "temporal_encoder.summary_projection.0.weight" not in state_dict
                    or "temporal_encoder.view_summary_projection.0.weight"
                    not in state_dict
                    or "temporal_encoder.node_summary_projection.0.weight"
                    not in state_dict
                )
            )
        )
        return super().load_state_dict(
            state_dict, strict=False if legacy else strict, assign=assign
        )


class GlobalFeatureMLP(nn.Module):
    """Mean/max pooled baseline with the same bounded prediction interface."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        node_layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            node_layers.extend(
                (nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
            )
        self.input_dim = input_dim
        self.node_encoder = nn.Sequential(*node_layers)
        self.pool_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.material_heads = _MaterialHeads(hidden_dim)
        self.node_head = nn.Linear(hidden_dim, 2)
        self.node_sdf_head = nn.Linear(hidden_dim, 2)
        self.partition_head = nn.Linear(hidden_dim, 1)

    def _forward_one(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must have shape (nodes, {self.input_dim})")
        h = self.node_encoder(x.to(dtype=self.node_encoder[0].weight.dtype))
        pooled = self.pool_projection(torch.cat((h.mean(dim=0), h.amax(dim=0)))).unsqueeze(0)
        output = self.material_heads(pooled)
        node_raw = self.node_head(h)
        output["node_log_E"] = _bounded(node_raw[:, 0], *NODE_LOG_E_BOUNDS)
        output["node_log_E_logvar"] = _bounded(node_raw[:, 1], *LOGVAR_BOUNDS)
        sdf_raw = self.node_sdf_head(h)
        output["node_sdf_mean"] = 2.0 * torch.tanh(sdf_raw[:, 0])
        output["node_sdf_logvar"] = _bounded(sdf_raw[:, 1], *LOGVAR_BOUNDS)
        output["partition_logits"] = self.partition_head(h).squeeze(-1)
        return output

    def forward(
        self, graph: dict[str, Any] | list[dict[str, Any]] | torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if isinstance(graph, list):
            if not graph:
                raise ValueError("Graph batch must not be empty")
            return _merge_graph_outputs(
                [self._forward_one(item["x"]) for item in graph]
            )
        return self._forward_one(graph["x"] if isinstance(graph, dict) else graph)

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False
    ) -> Any:
        legacy = "node_sdf_head.weight" not in state_dict
        return super().load_state_dict(
            state_dict, strict=False if legacy else strict, assign=assign
        )


def decode_predictions(
    output: dict[str, torch.Tensor | list[torch.Tensor]],
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Decode bounded network means to physical FEM parameters."""
    log_background = output["log_E_background_mean"]
    log_ratio = output["log_ratio_mean"]
    assert isinstance(log_background, torch.Tensor)
    assert isinstance(log_ratio, torch.Tensor)
    background = log_background.exp()
    ratio = log_ratio.exp()
    node_log_E = output["node_log_E"]
    node_E = (
        [value.exp() for value in node_log_E]
        if isinstance(node_log_E, list)
        else node_log_E.exp()
    )
    return {
        "E_background": background,
        "inclusion_ratio": ratio,
        "E_inclusion": background * ratio,
        "center_fraction": output["center_fraction_mean"],
        "radius_fraction": output["radius_fraction_mean"],
        "heterogeneity_probability": torch.sigmoid(
            output["heterogeneity_logit"]  # type: ignore[arg-type]
        ),
        "E_nodes": node_E,
    }


# Alternate descriptive name for callers.
decode_material_predictions = decode_predictions
