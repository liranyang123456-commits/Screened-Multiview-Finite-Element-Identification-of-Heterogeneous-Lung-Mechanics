"""Visual initializer for low-dimensional, physics-constrained material models.

This module never outputs a free per-pixel Young's-modulus map.  It predicts a
probability over physically interpretable hypotheses and supplies bounded
initial values to the subsequent FEM inverse solve:

``uniform`` -> one modulus;
``single_inclusion`` -> background/inclusion moduli, center and radius;
``complex_or_unknown`` -> reject automatic mechanical recovery.

The semantic tissue/tool head deliberately requires reviewed masks.  It is not
trained from pathology directories or from a clinical label inferred by name.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


HYPOTHESES = ("uniform", "single_inclusion", "complex_or_unknown")
SEMANTIC_CLASSES = ("background", "tissue", "instrument", "lesion_or_target")


class HierarchicalMaterialInitializer(nn.Module):
    """Frame/clip visual prior for initializing, not replacing, FEM inversion."""

    def __init__(
        self,
        pretrained: bool = True,
        semantic_classes: int = len(SEMANTIC_CLASSES),
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)
        self.features = nn.Sequential(*list(encoder.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.semantic_head = nn.Conv2d(512, semantic_classes, kernel_size=1)
        # Temporal mean/variation, contact-load summary and image-motion
        # summary. Absolute stiffness is not identifiable from static RGB;
        # apparent displacement under a measured load is required.
        feature_dim = 512 * 2 + 4
        self.hypothesis_head = nn.Linear(feature_dim, len(HYPOTHESES))
        # log E_bg, log(E_inc / E_bg), cx, cy, log radius, log variance
        self.material_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 6),
        )
        self.register_buffer("log_E_reference", torch.tensor(math.log(5_000.0)))

    def forward(
        self,
        images: torch.Tensor,
        force_features: torch.Tensor | None = None,
        motion_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict priors for ``(B,C,H,W)`` or temporal mean of ``(B,T,C,H,W)``."""
        if images.ndim == 5:
            batch, time, channels, height, width = images.shape
            images = images.reshape(batch * time, channels, height, width)
            temporal = (batch, time)
        elif images.ndim == 4:
            temporal = None
        else:
            raise ValueError("images must have shape (B,C,H,W) or (B,T,C,H,W)")

        feature_map = self.features(images)
        pooled = self.pool(feature_map).flatten(1)
        semantic_logits = self.semantic_head(feature_map)
        if temporal is not None:
            batch, time = temporal
            pooled = pooled.reshape(batch, time, -1)
            temporal_mean = pooled.mean(dim=1)
            temporal_std = pooled.std(dim=1, unbiased=False)
            if force_features is None:
                force_features = torch.zeros(
                    batch, time, 1, dtype=pooled.dtype, device=pooled.device
                )
            if force_features.shape[:2] != (batch, time):
                raise ValueError("force_features must have shape (B,T,F)")
            force_summary = torch.cat(
                [
                    force_features.mean(dim=1),
                    force_features.std(dim=1, unbiased=False),
                ],
                dim=1,
            )
            if motion_features is None:
                motion_features = torch.zeros(
                    batch, 2, dtype=pooled.dtype, device=pooled.device
                )
            if motion_features.shape != (batch, 2):
                raise ValueError("motion_features must have shape (B,2)")
            pooled = torch.cat(
                [temporal_mean, temporal_std, force_summary, motion_features], dim=1
            )
            semantic_logits = semantic_logits.reshape(
                batch, time, *semantic_logits.shape[1:]
            ).mean(dim=1)
        else:
            if force_features is None:
                force_features = torch.zeros(
                    pooled.shape[0], 1, dtype=pooled.dtype, device=pooled.device
                )
            if force_features.ndim != 2:
                raise ValueError("force_features must have shape (B,F)")
            pooled = torch.cat(
                [
                    pooled,
                    torch.zeros_like(pooled),
                    force_features,
                    torch.zeros_like(force_features),
                    torch.zeros(pooled.shape[0], 2, dtype=pooled.dtype, device=pooled.device),
                ],
                dim=1,
            )

        raw = self.material_head(pooled)
        # Bounded geometric initializations; E quantities stay positive.
        center = torch.sigmoid(raw[:, 2:4])
        radius = 0.05 + 0.35 * torch.sigmoid(raw[:, 4:5])
        return {
            "semantic_logits": semantic_logits,
            "hypothesis_logits": self.hypothesis_head(pooled),
            # Center log-E near the sim_v3 stiffness range for stable regression.
            # sim_v3 spans 3--12 kPa. Bounding the visual prior prevents a
            # small-data regressor from supplying an unusable FEM initialization.
            "log_E_background": self.log_E_reference + 1.2 * torch.tanh(raw[:, :1]),
            "log_inclusion_ratio": raw[:, 1:2],
            "inclusion_center_xy": center,
            "inclusion_radius": radius,
            "log_material_variance": raw[:, 5:6],
        }


def physics_ready_predictions(
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert network output into positive FEM initializer parameters."""
    background = torch.exp(output["log_E_background"])
    ratio = 1.0 + torch.exp(output["log_inclusion_ratio"])
    return {
        "hypothesis_probability": output["hypothesis_logits"].softmax(dim=1),
        "E_background": background,
        "E_inclusion": background * ratio,
        "inclusion_center_xy": output["inclusion_center_xy"],
        "inclusion_radius": output["inclusion_radius"],
        "material_variance": torch.exp(output["log_material_variance"]),
    }
