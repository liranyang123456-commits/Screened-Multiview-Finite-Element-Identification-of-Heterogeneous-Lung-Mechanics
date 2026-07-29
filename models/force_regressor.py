"""3D ResNet force regressor for laparoscopic video windows."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class ForceRegressor(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in (self.backbone.layer3, self.backbone.layer4):
            for parameter in block.parameters():
                parameter.requires_grad = True

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.backbone(video)).squeeze(-1)
