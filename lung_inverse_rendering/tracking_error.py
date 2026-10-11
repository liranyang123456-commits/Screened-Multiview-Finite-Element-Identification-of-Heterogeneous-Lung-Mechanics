"""Optional structured tracking-error models stacked on the frozen 0.05 px i.i.d. noise.

Default model ``iid`` reproduces the published protocol. Other models add
view-constant bias, confidence-dependent scale, and random node dropout to
approximate tracker drift and correspondence loss. They are evaluation-time
injections and do not rewrite stored tracks.
"""
from __future__ import annotations

import torch


class TrackingError:
    def __init__(
        self,
        model: str,
        track_noise_px: float,
        generator: torch.Generator,
        view_bias_px: float = 0.04,
        dropout_rate: float = 0.10,
    ) -> None:
        if model not in {"iid", "view_bias", "conf_scaled", "dropout", "mixed"}:
            raise ValueError(f"Unknown tracking-error model: {model}")
        self.model = model
        self.track_noise_px = float(track_noise_px)
        self.generator = generator
        self.view_bias_px = float(view_bias_px)
        self.dropout_rate = float(dropout_rate)
        self._view_bias: dict[int, torch.Tensor] = {}

    def noise(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        view_key: int,
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        iid = self.track_noise_px * torch.randn(
            shape, generator=self.generator, dtype=dtype
        )
        extra = torch.zeros_like(iid)
        if self.model in {"view_bias", "mixed"}:
            if view_key not in self._view_bias:
                self._view_bias[view_key] = self.view_bias_px * torch.randn(
                    shape[-1], generator=self.generator, dtype=dtype
                )
            extra = extra + self._view_bias[view_key]
        if self.model in {"conf_scaled", "mixed"} and confidence is not None:
            scale = 1.0 / confidence.to(dtype=dtype).clamp(min=0.05).sqrt()
            while scale.ndim < iid.ndim:
                scale = scale.unsqueeze(-1)
            iid = iid * scale
        keep = None
        if self.model in {"dropout", "mixed"}:
            keep = (
                torch.rand(shape[0], generator=self.generator, dtype=torch.float64)
                >= self.dropout_rate
            )
        return iid + extra, keep
