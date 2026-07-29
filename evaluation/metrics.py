"""Evaluation metrics for endoscopy material recovery.

Three families:
  - Image quality: PSNR, SSIM (numpy, no learned weights), LPIPS placeholder.
  - Depth/geometry: AbsRel, δ1/δ2/δ3 thresholds.
  - Material: relative error for E (mechanical), albedo, roughness (optical).
All operate on torch tensors (float, [0,1] for images).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import math


# ---------- image quality ----------

def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR in dB. Tensors any shape; pred/gt in [0,1] (or same scale)."""
    mse = ((pred - gt) ** 2).mean().item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_kernel(window=11, sigma=1.5, channels=1, dtype=torch.float64, device='cpu'):
    coords = torch.arange(window, dtype=dtype, device=device) - window // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g[:, None] * g[None, :]
    return k2d.expand(channels, 1, window, window).contiguous()


def ssim(pred: torch.Tensor, gt: torch.Tensor, window=11) -> float:
    """Structural similarity, averaged. pred/gt shape (C,H,W) in [0,1]."""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0); gt = gt.unsqueeze(0)
    C = pred.shape[0]
    k = _gaussian_kernel(window, 1.5, C, pred.dtype, pred.device)
    pad = window // 2
    mu_p = F.conv2d(pred.unsqueeze(0), k, padding=pad, groups=C)
    mu_g = F.conv2d(gt.unsqueeze(0), k, padding=pad, groups=C)
    mu_p2 = mu_p * mu_p; mu_g2 = mu_g * mu_g; mu_pg = mu_p * mu_g
    sig_p = F.conv2d(pred.unsqueeze(0) * pred.unsqueeze(0), k, padding=pad, groups=C) - mu_p2
    sig_g = F.conv2d(gt.unsqueeze(0) * gt.unsqueeze(0), k, padding=pad, groups=C) - mu_g2
    sig_pg = F.conv2d(pred.unsqueeze(0) * gt.unsqueeze(0), k, padding=pad, groups=C) - mu_pg
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_pg + C1) * (2 * sig_pg + C2)) / \
               ((mu_p2 + mu_g2 + C1) * (sig_p + sig_g + C2))
    return ssim_map.mean().item()


def lpips_placeholder(pred, gt):
    """LPIPS requires a pretrained VGG; not available offline. We expose the
    interface but fall back to (1-SSIM)/2 as a perceptual-ish proxy, clearly
    labelled. Replace with the real LPIPS model when online."""
    return (1.0 - ssim(pred, gt)) / 2.0


# ---------- depth / geometry ----------

def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor = None):
    """Standard monocular depth metrics. pred/gt/valid same shape (H,W) or flat.

    Returns dict with AbsRel, SqRel, RMSE, RMSElog, δ1, δ2, δ3.
    """
    pred = pred.reshape(-1); gt = gt.reshape(-1)
    if valid is not None:
        m = valid.reshape(-1).bool()
        pred = pred[m]; gt = gt[m]
    eps = 1e-6
    absrel = (torch.abs(pred - gt) / gt.clamp_min(eps)).mean().item()
    sqrel = ((pred - gt) ** 2 / gt.clamp_min(eps)).mean().item()
    rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()
    rmselog = torch.sqrt((((torch.log(pred.clamp_min(eps)) - torch.log(gt.clamp_min(eps))) ** 2)
                          .mean())).item()
    thresh = torch.max(pred / gt.clamp_min(eps), gt / pred.clamp_min(eps))
    d1 = (thresh < 1.25).float().mean().item()
    d2 = (thresh < 1.25 ** 2).float().mean().item()
    d3 = (thresh < 1.25 ** 3).float().mean().item()
    return {"AbsRel": absrel, "SqRel": sqrel, "RMSE": rmse, "RMSElog": rmselog,
            "d1": d1, "d2": d2, "d3": d3}


# ---------- material ----------

def material_metrics(recovered: dict, gt: dict):
    """recovered/gt dicts with keys E, albedo (3,), roughness. Returns rel errs."""
    e_err = abs(recovered["E"] - gt["E"]) / gt["E"]
    a = torch.tensor(recovered["albedo"]); ag = torch.tensor(gt["albedo"])
    a_err = (a - ag).abs().mean() / ag.abs().mean()
    r_err = abs(recovered["roughness"] - gt["roughness"]) / gt["roughness"]
    return {"E_rel": float(e_err), "albedo_rel": float(a_err),
            "rough_rel": float(r_err)}
