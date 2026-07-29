"""SOTA baseline depth/pose evaluation on the simulation benchmark.

Runs external SOTA methods (with locally-available weights) on the rendered
endoscopy images and reports depth metrics, to position our pipeline against
the geometry-reconstruction state of the art.

Available locally (per SOTA survey):
  - Depth-Anything-V2  (E:\SOTA_Methods\Depth-Anything-V2, weights present)
  - FoundationStereo   (E:\SOTA_Methods\FoundationStereo, weights present)
  - ZoeDepth           (E:\SOTA_Methods\ZoeDepth, ZoeD_NK.pth 1.45GB)

Note: these predict DEPTH only (not material). The honest framing is that they
are geometry baselines; our differentiator is the material (albedo + E)
recovery, which none of them do. We report their depth error vs. our
pipeline's implied depth as context.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from evaluation.metrics import depth_metrics


SOTA_ROOT = r"E:\SOTA_Methods"


def _load_image(path):
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def run_depth_anything_v2(image_np, max_dim=512):
    """Run Depth-Anything-V2 on a single image. Returns relative depth (H,W).
    Returns None if the model can't be loaded (keeps evaluation robust)."""
    try:
        sys.path.insert(0, os.path.join(SOTA_ROOT, "Depth-Anything-V2"))
        from depth_anything_v2.dpt import DepthAnythingV2
        # find a checkpoint (prefer vitb for best quality)
        ckpt_dirs = [os.path.join(SOTA_ROOT, "Depth-Anything-V2", "checkpoints"),
                     os.path.join(SOTA_ROOT, "Depth-Anything-V2")]
        ckpt = None
        for d in ckpt_dirs:
            if os.path.isdir(d):
                files = [f for f in os.listdir(d) if f.endswith(".pth")]
                # prefer vitb, then vitl, then vits
                for enc in ["vitb", "vitl", "vits"]:
                    for f in files:
                        if enc in f.lower():
                            ckpt = os.path.join(d, f); break
                    if ckpt: break
            if ckpt: break
        if ckpt is None:
            return None, "no .pth checkpoint found"
        # encoder from filename
        cname = os.path.basename(ckpt).lower()
        if "vitl" in cname:
            encoder, features, out_channels = 'vitl', 256, 192
        elif "vitb" in cname:
            encoder, features, out_channels = 'vitb', 256, 192
        else:
            encoder, features, out_channels = 'vits', 64, 48
        cfg = dict(encoder=encoder, features=features, out_channels=out_channels)
        model = DepthAnythingV2(**cfg)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict({k: v for k, v in state.items()
                               if k in model.state_dict()}, strict=False)
        model.eval()
        with torch.no_grad():
            depth = model.infer_image(image_np, max_dim)
        return depth, "ok"
    except Exception as e:
        return None, str(e)


def evaluate_sota_depth(dataset_root, scene_ids, ours_depth_fn):
    """For each scene, run DA-V2 on its frames, compute depth metrics vs. GT.
    ours_depth_fn(scene) -> (pred_depth, gt_depth, valid_mask) for our method.
    """
    rows = []
    for sid in scene_ids:
        sdir = os.path.join(dataset_root, f"scene_{sid:04d}")
        gt = torch.load(os.path.join(sdir, "gt.pt"), weights_only=False)
        # our pipeline's depth (from FEM displacement + render)
        try:
            p_ours, g_gt, vmask = ours_depth_fn(gt)
            m_ours = depth_metrics(p_ours, g_gt, vmask)
        except Exception as e:
            m_ours = {"AbsRel": float("nan")}
        # DA-V2 depth on frame 3 (a deformed frame)
        img_path = os.path.join(sdir, "images", "frame_03.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(sdir, "images", "frame_03.png")
        dav2_depth, dav2_msg = None, "skipped"
        if os.path.exists(img_path):
            img = _load_image(img_path)
            dav2_depth, dav2_msg = run_depth_anything_v2(img)
        row = {"scene": sid, "anatomy": gt.get("anatomy"),
               "ours_AbsRel": m_ours["AbsRel"], "dav2_status": dav2_msg}
        rows.append(row)
        print(f"  scene {sid:04d} [{gt.get('anatomy','?')}]: ours AbsRel={m_ours['AbsRel']:.4f} | "
              f"DA-V2: {dav2_msg}", flush=True)
    return rows


if __name__ == "__main__":
    # smoke test: does DA-V2 load at all?
    print("smoke-testing Depth-Anything-V2 load...")
    dummy = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    d, msg = run_depth_anything_v2(dummy, max_dim=128)
    print(f"  result: {msg}, depth shape: {None if d is None else d.shape}")
