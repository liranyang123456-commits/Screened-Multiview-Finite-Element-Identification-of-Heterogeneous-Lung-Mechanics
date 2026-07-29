"""PyTorch Dataset classes for the real endoscopy benchmarks.

EndoNeRF — for novel-view synthesis (NVS) evaluation:
  - LLFF poses_bounds.npy (3x5 + 2 bounds)
  - 8-bit depth png (quantized; un-quantize via near/far)
  - 640x512 RGB

SCARED — for metric depth evaluation:
  - Left_Image.png 1280x1024 RGBA
  - left_depth_map.tiff: XYZ point map (NOT scalar!), depth = Z channel, NaN=invalid
  - endoscope_calibration.yaml: M1/M2/D1/D2/R/T (OpenCV YAML, pixel units)
"""
from __future__ import annotations
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------- EndoNeRF ----------------

ENDONERF_ROOT = r"external_data/EndoNeRF"
ENDONERF_SCENES = ["cutting_tissues_twice", "pulling_soft_tissues"]


def _endonnerf_img_pattern(scene_dir, right=False):
    """EndoNeRF uses two filename conventions across its scenes."""
    img_dir = os.path.join(scene_dir, "images_right" if right else "images")
    if glob.glob(os.path.join(img_dir, "frame-*.color.png")):
        return img_dir, "frame-{:06d}.color.png"
    return img_dir, "{:06d}.png"


class EndoNeRFDataset(Dataset):
    """Loads an EndoNeRF scene. Returns dict with image (3,H,W) in [0,1],
    pose (3,4) OpenCV c2w, depth (H,W) metric (un-quantized), mask (H,W) bool,
    near, far, intrinsics (3,3) if derivable."""

    def __init__(self, scene="pulling_soft_tissues", root=ENDONERF_ROOT,
                 H=None, W=None, max_n=None):
        self.scene_dir = os.path.join(root, scene)
        pb = np.load(os.path.join(self.scene_dir, "poses_bounds.npy"))   # (N,17)
        self.near = float(pb[0, -2]); self.far = float(pb[0, -1])
        poses = pb[:, :15].reshape(-1, 3, 5)                              # (N,3,5)
        # LLFF: each pose row [r0 r1 r2 t] x3, last col = hwf
        self.poses = poses[:, :, :4]                                       # (N,3,4)
        # hwf: take from first pose's last column
        hwf = poses[0, :, 4]
        self.H_gt, self.W_gt, self.focal = float(hwf[0]), float(hwf[1]), float(hwf[2])
        img_dir, pat = _endonnerf_img_pattern(self.scene_dir)
        self.img_dir = img_dir; self.img_pat = pat
        # enumerate actual files
        files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        # filter the stray mp4 / non-image if any
        files = [f for f in files if f.lower().endswith(".png")]
        if max_n is not None:
            files = files[:max_n]
        self.n = min(len(files), self.poses.shape[0])
        self.files = files[:self.n]
        self.H = H or 256; self.W = W or 320    # downscale for speed
        from PIL import Image
        # intrinsics (downscaled): focal scales with resize
        sx = self.W / self.W_gt; sy = self.H / self.H_gt
        self.K = np.array([[self.focal * sx, 0, self.W / 2.0],
                           [0, self.focal * sy, self.H / 2.0],
                           [0, 0, 1.0]])

    def __len__(self):
        return self.n

    def _load_image(self, path):
        from PIL import Image
        im = Image.open(path).convert("RGB").resize((self.W, self.H))
        return np.array(im, dtype=np.float32) / 255.0          # (H,W,3)

    def _load_depth(self, idx):
        """Un-quantize 8-bit depth via near/far. Returns (H,W) metric depth."""
        from PIL import Image
        depth_dir = os.path.join(self.scene_dir, "depth")
        cands = [os.path.join(depth_dir, f"frame-{idx:06d}.depth.png"),
                 os.path.join(depth_dir, f"{idx:06d}.png")]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            return None
        d8 = np.array(Image.open(path).convert("L").resize((self.W, self.H)),
                      dtype=np.float32) / 255.0
        # standard LLFF un-quantization
        return d8 * (self.far - self.near) + self.near

    def __getitem__(self, idx):
        img = self._load_image(self.files[idx])
        pose = self.poses[idx]                               # (3,4)
        depth = self._load_depth(idx)
        return {"image": torch.from_numpy(img).permute(2, 0, 1),    # (3,H,W)
                "pose": torch.from_numpy(pose.astype(np.float32)),
                "depth": None if depth is None else torch.from_numpy(depth),
                "K": torch.from_numpy(self.K.astype(np.float32)),
                "idx": idx}


# ---------------- SCARED ----------------

SCARED_ROOT = r"external_data/SCARED"


class SCAREDDataset(Dataset):
    """Iterates over SCARED keyframes. Returns left image (3,H,W) [0,1],
    metric depth Z (mm) with valid mask, intrinsics (3,3) pixels, stereo T.

    NOTE: per-frame video poses live in kf/data/frame_data.tar.gz (absent in
    some keyframes). Keyframe images themselves have no explicit pose; the
    keyframe is treated as its own frame. We expose depth + image metrics.
    """

    def __init__(self, dataset_id=1, root=SCARED_ROOT, H=512, W=640,
                 keyframes=None):
        self.dkey = f"dataset_{dataset_id}"
        self.kfs = keyframes or [1, 2, 3, 4, 5]
        self.entries = []
        for kf in self.kfs:
            d = os.path.join(root, self.dkey, f"keyframe_{kf}")
            if os.path.exists(os.path.join(d, "Left_Image.png")):
                self.entries.append(d)
        self.H = H; self.W = W
        from PIL import Image
        # load intrinsics from the first available calibration
        self.K = None
        for d in self.entries:
            cal = os.path.join(d, "endoscope_calibration.yaml")
            if os.path.exists(cal):
                self.K = self._read_calib(cal)
                break

    @staticmethod
    def _read_calib(path):
        """Read OpenCV-style YAML (M1/D1/...). Uses cv2 if available."""
        try:
            import cv2
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            M1 = fs.getNode("M1").mat()                      # (3,3) pixels
            fs.release()
            return M1.astype(np.float32)
        except Exception:
            return None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from PIL import Image
        d = self.entries[idx]
        img = Image.open(os.path.join(d, "Left_Image.png")).convert("RGB").resize((self.W, self.H))
        img = np.array(img, dtype=np.float32) / 255.0
        # depth = Z channel of the XYZ point map (mm), NaN where invalid
        depth_xyz = self._read_depth_tiff(os.path.join(d, "left_depth_map.tiff"))
        if depth_xyz is not None:
            depth_mm = depth_xyz[..., 2]
            valid = ~np.isnan(depth_mm)
            depth_mm = np.nan_to_num(depth_mm, nan=0.0)
            depth_mm = np.array(Image.fromarray(depth_mm).resize((self.W, self.H)),
                                dtype=np.float32)
            valid = np.array(Image.fromarray(valid.astype(np.uint8) * 255).resize(
                (self.W, self.H))).astype(bool)
        else:
            depth_mm = np.zeros((self.H, self.W), np.float32); valid = np.zeros((self.H, self.W), bool)
        # rescale intrinsics to (H,W)
        K = None
        if self.K is not None:
            sx = self.W / 1280.0; sy = self.H / 1024.0
            K = self.K.copy()
            K[0, :] *= sx; K[1, :] *= sy
        return {"image": torch.from_numpy(img).permute(2, 0, 1),
                "depth_mm": torch.from_numpy(depth_mm),
                "valid": torch.from_numpy(valid),
                "K": None if K is None else torch.from_numpy(K),
                "keyframe_dir": d}

    @staticmethod
    def _read_depth_tiff(path):
        try:
            import cv2
            d = cv2.imread(path, cv2.IMREAD_UNCHANGED)        # (H,W,3) float32
            return d.astype(np.float32) if d is not None else None
        except Exception:
            try:
                import tifffile
                return tifffile.imread(path).astype(np.float32)
            except Exception:
                return None


if __name__ == "__main__":
    print("=== EndoNeRF ===")
    ds = EndoNeRFDataset(scene="pulling_soft_tissues", max_n=5)
    print(f"  {len(ds)} frames, H={ds.H} W={ds.W} focal={ds.focal:.1f} near={ds.near:.2f} far={ds.far:.2f}")
    s = ds[0]
    print(f"  image {tuple(s['image'].shape)} range [{s['image'].min():.3f},{s['image'].max():.3f}]")
    print(f"  pose {tuple(s['pose'].shape)}, depth {'present' if s['depth'] is not None else 'absent'}")
    print("\n=== SCARED ===")
    sc = SCAREDDataset(dataset_id=1, keyframes=[1, 2])
    print(f"  {len(sc)} keyframes, K={'present' if sc.K is not None else 'absent'}")
    s2 = sc[0]
    print(f"  image {tuple(s2['image'].shape)} depth_mm range [{s2['depth_mm'][s2['valid']].min():.1f},"
          f"{s2['depth_mm'][s2['valid']].max():.1f}] valid_px={int(s2['valid'].sum())}/{s2['valid'].numel()}")
