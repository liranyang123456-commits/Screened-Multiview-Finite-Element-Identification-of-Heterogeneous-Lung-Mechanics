"""Eval the v2 (multi-anatomy, 128x128) dataset. Thin wrapper over eval_sim
that uses 4-digit scene ids and reports per-anatomy breakdown.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from evaluation.eval_sim import eval_scene, aggregate

ROOT = "dataset/sim_v2"


def main():
    torch.set_default_dtype(torch.float64)
    with open(os.path.join(ROOT, "manifest.json")) as f:
        manifest = json.load(f)
    scenes = manifest["scenes"]
    print(f"evaluating {len(scenes)} v2 scenes...", flush=True)
    os.makedirs("results", exist_ok=True)
    ckpt = "results/sim_eval_v2.pt"
    rows = []
    done_ids = set()
    if os.path.exists(ckpt):
        try:
            prev = torch.load(ckpt, weights_only=False)
            rows = prev.get("rows", [])
            done_ids = {r.get("_id") for r in rows if "_id" in r}
            print(f"  resumed {len(rows)} scenes", flush=True)
        except Exception:
            pass

    for sc in scenes:
        if sc["id"] in done_ids:
            continue
        scene_root = os.path.join(ROOT, f"scene_{sc['id']:04d}")  # 4-digit for v2
        gt = torch.load(os.path.join(scene_root, "gt.pt"), weights_only=False)
        try:
            r = eval_scene(gt, H=gt["H"], W=gt["W"], iters=40, scene_root=scene_root)
        except Exception as e:
            print(f"  scene {sc['id']:04d}: FAILED ({e})", flush=True)
            continue
        r["_id"] = sc["id"]
        r["anatomy"] = sc["anatomy"]
        rows.append(r)
        print(f"  scene {sc['id']:04d} [{sc['anatomy']:13s}] E_rel={r['E_rel']*100:5.1f}% "
              f"alb={r['albedo_rel']*100:5.1f}% PSNR={r['PSNR']:5.2f} SSIM={r['SSIM']:.3f}", flush=True)
        if len(rows) % 2 == 0:
            torch.save({"rows": rows, "agg": None}, ckpt)

    agg = aggregate(rows)
    # per-anatomy breakdown
    import collections
    by_anat = collections.defaultdict(list)
    for r in rows:
        by_anat[r.get("anatomy", "?")].append(r)
    agg["per_anatomy"] = {a: {k: sum(r[k] for r in rs) / len(rs)
                              for k in ["E_rel", "albedo_rel", "rough_rel", "PSNR", "SSIM"]}
                          for a, rs in by_anat.items()}
    torch.save({"rows": rows, "agg": agg}, ckpt)
    print("\n=== AGGREGATE ===")
    for k in ["E_rel", "albedo_rel", "rough_rel", "PSNR", "SSIM"]:
        print(f"  {k}: mean={agg[k+'_mean']:.4f} median={agg[k+'_med']:.4f}")
    print("\n=== PER ANATOMY ===")
    for a, m in agg["per_anatomy"].items():
        print(f"  {a:13s}: E_err={m['E_rel']*100:4.0f}% alb={m['albedo_rel']*100:4.0f}% "
              f"PSNR={m['PSNR']:5.2f} SSIM={m['SSIM']:.3f}")
    # markdown
    L = ["# v2 dataset evaluation (multi-anatomy, 128x128)\n\n",
         "## Aggregate\n| metric | mean | median |\n|---|---|---|\n"]
    for k in ["E_rel", "albedo_rel", "rough_rel", "PSNR", "SSIM"]:
        L.append(f"| {k} | {agg[k+'_mean']:.4f} | {agg[k+'_med']:.4f} |\n")
    L.append("\n## Per anatomy\n| anatomy | E_err% | alb_err% | PSNR | SSIM |\n|---|---|---|---|---|\n")
    for a, m in agg["per_anatomy"].items():
        L.append(f"| {a} | {m['E_rel']*100:.0f} | {m['albedo_rel']*100:.0f} | {m['PSNR']:.2f} | {m['SSIM']:.3f} |\n")
    with open("results/sim_eval_v2.md", "w", encoding="utf-8") as f:
        f.write("".join(L))
    print("\nwrote results/sim_eval_v2.md")


if __name__ == "__main__":
    main()
