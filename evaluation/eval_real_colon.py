"""Eval the real-colon dataset (C3VD geometry). Same logic as eval_sim_v2 but
points at dataset/real_colon and reports per-segment breakdown.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from evaluation.eval_sim import eval_scene, aggregate

ROOT = "dataset/real_colon"


def main():
    torch.set_default_dtype(torch.float64)
    with open(os.path.join(ROOT, "manifest.json")) as f:
        manifest = json.load(f)
    scenes = manifest["scenes"]
    print(f"evaluating {len(scenes)} real-colon scenes...", flush=True)
    os.makedirs("results", exist_ok=True)
    ckpt = "results/real_colon_eval.pt"
    rows = []; done_ids = set()
    if os.path.exists(ckpt):
        try:
            prev = torch.load(ckpt, weights_only=False)
            rows = prev.get("rows", [])
            done_ids = {r.get("_id") for r in rows if "_id" in r}
            print(f"  resumed {len(rows)}", flush=True)
        except Exception:
            pass
    for sc in scenes:
        if sc["id"] in done_ids: continue
        sr = os.path.join(ROOT, f"scene_{sc['id']:04d}")
        gt = torch.load(os.path.join(sr, "gt.pt"), weights_only=False)
        try:
            r = eval_scene(gt, H=gt["H"], W=gt["W"], iters=40, scene_root=sr)
        except Exception as e:
            print(f"  scene {sc['id']:04d}: FAILED ({e})", flush=True); continue
        r["_id"] = sc["id"]; r["segment"] = sc.get("segment", sc["anatomy"])
        rows.append(r)
        print(f"  {sc['id']:04d} [{r['segment']:10s}] E_rel={r['E_rel']*100:5.1f}% "
              f"alb={r['albedo_rel']*100:5.1f}% PSNR={r['PSNR']:5.2f}", flush=True)
        if len(rows) % 2 == 0:
            torch.save({"rows": rows, "agg": None}, ckpt)
    agg = aggregate(rows)
    import collections
    by_seg = collections.defaultdict(list)
    for r in rows: by_seg[r.get("segment", "?")].append(r)
    agg["per_segment"] = {s: {k: sum(r[k] for r in rs)/len(rs)
                              for k in ["E_rel","albedo_rel","rough_rel","PSNR","SSIM"]}
                          for s, rs in by_seg.items()}
    torch.save({"rows": rows, "agg": agg}, ckpt)
    print("\n=== REAL-COLON AGGREGATE ===")
    for k in ["E_rel","albedo_rel","rough_rel","PSNR","SSIM"]:
        print(f"  {k}: mean={agg[k+'_mean']:.4f} median={agg[k+'_med']:.4f}")
    print("\n=== PER SEGMENT ===")
    for s, m in agg["per_segment"].items():
        print(f"  {s:10s}: E_err={m['E_rel']*100:4.0f}% alb={m['albedo_rel']*100:4.0f}% PSNR={m['PSNR']:5.2f}")
    L = ["# Real-colon evaluation (C3VD geometry)\n\n## Aggregate\n| metric | mean | median |\n|---|---|---|\n"]
    for k in ["E_rel","albedo_rel","rough_rel","PSNR","SSIM"]:
        L.append(f"| {k} | {agg[k+'_mean']:.4f} | {agg[k+'_med']:.4f} |\n")
    L.append("\n## Per segment\n| segment | E_err% | alb_err% | PSNR |\n|---|---|---|---|\n")
    for s, m in agg["per_segment"].items():
        L.append(f"| {s} | {m['E_rel']*100:.0f} | {m['albedo_rel']*100:.0f} | {m['PSNR']:.2f} |\n")
    with open("results/real_colon_eval.md", "w", encoding="utf-8") as f:
        f.write("".join(L))
    print("\nwrote results/real_colon_eval.md")


if __name__ == "__main__":
    main()
