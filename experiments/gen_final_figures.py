"""Generate the final combined results figure once v2 + real-colon evals finish.

Panel layout:
  (a) v2 multi-anatomy: E_err per anatomy (bar)
  (b) real-colon: E_err per segment (bar)
  (c) combined PSNR distribution
Run after both results/sim_eval_v2.pt and results/real_colon_eval.pt exist.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(ROOT, "paper_tbme", "figures")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    v2 = torch.load(os.path.join(ROOT, "results", "sim_eval_v2.pt"), weights_only=False)
    rc = torch.load(os.path.join(ROOT, "results", "real_colon_eval.pt"), weights_only=False)
    v2_rows = v2["rows"]; rc_rows = rc["rows"]

    import collections
    v2_by = collections.defaultdict(list)
    for r in v2_rows: v2_by[r.get("anatomy", "?")].append(r["E_rel"] * 100)
    rc_by = collections.defaultdict(list)
    for r in rc_rows: rc_by[r.get("segment", "?")].append(r["E_rel"] * 100)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    # (a) v2 per anatomy
    anats = sorted(v2_by)
    means = [sum(v2_by[a]) / len(v2_by[a]) for a in anats]
    ax[0].bar(anats, means, color='#6fa8dc', edgecolor='k')
    ax[0].set_ylabel("Young-modulus error (%)"); ax[0].set_title("(a) Multi-anatomy (procedural)")
    ax[0].tick_params(axis='x', labelrotation=30)
    # (b) real colon per segment
    segs = sorted(rc_by)
    rmeans = [sum(rc_by[s]) / len(rc_by[s]) for s in segs]
    ax[1].bar(segs, rmeans, color='#93c47d', edgecolor='k')
    ax[1].set_ylabel("Young-modulus error (%)"); ax[1].set_title("(b) Real C3VD colon geometry")
    ax[1].tick_params(axis='x', labelrotation=30)
    # (c) combined PSNR
    psnr = [r["PSNR"] for r in v2_rows] + [r["PSNR"] for r in rc_rows]
    ax[2].hist(psnr, bins=12, color='#e6b3cc', edgecolor='k')
    ax[2].axvline(sum(psnr) / len(psnr), color='r', ls='--', label=f'mean {sum(psnr)/len(psnr):.1f}')
    ax[2].set_xlabel("PSNR (dB)"); ax[2].set_title("(c) Render quality (v2 + real-colon)")
    ax[2].legend()
    plt.tight_layout()
    output = os.path.join(FIGURE_DIR, "fig3_multi_anatomy.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
