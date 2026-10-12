#!/usr/bin/env python3
"""New Array-round figures from locked (not frozen) result JSON.

Does not overwrite frozen 120/30-scene or screened-MAP artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_aiim" / "figures"))
from generate_aiim_figures import (  # noqa: E402
    BLUE,
    GRAY,
    GREEN,
    INK,
    LIGHT,
    ORANGE,
    PURPLE,
    configure_style,
    panel_label,
    save_figure,
)

OUT = Path(__file__).resolve().parent
MULTISTART = ROOT / "results" / "lung_mesh_ai_v2" / "fem_multistart50_v1" / "summary.json"
TRACK = (
    ROOT
    / "results"
    / "lung_mesh_ai_v2"
    / "structured_tracking_v1"
    / "mixed"
    / "metrics_image_tracks_force_map_sigma_0.05_px_0.05_nodes_allframes_temporalreg_smooth2_multiview_conf0.20_regionweight8.0_track_mixed_gnn_init_predicted_region.json"
)
LOCK = ROOT / "results" / "ion_ct_ratio_optimization" / "fair_calibrate" / "official_lock.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig_fem_multistart() -> None:
    report = load(MULTISTART)
    ours = report["reference_screened_map"]
    fixed = report["reference_fixed"]
    labels = ["Learned-prior\nScreened MAP", "Fixed\nstart", "4-start\nFEM"]
    colors = [BLUE, GRAY, ORANGE]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 7.2))

    for ax, key, ci_key, title, gate in (
        (axes[0, 0], "E", "E_ci", "Background-modulus error", 10),
        (axes[0, 1], "ratio", "ratio_ci", "Inclusion-ratio error", 25),
    ):
        vals = np.asarray([ours[key], fixed[key], report[key]]) * 100
        cis = np.asarray([ours[ci_key], fixed[ci_key], report[ci_key]]) * 100
        yerr = np.vstack((vals - cis[:, 0], cis[:, 1] - vals))
        bars = ax.bar(np.arange(3), vals, yerr=yerr, capsize=4, color=colors, edgecolor=INK)
        for bar, value, upper in zip(bars, vals, yerr[1], strict=True):
            # Label above the upper CI cap so that it never sits on the error bar;
            # if that position would touch the dashed gate line, lift it above the gate.
            label_y = value + upper + 1.0
            if abs(label_y - gate) < 2.5:
                label_y = gate + 1.2
            ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{value:.1f}%", ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(0, (vals + yerr[1]).max() * 1.18)
        ax.axhline(gate, color=INK, ls="--", lw=0.8)
        ax.set_xticks(range(3), labels)
        ax.set_ylabel("Median relative error (%)")
        ax.set_title(title)
        ax.grid(axis="y", color=LIGHT)

    ax = axes[1, 0]
    evals = np.asarray([ours["evals"], fixed["evals"], report["total_budget"]])
    accepted = np.asarray([ours["accepted"], fixed["accepted"], report["accepted"]]) * 100
    x = np.arange(3)
    bars_e = ax.bar(x - 0.18, evals, 0.36, color=BLUE, edgecolor=INK)
    ax2 = ax.twinx()
    bars_a = ax2.bar(x + 0.18, accepted, 0.36, color=ORANGE, edgecolor=INK, hatch="//")
    for bar, value in zip(bars_e, evals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}", ha="center", fontsize=7.5)
    for bar, value in zip(bars_a, accepted, strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median / total evaluations")
    ax2.set_ylabel("Refinement accepted (%)")
    ax.set_title("Budget and acceptance")
    ax.set_ylim(0, max(evals) * 1.25)
    ax2.set_ylim(0, 115)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(
        [bars_e, bars_a],
        ["Evaluations", "Accepted (%)"],
        loc="upper left",
        fontsize=8,
    )

    ax = axes[1, 1]
    order = ["2500/1.25", "5000/1.8", "8000/2.8", "12000/4"]
    # Compact "E_b (Pa) / ratio" start labels; the long "2500/1.25000..." form
    # ran the category labels into one another at the printed width.
    short = {"2500/1.25": "2.5k/1.25", "5000/1.8": "5k/1.8", "8000/2.8": "8k/2.8", "12000/4": "12k/4"}
    wins = [report["start_wins"][key] for key in order]
    bars = ax.bar(range(len(order)), wins, color=BLUE, edgecolor=INK)
    bars[1].set_color(GREEN)
    for bar, value in zip(bars, wins, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value}/50", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(order)), [short[key] for key in order])
    ax.set_xlabel("Start: $E_b$ (Pa) / ratio")
    ax.set_ylabel("Cases selecting this start")
    ax.set_title("Lowest-cost start among 4")
    ax.set_ylim(0, 28)
    ax.grid(axis="y", color=LIGHT)

    for label, axis in zip(("A", "B", "C", "D"), axes.flat, strict=True):
        panel_label(axis, label, x=-0.16)
    fig.suptitle("Stronger FEM control: 4 deterministic starts on 50 synthetic cases", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0, w_pad=1.5)
    save_figure(fig, OUT, "fig_fem_multistart")


def fig_ct_ratio_calibration() -> None:
    lock = load(LOCK)
    official = lock["official_hybrid"]
    frozen_geoms = {
        "58db559b": 0.2702,
        "62010c56": 0.2686,
        "a417d13e": 0.3355,
        "ab578f1b": 0.2920,
        "db38a9c3": 0.2042,
        "fac5cdc8": 0.2481,
    }
    official_geoms = {
        key.replace("geom_", "")[:8]: value
        for key, value in official["test_ratio"]["geometry_medians"].items()
    }
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.8))

    ax = axes[0]
    names = ["Raw MeshGNN\n(uncalibrated)", "Calibrated hybrid\ninitializer", "Ridge\nsame protocol", "PCA–Ridge\nsame protocol"]
    vals = np.asarray([0.2695, official["ratio_median"], 0.4163, 0.4483]) * 100
    colors = [GRAY, BLUE, ORANGE, PURPLE]
    bars = ax.bar(np.arange(4), vals, color=colors, edgecolor=INK)
    for bar, value in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}%", ha="center", fontsize=7.5)
    ax.axhline(25, color=INK, ls="--", lw=0.9, label="25% gate")
    ax.set_xticks(range(4), names)
    ax.set_ylabel("Held-out median ratio error (%)")
    ax.set_title("Same train-fit / validation-select protocol")
    ax.set_ylim(0, 52)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    keys = list(frozen_geoms)
    x = np.arange(len(keys))
    frozen = np.asarray([frozen_geoms[key] for key in keys]) * 100
    cal = np.asarray([official_geoms[key] * 100 for key in keys])
    ax.bar(x - 0.18, frozen, 0.36, color=GRAY, edgecolor=INK, label="Raw MeshGNN (uncalibrated), 26.95%")
    ax.bar(x + 0.18, cal, 0.36, color=BLUE, edgecolor=INK, label="Official calibration (17.77%)")
    ax.axhline(25, color=INK, ls="--", lw=0.9)
    ax.set_xticks(x, keys, rotation=20, ha="right")
    ax.set_ylabel("Geometry-median ratio error (%)")
    ax.set_title("Six locked test geometries")
    ax.set_ylim(0, 42)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(fontsize=8)
    for label, axis in zip(("A", "B"), axes, strict=True):
        panel_label(axis, label)
    fig.suptitle("Official CT ratio calibration: a scalar post-process of the frozen network", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, OUT, "fig_ct_ratio_calibration")


def fig_structured_tracking() -> None:
    mixed = load(TRACK)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.2))
    labels = ["Frozen i.i.d.\n0.05 px", "Structured\nmixed noise"]
    e_vals = [4.11, mixed["E_background_median_relative_error"] * 100]
    r_vals = [18.69, mixed["inclusion_ratio_median_relative_error"] * 100]
    accept = [8.0, mixed["refinement_acceptance_rate"] * 100]
    evals = [7.0, mixed["function_evaluations_median"]]

    ax = axes[0]
    x = np.arange(2)
    bars_e = ax.bar(x - 0.18, e_vals, 0.36, color=BLUE, edgecolor=INK, label="Background E")
    bars_r = ax.bar(x + 0.18, r_vals, 0.36, color=ORANGE, edgecolor=INK, hatch="//", label="Ratio")
    for bars, values in ((bars_e, e_vals), (bars_r, r_vals)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}%", ha="center", fontsize=7.5)
    ax.axhline(10, color=BLUE, ls=":", lw=0.8)
    ax.axhline(25, color=ORANGE, ls=":", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Learned-prior screened MAP median error (%)")
    ax.set_title("Accuracy under a more realistic track model")
    ax.set_ylim(0, 28)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(fontsize=8)

    ax = axes[1]
    bars_a = ax.bar(x - 0.18, accept, 0.36, color=GREEN, edgecolor=INK)
    ax2 = ax.twinx()
    bars_v = ax2.bar(x + 0.18, evals, 0.36, color=GRAY, edgecolor=INK, hatch="//")
    for bar, value in zip(bars_a, accept, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.0f}%", ha="center", fontsize=7.5)
    for bar, value in zip(bars_v, evals, strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Refinement accepted (%)")
    ax2.set_ylabel("Median evaluations")
    ax.set_title("Screen still refuses most updates")
    ax.set_ylim(0, 16)
    ax2.set_ylim(0, 12)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(
        [bars_a, bars_v],
        ["Accepted (%)", "Evaluations"],
        loc="upper right",
        fontsize=8,
    )
    for label, axis in zip(("A", "B"), axes, strict=True):
        panel_label(axis, label)
    fig.suptitle("Structured tracking noise does not collapse screened MAP", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, OUT, "fig_structured_tracking")


def main() -> None:
    global OUT
    import argparse

    builders = {
        "fig_fem_multistart": fig_fem_multistart,
        "fig_ct_ratio_calibration": fig_ct_ratio_calibration,
        "fig_structured_tracking": fig_structured_tracking,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Destination for PDF/PNG assets.")
    parser.add_argument("--only", nargs="+", choices=sorted(builders), default=None,
                        help="Regenerate only the listed figure stems.")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    configure_style()
    for stem, builder in builders.items():
        if args.only is None or stem in args.only:
            builder()
            print(f"wrote {stem} -> {OUT}")


if __name__ == "__main__":
    main()
