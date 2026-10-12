"""Plot the recorded continuation-threshold / probe-count sweeps without retuning them.

Rows are sweeps (18 CT-derived scenes: threshold and probe directions;
50 synthetic cases: threshold), columns are the three recorded metrics.
Every column shares one y-range across the rows, so the probe sweep is
shown on the same background-error scale as the threshold sweeps instead of
a zoomed 24.2-24.6 % axis. All data come from the recorded JSON summaries:

* results/ct_reviewer_drivers_v6/reconciled_paired_experiments.json
  (``sensitivity_arms``: 18 CT-derived synthetic-mechanics scenes)
* results/lung_mesh_ai_v2/synthetic50_threshold_sweep/summary.json
  (50 synthetic cases; threshold 0.05 = the existing six-arm reference)
"""
from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CT = ROOT / "results" / "ct_reviewer_drivers_v6" / "reconciled_paired_experiments.json"
SOURCE_SYN = ROOT / "results" / "lung_mesh_ai_v2" / "synthetic50_threshold_sweep" / "summary.json"
OUTPUT = ROOT / "figures"

BLUE, GREEN, ORANGE, GRAY, INK = "#0072B2", "#009E73", "#D55E00", "#5B6573", "#17212B"


def ct_rows(keys, evidence):
    rows = [evidence[key] for key in keys]
    expected_ids = rows[0]["patient_ids"]
    for row in rows:
        if row["n"] != 18 or row["patient_ids"] != expected_ids:
            raise ValueError("CT sensitivity arms must contain the same 18 scenes")
    return {
        "n": 18,
        "error": [100 * row["median_background_error"] for row in rows],
        "accepted": [100 * row["accepted"] / 18 for row in rows],
        "stopped": [100 * row["screen_rejected"] / 18 for row in rows],
        "evals": [row["median_function_evaluations"] for row in rows],
    }


def synthetic_rows(summary):
    arms = dict(summary["arms"])
    arms["s0p05"] = summary["existing_threshold_0p05_reference"]
    order = ["s0p00", "s0p01", "s0p025", "s0p05", "s0p10", "s0p20"]
    xs = []
    rows = []
    for key in order:
        arm = arms[key]
        if arm["n"] != 50 or arm["unique_ids"] != 50 or not arm["ids_equal_locked_set"] or not arm["complete"]:
            raise ValueError(f"Synthetic threshold arm {key} is not a complete locked 50-case arm")
        xs.append(float(arm["screen_threshold"]))
        rows.append(arm)
    return xs, {
        "n": 50,
        "error": [row["median_background_error_percent"] for row in rows],
        "accepted": [100 * row["accepted"] / 50 for row in rows],
        "stopped": [100 * row["continuation_stopped_screening_rejected_flag"] / 50 for row in rows],
        "evals": [row["median_function_evaluations"] for row in rows],
        "accepted_n": [row["accepted"] for row in rows],
        "stopped_n": [row["continuation_stopped_screening_rejected_flag"] for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    evidence = json.loads(SOURCE_CT.read_text(encoding="utf-8-sig"))["sensitivity_arms"]
    summary = json.loads(SOURCE_SYN.read_text(encoding="utf-8-sig"))
    threshold_keys = (
        "w2_s0_p2", "w2_s0p01_p2", "w2_s0p025_p2",
        "w2_s0p05_p2", "w2_s0p1_p2", "w2_s0p2_p2",
    )
    probe_keys = ("w2_s0p05_p2", "w2_s0p05_p4", "w2_s0p05_p6")
    ct_thresholds = [0, 0.01, 0.025, 0.05, 0.10, 0.20]
    probes = [2, 4, 6]
    syn_x, syn = synthetic_rows(summary)
    print("50-case threshold sweep (from JSON):")
    print("  thresholds", syn_x)
    print("  median background error (%)", syn["error"])
    print("  accepted", syn["accepted_n"], "stop flags", syn["stopped_n"], "median evaluations", syn["evals"])
    sweeps = (
        ("18 CT-derived\nscenes:\nthreshold\nsweep", ct_thresholds, ct_rows(threshold_keys, evidence), "Early continuation threshold"),
        ("18 CT-derived\nscenes:\nprobe sweep", probes, ct_rows(probe_keys, evidence), "Directions per active parameter"),
        ("50 synthetic\ncases:\nthreshold\nsweep", syn_x, syn, "Early continuation threshold"),
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(
        3, 3, figsize=(6.5, 6.6),
        gridspec_kw={"left": 0.215, "right": 0.985, "top": 0.93, "bottom": 0.08, "hspace": 0.55, "wspace": 0.42},
    )
    columns = (
        ("Median background error (%)", "error", BLUE),
        ("Accepted / stopped (% cases)", "accepted", GREEN),
        ("Median objective evaluations", "evals", ORANGE),
    )
    # One y-range per metric column, shared by the three sweeps.
    limits = {
        "error": (0, 1.12 * max(max(s[2]["error"]) for s in sweeps)),
        "accepted": (0, 108),
        "evals": (0, 1.12 * max(max(s[2]["evals"]) for s in sweeps)),
    }
    for row_index, (row_title, xs, data, xlabel) in enumerate(sweeps):
        for column_index, (ylabel, key, color) in enumerate(columns):
            ax = axes[row_index, column_index]
            if key == "accepted":
                ax.plot(xs, data["accepted"], "-o", color=GREEN, markersize=4, label="Accepted updates")
                ax.plot(xs, data["stopped"], "--s", color=GRAY, markersize=3.5, label="Continuation stopped")
                if row_index == 1:
                    # The probe-sweep panel has flat lines at 83 % and 17 %,
                    # so the legend sits in its empty centre band.
                    ax.legend(loc="center", frameon=False, handlelength=1.6)
            else:
                ax.plot(xs, data[key], "-o", color=color, markersize=4)
            ax.set_ylim(*limits[key])
            ax.grid(axis="y", alpha=0.25)
            ax.set_xlabel(xlabel, fontsize=7.5)
            if row_index == 0:
                ax.set_title(ylabel, fontsize=8)
            if xlabel.startswith("Directions"):
                ax.set_xticks(xs)
            else:
                ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
                ax.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.20"])
            if key == "error" and xlabel.startswith("Directions"):
                low, high = min(data["error"]), max(data["error"])
                ax.annotate(
                    f"range {low:.2f}–{high:.2f}%\n(Δ {high - low:.2f} pp)",
                    xy=(xs[1], low), xytext=(0.5, 0.18), textcoords="axes fraction",
                    ha="center", va="bottom", fontsize=7.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
                )
        axes[row_index, 0].set_ylabel(f"n = {data['n']}", fontsize=7.5, color=GRAY)
        fig.text(
            0.012, axes[row_index, 0].get_position().y1 - 0.01, row_title,
            fontsize=7.5, fontweight="bold", va="top", ha="left", linespacing=1.3,
        )
        fig.text(
            0.012, axes[row_index, 0].get_position().y1 + 0.012, chr(ord("A") + row_index),
            fontsize=10, fontweight="bold", va="bottom", ha="left",
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "fig_review_sensitivity.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.output_dir / "fig_review_sensitivity.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Generated 18-scene and 50-case sensitivity curves from fixed records.")


if __name__ == "__main__":
    main()
