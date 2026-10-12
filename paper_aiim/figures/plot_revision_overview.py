"""Simplified decision-layer and cohort diagrams (auxiliary, not compiled).

These reduced text-box / box-plot drawings were explored during the revision
and then withdrawn: the manuscript keeps the original illustrated designs of
``fig_pipeline``, ``fig_dataset_protocol`` and ``fig_main_results``, which are
drawn at their printed width by ``generate_aiim_figures.py``. To avoid
overwriting those compiled figures, this script writes its outputs under the
``*_simplified`` stems. ``fig_ct_controlled_distributions`` is likewise an
auxiliary diagnostic that is not referenced by the manuscript.

Usage: python plot_revision_overview.py [--only fig_dataset_protocol_simplified ...]
"""
from pathlib import Path
import argparse
import json
import statistics
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "figures"
COLORS = ("#0072B2", "#009E73", "#D55E00", "#6F4E7C")


def box(ax, x, y, width, height, text, color, fontsize=10, pad=0.012):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle=f"round,pad={pad}",
        edgecolor=color, facecolor="#F5F7FA", lw=1.5,
    ))
    ax.text(x + width / 2, y + height / 2, text,
            ha="center", va="center", fontsize=fontsize, linespacing=1.4)


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    stages = (
        (0.02, 0.58, "1  Multiview response\nMesh, loads, camera geometry\nSurface tracks + measured force"),
        (0.54, 0.58, "2  Learned initializer\nBackground, ratio, center, radius\nCalibrated under fixed splits"),
        (0.54, 0.15, "3  Prior-conditioned FEM probes\nRelative objective reduction\nContinue only with local support"),
        (0.02, 0.15, "4  Accept or retain\nFinal objective criterion + guards\nNo guarantee of material truth"),
    )
    for index, (x, y, text) in enumerate(stages):
        box(ax, x, y, 0.43, 0.31, text, COLORS[index])
    for first, last in (
        ((0.46, 0.735), (0.53, 0.735)),
        ((0.755, 0.565), (0.755, 0.475)),
        ((0.525, 0.305), (0.465, 0.305)),
    ):
        ax.add_patch(FancyArrowPatch(first, last, arrowstyle="-|>", mutation_scale=14, lw=1.5))
    save(fig, "fig_pipeline_simplified")


def dataset():
    manifest = json.loads((ROOT / "dataset" / "sim_lung_ai_v2_multiview250" / "manifest.json").read_text(encoding="utf-8-sig"))
    counts = {split: sum(row["split"] == split for row in manifest["patients"]) for split in ("train", "val", "test")}
    if counts != {"train": 150, "val": 50, "test": 50}:
        raise ValueError(f"Unexpected procedural split: {counts}")
    # Compact (about one third of a page) so that the three text boxes no
    # longer occupy a page of their own.
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(ax, 0.02, 0.56, 0.455, 0.36,
        "Procedural synthetic cohort\n250 generated geometries\n150 train / 50 validation / 50 test",
        COLORS[0], fontsize=8.5, pad=0.01)
    box(ax, 0.525, 0.56, 0.455, 0.36,
        "CT-derived geometry cohort\n27 CT case-record geometries\n18 train / 3 validation / 6 test",
        COLORS[1], fontsize=8.5, pad=0.01)
    box(ax, 0.02, 0.04, 0.96, 0.42,
        "Shared observation structure: 4 loads × 3 calibrated views × 7 frames\n"
        "Single-inclusion material targets and all motion are synthetic\n"
        "CT test: 120 scenes within 6 geometry clusters, not 120 anatomies",
        COLORS[2], fontsize=8.5, pad=0.01)
    save(fig, "fig_dataset_protocol_simplified")


def primary_results():
    source = ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json"
    evidence = json.loads(source.read_text(encoding="utf-8-sig"))
    rows = evidence["records"]
    if len(rows) != 50 or len({row["patient_id"] for row in rows}) != 50:
        raise ValueError("Expected exactly 50 unique primary test cases")
    heterogeneous = [row for row in rows if row["inclusion_ratio_true"] > 1 + 1e-8]
    metrics = (
        ("Background modulus", [abs(row["E_background_estimated"] / row["E_background_true"] - 1) for row in rows], 0.15, "E_background_median_relative_error"),
        ("Inclusion ratio", [abs(row["inclusion_ratio_estimated"] / row["inclusion_ratio_true"] - 1) for row in rows], 0.25, "inclusion_ratio_median_relative_error"),
        ("Center", [math.dist(row["center_fraction_estimated"], row["center_fraction_true"]) for row in heterogeneous], 0.10, "center_error_normalized_median"),
        ("Radius", [abs(row["radius_fraction_estimated"] / row["radius_fraction_true"] - 1) for row in heterogeneous], 0.12, "radius_relative_error_median"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5), constrained_layout=True)
    for ax, (title, values, gate, key), color in zip(axes.flat, metrics, COLORS):
        median = statistics.median(values)
        if not math.isclose(median, evidence["test"][key], abs_tol=1e-8):
            raise ValueError(f"Primary plot definition does not reproduce {key}")
        percent = [value * 100 for value in values]
        ax.boxplot(percent, positions=[1], widths=0.25, showfliers=False)
        xs = [1 + ((index * 37) % 101 - 50) / 350 for index in range(len(values))]
        ax.scatter(xs, percent, s=14, color=color, alpha=0.65)
        ax.axhline(gate * 100, ls="--", color="#5B6573", lw=1)
        ax.set_title(f"{title}: median {100 * median:.2f}%", fontsize=10)
        ax.set_xticks([1], [f"{len(values)} eligible cases"])
        ax.set_ylabel("Error (%)")
        ax.grid(axis="y", alpha=0.2)
    save(fig, "fig_main_results_simplified")


def ct_controls():
    source = ROOT / "results" / "ion_ct_expanded30_physics16" / "benchmark.json"
    evidence = json.loads(source.read_text(encoding="utf-8-sig"))
    methods = (
        ("pca_ridge", "PCA-Ridge"), ("mesh_gnn", "Raw MeshGNN"),
        ("ours_hybrid_initializer", "Raw hybrid"), ("fem_fixed_init", "Fixed FEM"),
        ("fem_learned_screened_map", "Raw-prior MAP"),
        ("fem_oracle_region_force", "True-input ref."),
    )
    metrics = (
        ("E_background_relative_error", "Background modulus", 15),
        ("inclusion_ratio_relative_error", "Inclusion ratio", 25),
        ("center_error_normalized", "Center", 10),
        ("radius_relative_error", "Radius", 12),
    )
    expected_ids = None
    for method, _ in methods:
        ids = {row["patient_id"] for row in evidence["records"] if row["method"] == method}
        if len(ids) != 30 or (expected_ids is not None and ids != expected_ids):
            raise ValueError(f"Controlled CT methods are not paired: {method}")
        expected_ids = ids
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.4), constrained_layout=True)
    for ax, (metric, title, gate) in zip(axes.flat, metrics):
        values = [
            [100 * row[metric] for row in evidence["records"]
             if row["method"] == method and row.get(metric) is not None]
            for method, _ in methods
        ]
        ax.boxplot(values, tick_labels=[label for _, label in methods], showfliers=True,
                   flierprops={"markersize": 2}, widths=0.55)
        ax.axhline(gate, ls="--", color="#5B6573", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Error (%)")
        ax.tick_params(axis="x", labelrotation=55, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.grid(axis="y", alpha=0.2)
    save(fig, "fig_ct_controlled_distributions")


def main():
    builders = {
        "fig_pipeline_simplified": pipeline,
        "fig_dataset_protocol_simplified": dataset,
        "fig_main_results_simplified": primary_results,
        "fig_ct_controlled_distributions": ct_controls,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(builders), default=None,
                        help="Regenerate only the listed figure stems.")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    for stem, builder in builders.items():
        if args.only is None or stem in args.only:
            builder()
            print(f"wrote {stem}")
    print("Generated simplified (auxiliary, *_simplified) inference and cohort diagrams.")


if __name__ == "__main__":
    main()
