"""Four-panel inverse-arm leaderboard (Supplementary Material).

Ranking pool = methods scored on the same split and the same four endpoints.
Panels A-C restate the locked scoreboard values
(``results/inverse_arm_scoreboard_20260825.json`` /
``results/ion_ct_ratio_optimization/fair_calibrate/official_lock.json``).
Panel D is computed from the complete official-prior screened-FEM audit
(``results/ion_ct_official_prior_screened_fem120_v1/records.jsonl``) and
cross-checked against its ``summary.json``: per-geometry medians of the
initializer-versus-truth and audited-versus-truth background error.

The figure is drawn at its final 6.5 in width so that every label is >= 7 pt
on the printed page; estimator categories use short labels whose full names
are listed in the in-figure key. Every per-panel legend sits between the
panel title and the axes, and every explanatory note lives in the key block
under the panels, so no legend or annotation overlaps a plotted region.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / "results" / "ion_ct_official_prior_screened_fem120_v1"

C = {
    "ours": "#0072B2",
    "screened": "#CC79A7",
    "ctcal": "#009E73",
    "prior": "#56B4E9",
    "pinn": "#E69F00",
    "learn": "#999999",
    "fem": "#D55E00",
    "ctrl": "#BBBBBB",
    "audit": "#6F4E7C",
    "ink": "#222222",
    "muted": "#666666",
}

# Synthetic 50, same cases. Sorted by E median.
SYN_RANK = [
    ("Response calibrator", "Cal", 3.82, "prior"),
    ("Learned-prior screened MAP", "MAP", 4.11, "ours"),
    ("Locked-ρ screened FEM", "FEM-ρ", 4.11, "screened"),
    ("Unscreened PINN inverse", "PINN", 4.55, "pinn"),
    ("Four-start FEM (no screen)", "FEM-4", 29.64, "fem"),
    ("Fixed-start FEM", "FEM-f", 34.14, "ctrl"),
]

SYN_ENDPOINTS = {
    "short": ["Cal", "MAP", "FEM-ρ", "PINN", "FEM-4", "FEM-f"],
    "E": [3.82, 4.11, 4.11, 4.55, 29.64, 34.14],
    "ratio": [18.69, 18.69, 18.69, 17.87, 33.78, 44.87],
    "center": [6.96, 6.96, 6.96, 6.96, np.nan, np.nan],
    "radius": [9.93, 9.93, 9.93, 9.93, np.nan, np.nan],
}

# CT 120 ratio geometry-pass (6 held-out geometries).
CT_GEOM = [
    ("Calibrated hybrid initializer (17.77%)", "Hyb-cal", 6, 0, "ctcal"),
    ("PINN, official start", "PINN-o", 3, 3, "pinn"),
    ("PINN, hybrid-initializer start", "PINN-h", 2, 4, "pinn"),
    ("Raw MeshGNN (26.95%)", "GNN", 2, 4, "learn"),
    ("Ridge (same protocol)", "R", 1, 5, "learn"),
    ("PCA–Ridge (same protocol)", "PCR", 0, 6, "learn"),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# Vertical stack above every panel: axes top -> legend (<= 2 rows of 7 pt)
# -> title. ``TITLE_PAD`` is in points; the panel letter is placed at the
# same height as the title so that letters and titles align across a row.
TITLE_PAD = 30.0
LEGEND_KW = dict(
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(0.0, 1.0),
    borderaxespad=0.0,
    fontsize=7,
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=1.0,
)


def _panel_label(ax, letter: str, x: float = -0.14) -> None:
    axes_height_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72.0
    y = 1.0 + (TITLE_PAD + 9.0) / axes_height_pt
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color=C["ink"],
    )


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_rank(ax) -> None:
    names = [row[1] for row in SYN_RANK][::-1]
    vals = [row[2] for row in SYN_RANK][::-1]
    cols = [C[row[3]] for row in SYN_RANK][::-1]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=cols, height=0.72, edgecolor="none", zorder=2)
    for i, (name, _short, value, _kind) in enumerate(SYN_RANK[::-1]):
        if name == "Response calibrator":
            bars[i].set_edgecolor(C["ink"])
            bars[i].set_linewidth(1.0)
            ax.text(value + 0.6, y[i], f"{value:.2f}  (#1 E)", va="center", fontsize=7.2, color=C["prior"], fontweight="bold")
        else:
            ax.text(value + 0.6, y[i], f"{value:.2f}", va="center", fontsize=7.2, color=C["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Median background-E error (%), lower is better")
    ax.set_xlim(0, 46)
    ax.axvline(3.82, color=C["prior"], ls="--", lw=0.8, zorder=1, alpha=0.85, label="Cal 3.82% (E floor)")
    ax.axvline(4.11, color=C["ours"], ls=":", lw=0.8, zorder=1, alpha=0.85, label="MAP 4.11%")
    ax.set_title("Synthetic 50: background-E ranking", loc="left", pad=TITLE_PAD)
    ax.legend(ncol=2, **LEGEND_KW)
    ax.grid(axis="x", color="#eeeeee", lw=0.5, zorder=0)
    _despine(ax)
    ax._rank_handles = [
        mpatches.Patch(color=C["prior"], label="Response calibrator (no FEM)"),
        mpatches.Patch(color=C["ours"], label="Learned-prior screened MAP (synthetic)"),
        mpatches.Patch(color=C["screened"], label="Locked-ρ screened FEM"),
        mpatches.Patch(color=C["pinn"], label="Unscreened PINN"),
        mpatches.Patch(color=C["fem"], label="Unscreened four-start FEM"),
        mpatches.Patch(color=C["ctrl"], label="Fixed-start control"),
        mpatches.Patch(color=C["ctcal"], label="Calibrated hybrid initializer (CT)"),
        mpatches.Patch(color=C["learn"], label="Learning baseline"),
        mpatches.Patch(color=C["audit"], label="Official-prior screened FEM audit (CT)"),
    ]
    _panel_label(ax, "A", x=-0.20)


def panel_endpoints(ax) -> None:
    short = SYN_ENDPOINTS["short"]
    x = np.arange(len(short))
    width = 0.20
    series = [
        ("E", SYN_ENDPOINTS["E"], C["ours"]),
        ("ratio", SYN_ENDPOINTS["ratio"], C["pinn"]),
        ("center", SYN_ENDPOINTS["center"], "#009E73"),
        ("radius", SYN_ENDPOINTS["radius"], "#C44E52"),
    ]
    for i, (label, values, color) in enumerate(series):
        offset = (i - 1.5) * width
        plotted = [0.0 if np.isnan(value) else value for value in values]
        bars = ax.bar(x + offset, plotted, width, color=color, label=label, zorder=2)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                bar.set_alpha(0.0)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Median relative error (%)")
    ax.set_ylim(0, 50)
    ax.set_title("Synthetic 50: four endpoints", loc="left", pad=TITLE_PAD)
    ax.axhline(3.82, color=C["prior"], ls="--", lw=0.7, alpha=0.8)
    # Legend between title and axes (two rows, series first, floor last);
    # the center/radius caveat is written in the key block below the panels
    # instead of inside the axes.
    handles = [mpatches.Patch(color=color, label=label) for label, _values, color in series]
    handles.append(Line2D([0], [0], color=C["prior"], ls="--", lw=0.7, alpha=0.8, label="E floor 3.82% (Cal)"))
    ax.legend(handles=handles, ncol=3, **LEGEND_KW)
    ax.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
    _despine(ax)
    _panel_label(ax, "B")


def panel_geom(ax) -> None:
    short = [row[1] for row in CT_GEOM]
    ok = np.array([row[2] for row in CT_GEOM], dtype=float)
    lost = np.array([row[3] for row in CT_GEOM], dtype=float)
    cols = [C[row[4]] for row in CT_GEOM]
    x = np.arange(len(short))
    ax.bar(x, ok, color=cols, zorder=2, label="Passing (median ≤ 25%)")
    ax.bar(
        x,
        lost,
        bottom=ok,
        color="#F0E6E6",
        edgecolor="#AA5555",
        linewidth=0.6,
        zorder=2,
        label="Failing",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.margins(x=0.03)
    ax.set_ylabel("Test geometries (of 6)")
    ax.set_ylim(0, 7.2)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_title("CT 120: ratio geometry-pass (≤ 25%)", loc="left", pad=TITLE_PAD)
    for i, (_, _, n_ok, n_lost, _) in enumerate(CT_GEOM):
        ax.text(
            i,
            n_ok + n_lost + 0.18,
            f"{n_ok}/6",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=C["ours"] if not n_lost else "#AA3333",
        )
    ax.legend(ncol=2, **LEGEND_KW)
    _despine(ax)
    ax.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
    _panel_label(ax, "C")


def load_audit() -> tuple[list[str], np.ndarray, np.ndarray, list[int], dict[str, float]]:
    """Per-geometry initializer-vs-truth and audited-vs-truth E_bg medians."""
    records_path = AUDIT_DIR / "records.jsonl"
    summary = json.loads((AUDIT_DIR / "summary.json").read_text(encoding="utf-8-sig"))
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 120 or len({row["patient_id"] for row in rows}) != 120:
        raise ValueError(f"Expected 120 unique audit scenes, found {len(rows)}")
    if {row["method"] for row in rows} != {"official_prior_screened_fem"}:
        raise ValueError("Unexpected method label in the official-prior audit records")
    by_geometry: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_geometry[row["geometry_id"]].append(row)
    geometry_ids = sorted(by_geometry)
    if len(geometry_ids) != 6 or any(len(by_geometry[g]) != 20 for g in geometry_ids):
        raise ValueError("Expected six geometries with 20 scenes each")
    init = np.asarray([
        np.median([abs(r["initial_E_background"] / r["E_background_true"] - 1.0) for r in by_geometry[g]])
        for g in geometry_ids
    ]) * 100
    final = np.asarray([
        np.median([abs(r["E_background_estimated"] / r["E_background_true"] - 1.0) for r in by_geometry[g]])
        for g in geometry_ids
    ]) * 100
    recorded = np.asarray([
        np.median([r["E_background_relative_error"] for r in by_geometry[g]]) for g in geometry_ids
    ]) * 100
    if not np.allclose(final, recorded, atol=1e-6):
        raise ValueError("Recomputed audited errors do not match the recorded per-scene errors")
    accepted = [int(sum(bool(r["refinement_accepted"]) for r in by_geometry[g])) for g in geometry_ids]
    pooled = {
        "init": float(np.median([abs(r["initial_E_background"] / r["E_background_true"] - 1.0) for r in rows]) * 100),
        "final": float(np.median([r["E_background_relative_error"] for r in rows]) * 100),
        "accepted": int(sum(bool(r["refinement_accepted"]) for r in rows)),
        "n": len(rows),
    }
    # Cross-check against the regenerated summary.
    locked = summary["screened_fem_locked"]
    if locked["n"] != 120:
        raise ValueError("summary.json does not describe the 120-scene audit")
    if abs(locked["E_background_median_relative_error"]["median"] * 100 - pooled["final"]) > 0.05:
        raise ValueError("Pooled audited E_bg median disagrees with summary.json")
    if abs(summary["reference_official_hybrid_ct"]["E_background_median_relative_error"] * 100 - pooled["init"]) > 0.05:
        raise ValueError("Pooled initializer E_bg median disagrees with summary.json")
    if abs(locked["refinement_acceptance_rate"] * 120 - pooled["accepted"]) > 0.5:
        raise ValueError("Acceptance count disagrees with summary.json")
    return geometry_ids, init, final, accepted, pooled


def panel_audit(ax) -> None:
    geometry_ids, init, final, accepted, pooled = load_audit()
    x = np.arange(len(geometry_ids))
    width = 0.38
    ax.bar(x - width / 2, init, width, color=C["ctcal"], zorder=2, label="Initializer (calibrated hybrid)")
    ax.bar(x + width / 2, final, width, color=C["audit"], zorder=2, label="After screened audit")
    for xi, (a, b, n_acc) in enumerate(zip(init, final, accepted)):
        ax.text(xi, max(a, b) + 1.0, f"{n_acc}/20", ha="center", va="bottom", fontsize=7.2, color=C["muted"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{i + 1}" for i in range(len(geometry_ids))])
    ax.set_xlabel("Held-out CT geometry (20 scenes each)")
    ax.set_ylabel("Median background-E error (%)")
    ax.set_ylim(0, max(final.max(), init.max()) * 1.25)
    ax.set_title("CT 120: official-prior FEM audit", loc="left", pad=TITLE_PAD)
    # Legend between title and axes (one entry per row so that it stays
    # inside the panel width); the pooled numbers and the meaning of the
    # n/20 labels are written in the key block below the panels.
    ax.legend(ncol=1, **LEGEND_KW)
    ax._geometry_ids = geometry_ids
    ax._pooled = pooled
    ax.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
    _despine(ax)
    _panel_label(ax, "D")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Destination directory for the generated leaderboard PDF/PNG/JPG.")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    # Final 6.5 in width (\linewidth of the 3p elsarticle page); the height
    # leaves room for the title/legend stack above every panel and for the
    # key block (which also carries the per-panel notes) underneath.
    fig = plt.figure(figsize=(6.5, 7.7))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
        wspace=0.40, hspace=0.70, left=0.095, right=0.985, top=0.835, bottom=0.24,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    panel_rank(ax_a)
    panel_endpoints(ax_b)
    panel_geom(ax_c)
    panel_audit(ax_d)
    fig.legend(
        handles=ax_a._rank_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.972),
        fontsize=7,
        handlelength=1.2,
        columnspacing=1.2,
    )
    fig.suptitle(
        "Fair protocol: estimator arms on matched splits and endpoints",
        fontsize=9.5,
        fontweight="bold",
        y=0.993,
    )
    pooled = ax_d._pooled
    key_b = " · ".join(f"{short} = {name}" for name, short, _, _ in SYN_RANK)
    key_c = " · ".join(f"{short} = {name}" for name, short, _, _, _ in CT_GEOM)
    key_d = ", ".join(f"G{i + 1} = {g.replace('geom_', '')[:8]}" for i, g in enumerate(ax_d._geometry_ids))
    note_b = (
        "Panel B: center/radius share the predicted sphere and are not defined for FEM-4 / FEM-f."
    )
    note_d = (
        f"Panel D: n/20 above each pair = accepted refinements; pooled median "
        f"{pooled['init']:.2f}% → {pooled['final']:.2f}% ({pooled['accepted']}/{pooled['n']} accepted); "
        "ratio fixed at the locked 17.77%."
    )
    wrap = dict(width=110, break_on_hyphens=False)
    key_text = "\n".join(
        textwrap.wrap("Key, panels A–B (synthetic 50): " + key_b, **wrap)
        + textwrap.wrap(note_b, **wrap)
        + textwrap.wrap("Key, panel C (CT 120): " + key_c, **wrap)
        + textwrap.wrap("Key, panel D (CT 120 official-prior audit): " + key_d, **wrap)
        + textwrap.wrap(note_d, **wrap)
    )
    fig.text(0.03, 0.012, key_text, fontsize=7, va="bottom", ha="left", linespacing=1.35)
    pdf = output_dir / "fig_inverse_leaderboard.pdf"
    jpg = output_dir / "fig_inverse_leaderboard.jpg"
    png = output_dir / "fig_inverse_leaderboard.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(jpg, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {jpg}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
