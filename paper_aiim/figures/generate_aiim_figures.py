#!/usr/bin/env python3
"""Generate reproducible AIIM figures from frozen JSON and case tensors.

Quantitative results are loaded from publication artifacts, cohort summaries
from the dataset manifest, and the qualitative synthetic example from its four
saved ``gt.pt`` experiments.

Figures are laid out at their final printed width (6.5 in, ``FINAL_WIDTH_IN``)
so that every label is at least ``MIN_FONT_PT`` on the page. Use ``--only`` to
regenerate a subset, e.g.::

    python generate_aiim_figures.py --output-dir <figures dir> --only fig_fair_fem
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "lung_mesh_ai_v2"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "sim_lung_ai_v2_multiview250"
SOURCES = {
    "gates": Path("publication_gates_final.json"),
    "calibration": Path("node_response_calibrator_strict.json"),
    "fair_fem": Path("fair_multiview_fem50_strict/fair_multiview_summary.json"),
    "multiseed": Path("multiseed_node_summary/multiseed_aggregate.json"),
    "baselines": Path("response_baseline_benchmark.json"),
    "ood": Path("../ion_geometry_ood/formal_validation/aggregate.json"),
}
ION_FORMAL_ROOT = PROJECT_ROOT / "results" / "ion_geometry_ood" / "formal_validation"
ION_MESH_ROOT = PROJECT_ROOT / "results" / "ion_geometry_ood" / "deidentified_meshes27"
ION_COHORT_QC = PROJECT_ROOT / "results" / "ion_geometry_ood" / "geometry_qc_27.json"
ION_SYNTH_DATASET = PROJECT_ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
ION_SYNTH_RESULTS = PROJECT_ROOT / "results" / "ion_ct_expanded120_final"
ION_PHYSICS_RESULTS = PROJECT_ROOT / "results" / "ion_ct_expanded30_physics16"
ION_FIGURE_ASSETS = (
    PROJECT_ROOT
    / "results"
    / "ion_ct_synthetic_mechanics"
    / "deidentified_figure_assets"
)
FIG_WIDTH = 15.4
CT_METHOD_ORDER = (
    "ridge",
    "pls",
    "extra_trees",
    "pca_ridge",
    "training_population",
    "mesh_gnn",
    "ours_hybrid_initializer",
    "fem_fixed_init",
    "fem_deterministic_multistart",
    "fem_learned_screened_map",
    "ours_ood_screened_fem",
    "fem_oracle_region_force",
)
CT_METHOD_LABELS = {
    "ridge": "Ridge",
    "pls": "Partial least squares",
    "extra_trees": "ExtraTrees",
    "pca_ridge": "PCA–Ridge",
    "training_population": "Training population",
    "mesh_gnn": "Raw MeshGNN",
    "ours_hybrid_initializer": "Uncalibrated hybrid initializer",
    "fem_fixed_init": "Fixed-start FEM",
    "fem_deterministic_multistart": "Deterministic multistart FEM",
    "fem_learned_screened_map": "Uncalibrated hybrid-prior screened MAP",
    "ours_ood_screened_fem": "Uncalibrated hybrid-prior OOD fallback",
    "fem_oracle_region_force": "True-region/true-force reference*",
}
# Short category labels used on crowded estimator axes; the full names are
# always written in an in-figure key (see ``_abbreviation_key``).
CT_METHOD_ABBREV = {
    "ridge": "R",
    "pls": "PLS",
    "extra_trees": "ET",
    "pca_ridge": "PCR",
    "training_population": "Pop",
    "mesh_gnn": "GNN",
    "ours_hybrid_initializer": "Hyb",
    "fem_fixed_init": "FEM-f",
    "fem_deterministic_multistart": "FEM-m",
    "fem_learned_screened_map": "MAP",
    "ours_ood_screened_fem": "OOD",
    "fem_oracle_region_force": "Ref*",
}
CT_TIER_COLORS = {
    "common_input": BLUE if "BLUE" in globals() else "#0072B2",
    "secondary": ORANGE if "ORANGE" in globals() else "#D55E00",
    "oracle": PURPLE if "PURPLE" in globals() else "#6F4E7C",
}
# Final printed width of every manuscript/supplement figure (\linewidth).
FINAL_WIDTH_IN = 6.5
MIN_FONT_PT = 7.0

# Okabe--Ito-derived, distinguishable in grayscale with the accompanying hatches.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#6F4E7C"
GRAY = "#5B6573"
LIGHT = "#E8EDF2"
INK = "#17212B"
PALE_BLUE = "#E8F2F8"
PALE_GREEN = "#E6F4EF"
PALE_ORANGE = "#FAEDE8"
PALE_PURPLE = "#F0EBF3"


def configure_style() -> None:
    """Shared style: figures are drawn at their final 6.5 in width so that
    every label is >= 7 pt on the printed page (DejaVu Sans throughout)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.titlesize": 11,
            "figure.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sources(results_root: Path) -> dict[str, dict[str, Any]]:
    payloads = {
        name: load_json(results_root / relative) for name, relative in SOURCES.items()
    }
    gates = payloads["gates"]
    if not gates.get("publication_ready"):
        raise ValueError("publication_gates_final.json is not publication-ready")
    failed = [name for name, row in gates["gates"].items() if not row["passed"]]
    if failed:
        raise ValueError(f"Frozen publication gates failed: {failed}")
    if payloads["ood"].get("material_ground_truth_metrics_reported") is not False:
        raise ValueError("OOD artifact unexpectedly reports material ground truth")
    return payloads


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.10,
    fontsize: float = 10,
) -> None:
    text_method = getattr(ax, "text2D", ax.text)
    text_method(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _abbreviation_key(
    fig: plt.Figure,
    pairs: list[tuple[str, str]],
    *,
    y: float = 0.01,
    ncol: int = 3,
    fontsize: float = 7.0,
    title: str | None = "Estimator key",
    left: float = 0.07,
    right: float = 0.99,
    inline: bool = False,
) -> None:
    """Write ``ABBR = full name`` entries as an in-figure key (>= 7 pt)."""
    if inline:
        text = " · ".join(f"{short} = {full}" for short, full in pairs)
        if title:
            text = f"{title}: {text}"
        # ~105 characters of 7 pt DejaVu Sans fit in 6.3 in; wrap beyond that.
        text = "\n".join(textwrap.wrap(text, width=105))
        fig.text(left, y, text, fontsize=fontsize, va="bottom", ha="left", linespacing=1.35)
        return
    rows = -(-len(pairs) // ncol)
    columns = [pairs[index * rows:(index + 1) * rows] for index in range(ncol)]
    span = right - left
    for column_index, column in enumerate(columns):
        if not column:
            continue
        text = "\n".join(f"{short} = {full}" for short, full in column)
        fig.text(
            left + span * column_index / ncol,
            y,
            text,
            fontsize=fontsize,
            va="bottom",
            ha="left",
            linespacing=1.35,
        )
    if title:
        line_height = fontsize * 1.35 / 72.0 / fig.get_size_inches()[1]
        fig.text(
            left,
            y + line_height * (rows + 0.25),
            title,
            fontsize=fontsize,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def _ct_key_pairs(methods: list[str]) -> list[tuple[str, str]]:
    return [
        (CT_METHOD_ABBREV[method], CT_METHOD_LABELS[method].replace("*", ""))
        for method in methods
    ]


def _broken_axes(
    fig: plt.Figure,
    spec,
    *,
    sharex: plt.Axes | None,
    upper_ratio: float = 0.32,
    gap: float = 0.10,
) -> tuple[plt.Axes, plt.Axes]:
    """Split one gridspec slot into an upper (compressed) and a lower axes.

    Both axes receive the same artists; the caller sets disjoint y-limits so
    that a few extreme values no longer crush the bulk of the distribution.
    The break is marked with the usual slanted ticks on both spines.
    """
    sub = spec.subgridspec(2, 1, height_ratios=(upper_ratio, 1.0), hspace=gap)
    upper = fig.add_subplot(sub[0], sharex=sharex)
    lower = fig.add_subplot(sub[1], sharex=sharex)
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    marks = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=7,
        linestyle="none",
        color=INK,
        mec=INK,
        mew=0.8,
        clip_on=False,
    )
    upper.plot([0, 1], [0, 0], transform=upper.transAxes, **marks)
    lower.plot([0, 1], [1, 1], transform=lower.transAxes, **marks)
    return upper, lower


def _broken_ylabel(upper: plt.Axes, lower: plt.Axes, text: str) -> None:
    """Put one y-label on the lower axes, centred on the full slot."""
    up, lo = upper.get_position(), lower.get_position()
    centre = 0.5 * (up.y1 + lo.y0)
    lower.set_ylabel(text, y=(centre - lo.y0) / lo.height)


def _slot_panel_label(fig: plt.Figure, spec, label: str, *, x: float, y: float) -> None:
    """Panel letter for a split slot, at the same offset as ``panel_label``."""
    box = spec.get_position(fig)
    fig.text(
        box.x0 + x * box.width,
        box.y0 + y * box.height,
        label,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "paper_aiim/figures/generate_aiim_figures.py",
        "Title": stem,
    }
    # ``bbox_inches="tight"`` silently widens the page when an artist pokes
    # out of the canvas, which shrinks every font once the PDF is scaled to
    # \linewidth. Flag that so the layout can be fixed instead.
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    width_in, height_in = fig.get_size_inches()
    if tight.width > width_in * 1.02 or tight.height > height_in * 1.02:
        print(
            f"  WARNING {stem}: tight bbox {tight.width:.2f}x{tight.height:.2f} in exceeds "
            f"figure size {width_in:.2f}x{height_in:.2f} in; labels will shrink when scaled"
        )
    # A small pad keeps the page within ~1 % of the designed width so that the
    # point sizes chosen here are the point sizes printed at \linewidth.
    fig.savefig(
        output_dir / f"{stem}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=metadata,
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Software": metadata["Creator"], "Title": stem},
    )
    plt.close(fig)


def verify_assets(output_dir: Path, stems: tuple[str, ...]) -> None:
    """Validate all assets; CT pixel panels are the only allowed PDF rasters."""
    raster_allowed = {
        "fig_ion_ct_synthetic_pipeline",
        "fig_ion_ct_material_recovery",
        "fig_ion_ct_physics_material_maps",
    }
    for stem in stems:
        pdf_path = output_dir / f"{stem}.pdf"
        png_path = output_dir / f"{stem}.png"
        if pdf_path.stat().st_size == 0 or png_path.stat().st_size == 0:
            raise RuntimeError(f"Empty generated asset for {stem}")
        pdf_bytes = pdf_path.read_bytes()
        if not pdf_bytes.startswith(b"%PDF"):
            raise RuntimeError(f"Invalid PDF output for {stem}")
        if b"/Subtype /Image" in pdf_bytes and stem not in raster_allowed:
            raise RuntimeError(f"{pdf_path} contains raster image objects; editable vector output required")
        with Image.open(png_path) as image:
            dpi = image.info.get("dpi")
            if dpi is None or any(abs(float(value) - 300.0) > 0.1 for value in dpi):
                raise RuntimeError(f"{png_path} is not tagged as 300 dpi: {dpi}")
            # 6.5 in at 300 dpi is 1950 px; short single-row figures are allowed.
            if image.format != "PNG" or image.width < 1800 or image.height < 500:
                raise RuntimeError(f"Unexpected PNG output for {stem}: {image.format}, {image.size}")
            print(f"  verified {stem}: {image.width}x{image.height} px, {dpi[0]:.1f} dpi")


# Figure 1 stage boxes: (title, detail, edge colour, fill). The graphical
# abstract (generate_graphical_abstract.py) redraws the same boxes and icons
# through draw_pipeline_stages() with its own shortened labels.
PIPELINE_STAGES: tuple[tuple[str, str, str, str], ...] = (
    (
        "1  Geometry and\nconstraints",
        "surface mesh • tetrahedra\nboundary • contacts",
        GRAY,
        "#F4F6F8",
    ),
    (
        "2  Multiview motion",
        "4 loads • 3 views\n7-frame tracks + force",
        BLUE,
        PALE_BLUE,
    ),
    (
        "3  Response encoding",
        "visibility mask\ntemporal response tokens",
        BLUE,
        "#F2F7FB",
    ),
    (
        "4  Hybrid material\ninitializer",
        "MeshGNN field + calibrated radius\n"
        r"$\theta_0=(\hat E_b,\hat\rho_E,\hat{\mathbf{c}},\hat R)$",
        GREEN,
        PALE_GREEN,
    ),
    (
        "5  Physics decision",
        "seven probes\nretain prior or bounded MAP",
        ORANGE,
        PALE_ORANGE,
    ),
    (
        "6  Audited output",
        "accept or retain\nfield • interval • diagnostics",
        PURPLE,
        PALE_PURPLE,
    ),
)
PIPELINE_BOX_W, PIPELINE_BOX_H = 1.78, 1.00


def draw_pipeline_stages(
    ax: plt.Axes,
    positions: Sequence[tuple[float, float]],
    *,
    stages: Sequence[tuple[str, str, str, str]] = PIPELINE_STAGES,
    box_w: float = PIPELINE_BOX_W,
    box_h: float = PIPELINE_BOX_H,
    icon_dy: float = 0.50,
    title_dy: tuple[float, float] = (0.14, 0.15),
    detail_dy: float = 0.16,
    title_pt: float = 8.5,
    detail_pt: float = 7.1,
    label_pt: float = 7.1,
    check_pt: float = 11.0,
    box_lw: float = 1.3,
    rounding_size: float = 0.07,
    marker_scale: float = 1.0,
    lw_scale: float = 1.0,
    arrow_scale: float = 7.0,
    decision_area: float = 300.0,
    decision_arm: tuple[float, float] = (0.09, 0.07),
    fem_lattice: bool = True,
) -> list[tuple[FancyBboxPatch, list[plt.Text]]]:
    """Draw the six illustrated stage boxes of Figure 1 (box, title, detail, icon).

    ``positions`` holds the lower-left corner of each box in the data units of
    ``ax``, which must be isotropic (inches in ``fig_pipeline``; the graphical
    abstract uses an inset whose data unit is a scaled inch). All geometry
    (``box_w``, ``box_h``, ``icon_dy`` = icon centre above the box bottom,
    ``title_dy`` = title centre below the box top for one-/two-line titles,
    ``detail_dy`` = detail centre above the box bottom, ``decision_arm`` =
    offset of the prior/MAP arrow tails from the decision diamond) follows the
    data units, whereas font sizes, marker areas (``marker_scale``), line
    widths (``lw_scale``), icon arrow heads (``arrow_scale``) and the decision
    diamond area are display points and must be chosen for the target scale;
    ``fem_lattice`` switches the small FEM lattice beside the MAP branch off
    where the prior/MAP labels would collide with it at small sizes.
    With the defaults this reproduces Figure 1 exactly. Returns
    ``[(box_patch, [title_text, detail_text]), ...]`` for layout checks.
    """
    boxes: list[tuple[FancyBboxPatch, list[plt.Text]]] = []
    for (left, y0), (title, detail, color, fill) in zip(positions, stages, strict=True):
        patch = FancyBboxPatch(
            (left, y0),
            box_w,
            box_h,
            boxstyle=f"round,pad=0,rounding_size={rounding_size}",
            facecolor=fill,
            edgecolor=color,
            linewidth=box_lw,
        )
        ax.add_patch(patch)
        two_lines = "\n" in title
        title_text = ax.text(
            left + box_w / 2,
            y0 + box_h - (title_dy[1] if two_lines else title_dy[0]),
            title,
            ha="center",
            va="center",
            fontsize=title_pt,
            fontweight="bold",
            color=color,
            linespacing=1.05,
        )
        detail_text = ax.text(
            left + box_w / 2,
            y0 + detail_dy,
            detail,
            ha="center",
            va="center",
            fontsize=detail_pt,
            color=INK,
            linespacing=1.3,
        )
        boxes.append((patch, [title_text, detail_text]))

    def _icon_centre(index: int) -> tuple[float, float]:
        left, y0 = positions[index]
        return left + box_w / 2, y0 + icon_dy

    # 1: dense curved surface mesh with triangular facets and a fixed boundary.
    cx, cy = _icon_centre(0)
    rows, cols = 5, 7
    mesh_x = np.zeros((rows, cols))
    mesh_y = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            mesh_x[row, col] = cx - 0.50 + 0.14 * col + 0.035 * (row - 2)
            mesh_y[row, col] = cy + 0.14 - 0.07 * row + 0.022 * np.sin(0.9 * col + 0.45 * row)
    for row in range(rows - 1):
        for col in range(cols - 1):
            triangle_a = (
                (mesh_x[row, col], mesh_y[row, col]),
                (mesh_x[row, col + 1], mesh_y[row, col + 1]),
                (mesh_x[row + 1, col], mesh_y[row + 1, col]),
            )
            triangle_b = (
                (mesh_x[row, col + 1], mesh_y[row, col + 1]),
                (mesh_x[row + 1, col + 1], mesh_y[row + 1, col + 1]),
                (mesh_x[row + 1, col], mesh_y[row + 1, col]),
            )
            for triangle, alpha in ((triangle_a, 0.20), (triangle_b, 0.12)):
                ax.add_patch(
                    plt.Polygon(
                        triangle,
                        closed=True,
                        facecolor=GRAY,
                        edgecolor=GRAY,
                        linewidth=0.35 * lw_scale,
                        alpha=alpha,
                        zorder=2,
                    )
                )
    ax.scatter(mesh_x.ravel(), mesh_y.ravel(), s=3.0 * marker_scale, color=GRAY, zorder=3)
    # Fixed posterior boundary cue.
    ax.plot(mesh_x[-1], mesh_y[-1] - 0.03, color=ORANGE, lw=1.2 * lw_scale, solid_capstyle="round", zorder=4)

    # 2: rendered multiview tracks with three cameras.
    cx, cy = _icon_centre(1)
    view_w, view_h = 0.52, 0.30
    view_left, view_bottom = cx - view_w / 2, cy - view_h / 2 + 0.01
    ax.add_patch(
        FancyBboxPatch(
            (view_left, view_bottom),
            view_w,
            view_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#F8FBFD",
            edgecolor=BLUE,
            linewidth=0.7 * lw_scale,
        )
    )
    rng = np.random.default_rng(7)
    node_u = view_left + 0.06 + 0.38 * rng.random(18)
    node_v = view_bottom + 0.05 + 0.20 * rng.random(18)
    frame_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, 5))
    for index, (u_value, v_value) in enumerate(zip(node_u, node_v, strict=True)):
        track_u = u_value + 0.022 * np.arange(5)
        track_v = v_value - 0.014 * np.arange(5)
        ax.plot(track_u, track_v, color=GRAY, lw=0.45 * lw_scale, alpha=0.45, zorder=3)
        ax.scatter(track_u[-1], track_v[-1], s=5 * marker_scale, color=frame_colors[index % len(frame_colors)], zorder=4)
    target = (cx, cy + 0.01)
    cameras = ((cx - 0.60, cy + 0.15), (cx + 0.60, cy + 0.15), (cx, cy - 0.21))
    for camera_x, camera_y in cameras:
        ax.scatter(camera_x, camera_y, marker="s", s=14 * marker_scale, color=BLUE, zorder=5)
        ax.plot((camera_x, target[0]), (camera_y, target[1]), color=BLUE, lw=0.55 * lw_scale, alpha=0.65, zorder=2)

    # 3: temporal token sequence with masked frames.
    cx, cy = _icon_centre(2)
    token_x = cx - 0.46 + 0.14 * np.arange(7)
    token_y = cy + 0.02 + 0.025 * np.sin(np.linspace(0, np.pi, 7))
    ax.plot(token_x, token_y, color=BLUE, lw=0.9 * lw_scale, alpha=0.55, zorder=2)
    for index, (x_value, y_value) in enumerate(zip(token_x, token_y, strict=True)):
        visible = index not in (2, 5)
        ax.scatter(
            x_value,
            y_value,
            s=(20 if visible else 11) * marker_scale,
            color=BLUE if visible else LIGHT,
            edgecolor=BLUE,
            linewidth=0.6 * lw_scale,
            zorder=4,
        )
        if visible:
            ax.plot((x_value, x_value), (y_value - 0.11, y_value - 0.04), color=BLUE, lw=0.8 * lw_scale, alpha=0.55, zorder=3)
    ax.annotate(
        "",
        (token_x[-1] + 0.14, token_y[-1]),
        (token_x[-1] + 0.03, token_y[-1]),
        arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 0.8 * lw_scale, "mutation_scale": arrow_scale},
    )

    # 4: MeshGNN cue: node graph + global material heads.
    cx, cy = _icon_centre(3)
    layer_widths = (4, 5, 4, 3)
    layer_x = cx + np.asarray((-0.46, -0.21, 0.04, 0.29))
    layer_y = [cy + np.linspace(-0.16, 0.16, count) for count in layer_widths]
    for first_layer, second_layer in zip(range(3), range(1, 4), strict=True):
        for first_y in layer_y[first_layer]:
            for second_y in layer_y[second_layer]:
                ax.plot(
                    (layer_x[first_layer], layer_x[second_layer]),
                    (first_y, second_y),
                    color=GREEN,
                    lw=0.35 * lw_scale,
                    alpha=0.35,
                    zorder=2,
                )
    for x_value, ys in zip(layer_x, layer_y, strict=True):
        ax.scatter(np.full(len(ys), x_value), ys, s=9 * marker_scale, color=GREEN, edgecolor="#0B5F3A", linewidth=0.3 * lw_scale, zorder=4)
    head_x = cx + 0.52
    for y_value in cy + np.asarray((-0.10, 0.0, 0.10)):
        ax.plot(
            (layer_x[-1], head_x),
            (layer_y[-1][len(layer_y[-1]) // 2], y_value),
            color=GREEN,
            lw=0.45 * lw_scale,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(head_x, y_value, s=11 * marker_scale, marker="D", color=ORANGE, zorder=5)

    # 5: support screen with retain-prior / MAP-FEM branches and a FEM lattice.
    cx, cy = _icon_centre(4)
    decision_x, decision_y = cx, cy + 0.09
    ax.scatter(
        decision_x,
        decision_y,
        s=decision_area,
        marker="D",
        facecolor="#FFF7ED",
        edgecolor=ORANGE,
        linewidth=0.9 * lw_scale,
        zorder=3,
    )
    ax.text(decision_x, decision_y, r"$\Delta J$?", fontsize=label_pt, ha="center", va="center", color=ORANGE, zorder=4)
    for sign, word in ((-1, "prior"), (1, "MAP")):
        ax.annotate(
            "",
            (cx + sign * 0.40, cy - 0.06),
            (decision_x + sign * decision_arm[0], decision_y - decision_arm[1]),
            arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 0.6 * lw_scale, "mutation_scale": arrow_scale},
        )
        ax.text(cx + sign * 0.40, cy - 0.15, word, fontsize=label_pt, ha="center", va="center", color=ORANGE)
    for row in range(2 if fem_lattice else 0):
        for col in range(3):
            ax.scatter(
                cx + 0.58 + 0.075 * col,
                cy - 0.08 - 0.065 * row,
                s=8 * marker_scale,
                marker="s",
                facecolor=PALE_ORANGE,
                edgecolor=ORANGE,
                linewidth=0.45 * lw_scale,
                zorder=3,
            )

    # 6: audited material-field map with an accept mark.
    cx, cy = _icon_centre(5)
    field = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    for row in range(field.shape[0]):
        for col in range(field.shape[1]):
            ax.scatter(
                cx - 0.42 + 0.13 * col,
                cy + 0.13 - 0.09 * row,
                s=24 * marker_scale,
                marker="s",
                color=ORANGE if field[row, col] else BLUE,
                alpha=0.85 if field[row, col] else 0.55,
                edgecolor="white",
                linewidth=0.3 * lw_scale,
                zorder=3,
            )
    ax.text(cx + 0.50, cy, "✓", color=PURPLE, fontsize=check_pt, va="center", ha="center", fontweight="bold", zorder=4)
    return boxes


def fig_pipeline(output_dir: Path) -> None:
    """Draw the Calibrated Multiview screened-FEM inference path: observe → encode → prior → screen → audit.

    The canvas is laid out at its printed size (``FINAL_WIDTH_IN`` wide): one
    axes fills the figure and its data units are inches, so every coordinate
    below is an axes fraction scaled by the page width/height and every font
    size is the printed point size (titles 8.5 pt, all other text >= 7 pt).
    """
    W, H = FINAL_WIDTH_IN, 3.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    # save_figure pads the page by 0.03 in per side, so nominal 7.1 pt prints
    # at >= 7.0 pt once the PDF is scaled to \linewidth.
    title_pt, detail_pt, label_pt = 8.5, 7.1, 7.1
    # Estimator inference order (clockwise): geometry → motion → encoding →
    # hybrid prior → screened FEM decision → audited material field.
    box_w, box_h = PIPELINE_BOX_W, PIPELINE_BOX_H
    margin = 0.02
    gap = (W - 2 * margin - 3 * box_w) / 2
    col_x = [margin + index * (box_w + gap) for index in range(3)]
    row_y = (2.42, 1.05)  # bottom edge of the top and bottom rows (in)
    positions = (
        (col_x[0], row_y[0]),
        (col_x[1], row_y[0]),
        (col_x[2], row_y[0]),
        (col_x[2], row_y[1]),
        (col_x[1], row_y[1]),
        (col_x[0], row_y[1]),
    )
    draw_pipeline_stages(ax, positions, title_pt=title_pt, detail_pt=detail_pt, label_pt=label_pt)

    # Arrows between stages: symbol above the shaft, verb below it.
    mid_top = row_y[0] + box_h / 2
    mid_bottom = row_y[1] + box_h / 2
    label_gap = 0.16
    arrow_specs = (
        # start, end, formula, gloss
        ((col_x[0] + box_w + 0.02, mid_top), (col_x[1] - 0.02, mid_top), r"$\Pi_v,\mathbf{f}$", "project\n+ load"),
        ((col_x[1] + box_w + 0.02, mid_top), (col_x[2] - 0.02, mid_top), r"$\mathbf{z}_{i\ell vt}$", "tokenize"),
        ((col_x[2] - 0.02, mid_bottom), (col_x[1] + box_w + 0.02, mid_bottom), r"$\Delta J$", "screen"),
        ((col_x[1] - 0.02, mid_bottom), (col_x[0] + box_w + 0.02, mid_bottom), "accept /\nretain", "audit"),
    )
    for start, end, formula, gloss in arrow_specs:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color=INK, zorder=3, shrinkA=0, shrinkB=0)
        )
        mid_x = 0.5 * (start[0] + end[0])
        formula_lines = formula.count("\n") + 1
        ax.text(
            mid_x,
            start[1] + label_gap + (0.05 if formula_lines > 1 else 0.0),
            formula,
            ha="center",
            va="center",
            fontsize=label_pt,
            color=INK,
            fontweight="bold",
            linespacing=1.1,
            zorder=6,
        )
        gloss_lines = gloss.count("\n") + 1
        ax.text(
            mid_x,
            start[1] - label_gap - (0.05 if gloss_lines > 1 else 0.0),
            gloss,
            ha="center",
            va="center",
            fontsize=label_pt,
            color=GRAY,
            linespacing=1.1,
            zorder=6,
        )
    # Vertical arrow from the encoder to the hybrid initializer, labelled in
    # the inter-row gutter to the right of the shaft.
    shaft_x = col_x[2] + box_w / 2
    ax.add_patch(
        FancyArrowPatch(
            (shaft_x, row_y[0] - 0.02),
            (shaft_x, row_y[1] + box_h + 0.02),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.1,
            color=INK,
            zorder=3,
            shrinkA=0,
            shrinkB=0,
        )
    )
    gutter_mid = 0.5 * (row_y[0] + row_y[1] + box_h)
    ax.text(shaft_x + 0.07, gutter_mid + 0.075, r"$\theta_0$", ha="left", va="center", fontsize=label_pt, fontweight="bold", color=INK)
    ax.text(shaft_x + 0.07, gutter_mid - 0.075, "MeshGNN prior", ha="left", va="center", fontsize=label_pt, color=GRAY)

    ax.text(margin + 0.04, H - 0.07, "Screened multiview finite-element identification", fontsize=10.0, fontweight="bold", color=INK, va="top")
    ax.text(
        margin + 0.04,
        H - 0.31,
        "Geometry and motion  →  hybrid prior  →  screened FEM  →  audited material field",
        fontsize=7.5,
        color=GRAY,
        va="center",
    )
    ax.text(
        W / 2,
        0.86,
        "Learned-prior initializer + support-screened MAP FEM + objective-based retention",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=PURPLE,
    )
    band_w, band_h, band_y = 3.9, 0.54, 0.12
    ax.add_patch(
        FancyBboxPatch(
            (W / 2 - band_w / 2, band_y),
            band_w,
            band_h,
            boxstyle="round,pad=0,rounding_size=0.07",
            facecolor="#F7FAFC",
            edgecolor=BLUE,
            linewidth=1.1,
            linestyle="--",
        )
    )
    ax.text(W / 2, band_y + band_h - 0.15, "Selection protocol", ha="center", va="center", fontsize=8.0, fontweight="bold", color=BLUE)
    ax.text(
        W / 2,
        band_y + 0.16,
        "geometry split • train fit • validation selection • locked test",
        ha="center",
        va="center",
        fontsize=7.3,
        color=INK,
    )
    save_figure(fig, output_dir, "fig_pipeline")


def _patient_record(manifest: dict[str, Any], patient_id: str) -> dict[str, Any]:
    for patient in manifest["patients"]:
        if patient["patient_id"] == patient_id:
            return patient
    raise ValueError(f"Patient absent from manifest: {patient_id}")


def fig_dataset_protocol(manifest: dict[str, Any], output_dir: Path) -> None:
    """Summarize the generated cohort and acquisition protocol from its manifest.

    Laid out at the printed width (6.5 in): A case-level split bar, B four
    material/region histograms, C the four-load / three-view / seven-frame
    protocol and the observation-model card. Labels are >= 7 pt on the page.
    """
    patients = manifest["patients"]
    split_order = ("train", "val", "test")
    split_labels = ("Train", "Validation", "Test")
    counts = np.asarray([sum(p["split"] == key for p in patients) for key in split_order])
    if tuple(counts) != (150, 50, 50):
        raise ValueError(f"Unexpected case split: {dict(zip(split_order, counts))}")
    E = np.asarray([p["E_background"] for p in patients]) / 1000.0
    ratio = np.asarray([p["inclusion_ratio"] for p in patients])
    radius = np.asarray([p["inclusion_radius_fraction"] for p in patients])
    center = np.asarray([p["inclusion_center_fraction"] for p in patients])

    small_pt = 7.1
    W, H = FINAL_WIDTH_IN, 6.1
    fig = plt.figure(figsize=(W, H))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.42, 1.0, 1.0),
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.905,
        hspace=0.80,
    )
    ax_split = fig.add_subplot(grid[0])
    split_colors = [BLUE, GREEN, PURPLE]
    left = 0
    for count, label, color in zip(counts, split_labels, split_colors, strict=True):
        ax_split.barh(0, count, left=left, height=0.6, color=color, edgecolor="white", linewidth=1.5)
        ax_split.text(
            left + count / 2,
            0,
            f"{label}  $n={count}$",
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=small_pt + 0.4,
        )
        left += count
    ax_split.set_xlim(0, counts.sum())
    ax_split.set_ylim(-0.5, 0.5)
    ax_split.set_yticks([])
    ax_split.tick_params(axis="x", labelsize=small_pt)
    ax_split.set_xlabel("Case-level split (250 procedural synthetic CT-like surrogates)", fontsize=small_pt + 0.4)
    ax_split.set_title("A  Development-fixed case-level partition", loc="left", fontsize=8.5)
    ax_split.spines[["left", "right", "top"]].set_visible(False)
    ax_split.text(
        1.0,
        1.10,
        "selection: train • calibration: validation • test: locked",
        transform=ax_split.transAxes,
        ha="right",
        va="bottom",
        fontsize=small_pt,
        color=GRAY,
    )

    dist_grid = grid[1].subgridspec(1, 4, wspace=0.50)
    distribution_axes = [fig.add_subplot(dist_grid[index]) for index in range(4)]
    distribution_axes[0].set_title("B  Generated material/region distributions", loc="left", fontsize=8.5, x=0.0, pad=6)
    histogram_specs = [
        (E, "$E_b$ (kPa)", BLUE),
        (ratio, r"Inclusion ratio $\rho_E$", GREEN),
        (radius, "Radius fraction $R$", ORANGE),
    ]
    for ax, (values, label, color) in zip(distribution_axes[:3], histogram_specs, strict=True):
        ax.hist(values, bins=14, color=color, alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.axvline(np.median(values), color=INK, linestyle="--", linewidth=1.0)
        ax.set_xlabel(label, fontsize=small_pt + 0.4)
        ax.set_ylabel("Cases", fontsize=small_pt + 0.4)
        ax.tick_params(labelsize=small_pt)
        ax.grid(axis="y", color=LIGHT)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
        ax.text(0.97, 0.93, f"median {np.median(values):.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=small_pt)
    ax = distribution_axes[3]
    for dim, color, label in zip(range(3), (BLUE, ORANGE, GREEN), ("x", "y", "z"), strict=True):
        ax.hist(center[:, dim], bins=12, histtype="step", linewidth=1.4, color=color, label=label)
    ax.set_xlabel("Inclusion-center fraction", fontsize=small_pt + 0.4)
    ax.set_ylabel("Cases", fontsize=small_pt + 0.4)
    ax.tick_params(labelsize=small_pt)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(title="Axis", ncol=3, fontsize=small_pt, title_fontsize=small_pt, loc="upper center", columnspacing=0.8, handlelength=1.2)

    proto_grid = grid[2].subgridspec(1, 4, wspace=0.42, width_ratios=(1.45, 0.9, 0.95, 1.5))
    protocol_axes = [fig.add_subplot(proto_grid[index]) for index in range(4)]
    protocol_axes[0].set_title("C  Loads, views, frames, and observation model", loc="left", fontsize=8.5, x=0.0, pad=16)
    load_rows = _patient_record(manifest, "lung_patient_004")["experiments"]
    load_names = [row["name"].replace("_", "\n") for row in load_rows]
    load_values = [row["max_force_N"] for row in load_rows]
    ax = protocol_axes[0]
    bars = ax.bar(range(4), load_values, color=[BLUE, BLUE, ORANGE, ORANGE], edgecolor=INK, linewidth=0.6)
    ax.set_xticks(range(4), load_names, fontsize=small_pt)
    ax.tick_params(axis="y", labelsize=small_pt)
    ax.set_ylabel("Peak force (N)", fontsize=small_pt + 0.4)
    ax.text(0.5, 1.02, "4-load protocol", transform=ax.transAxes, ha="center", va="bottom", fontsize=small_pt + 0.4, fontweight="semibold")
    for bar, value in zip(bars, load_values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:g}", ha="center", fontsize=small_pt)
    ax.set_ylim(0, max(load_values) * 1.22)
    ax.set_xlim(-0.5, 3.5)
    ax.grid(axis="y", color=LIGHT)

    ax = protocol_axes[1]
    for view in range(3):
        angle = 2 * np.pi * view / 3 + np.pi / 2
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color=(BLUE, GREEN, PURPLE)[view], lw=2.2)
        ax.scatter(np.cos(angle), np.sin(angle), s=50, color=(BLUE, GREEN, PURPLE)[view], edgecolor=INK, linewidth=0.6, zorder=3)
        label_xy = (0.0, 1.32) if view == 0 else (0.62 * np.sign(np.cos(angle)), -1.0)
        ax.text(label_xy[0], label_xy[1], f"view {view + 1}", ha="center", va="center", fontsize=small_pt)
    ax.scatter(0, 0, s=90, color=LIGHT, edgecolor=INK, marker="h", linewidth=0.6, zorder=3)
    # Isotropic axes without shrinking the box: match the x range to the
    # box aspect so the three views stay 120 degrees apart on the page.
    box = ax.get_position()
    y_range = (-1.45, 1.6)
    x_half = 0.5 * (y_range[1] - y_range[0]) * (box.width * W) / (box.height * H)
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(*y_range)
    ax.text(0.5, 1.02, "3 synchronized views", transform=ax.transAxes, ha="center", va="bottom", fontsize=small_pt + 0.4, fontweight="semibold")
    ax.axis("off")

    ax = protocol_axes[2]
    frames = np.arange(7)
    ramp = np.sin(np.linspace(0, np.pi / 2, 7)) ** 1.6
    ax.plot(frames, ramp, "o-", color=BLUE, lw=1.6, markersize=3.5)
    ax.fill_between(frames, 0, ramp, color=BLUE, alpha=0.12)
    ax.set_xticks(frames)
    ax.tick_params(labelsize=small_pt)
    ax.set_xlabel("Frame", fontsize=small_pt + 0.4)
    ax.set_ylabel("Normalized load", fontsize=small_pt + 0.4)
    ax.text(0.5, 1.02, "7-frame load ramp", transform=ax.transAxes, ha="center", va="bottom", fontsize=small_pt + 0.4, fontweight="semibold")
    ax.grid(color=LIGHT)

    ax = protocol_axes[3]
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02",
            facecolor="#F7F8FA",
            edgecolor=ORANGE,
            linewidth=1.2,
            clip_on=False,
        )
    )
    noise_px = manifest["generation_config"]["motion_noise_std"] * 200.0
    force_prior = manifest["generation_config"]["force_prior_fraction"]
    ax.text(0.5, 1.02, "Observation model", transform=ax.transAxes, ha="center", va="bottom", fontsize=small_pt + 0.4, fontweight="semibold", color=ORANGE)
    # Six compact entries in two columns, formula on the last row.
    cards = (
        (0.04, 0.80, "Track noise", f"{noise_px:.2f} px"),
        (0.57, 0.80, "Force error", f"{force_prior:.0%}"),
        (0.04, 0.56, "Depth change", "normalized"),
        (0.57, 0.56, "Visibility", "binary mask"),
        (0.04, 0.32, "Occlusion", "confidence"),
        (0.57, 0.32, "Camera pose", "known"),
    )
    for x0, y0, title, value in cards:
        ax.text(x0, y0 + 0.095, title, transform=ax.transAxes, fontsize=small_pt, color=GRAY, va="center")
        ax.text(x0, y0, value, transform=ax.transAxes, fontsize=small_pt + 0.2, color=INK, fontweight="semibold", va="center")
    ax.text(
        0.5,
        0.09,
        r"$y_{t,v}=\Pi_v(x+u_t)+\epsilon$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=small_pt + 0.4,
        color=INK,
    )
    ax.text(
        0.5,
        -0.08,
        r"$\epsilon\sim\mathcal{N}(0,0.05^2)$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=small_pt + 0.4,
        color=INK,
    )
    fig.suptitle("Dataset and observation protocol", y=0.985, fontsize=9.5)
    save_figure(fig, output_dir, "fig_dataset_protocol")


def _load_case(dataset_root: Path, manifest: dict[str, Any], patient_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the qualitative case figure") from exc
    patient = _patient_record(manifest, patient_id)
    experiments = []
    for row in patient["experiments"]:
        path = dataset_root / Path(row["relative_path"].replace("\\", "/"))
        if not path.is_file():
            raise FileNotFoundError(f"Case tensor missing: {path}")
        experiments.append(torch.load(path, map_location="cpu", weights_only=False))
    return patient, experiments


def _clean_3d_axis(ax: plt.Axes, *, zoom: float = 1.0) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ``zoom`` > 1 trims the generous default 3-D margins so that small
    # (~2 in) panels at the final 6.5 in width still show a legible surface.
    ax.set_box_aspect((1, 0.85, 0.65), zoom=zoom)
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.grid(False)
    if zoom > 1.0:
        # Zoomed panels sit close together; the residual axis lines would
        # otherwise run into the neighbouring panel titles.
        ax.set_axis_off()


def _surface_partition(
    ax: plt.Axes,
    nodes: np.ndarray,
    triangles: np.ndarray,
    center: np.ndarray,
    radius: float,
    title: str,
    *,
    fontsize: float = 8.5,
    zoom: float = 1.0,
) -> None:
    centroids = nodes[triangles].mean(axis=1)
    inside = np.linalg.norm(centroids - center[None, :], axis=1) <= radius
    facecolors = np.where(inside[:, None], mcolors.to_rgba(ORANGE), mcolors.to_rgba(BLUE))
    surface = ax.plot_trisurf(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        triangles=triangles,
        linewidth=0.28,
        edgecolor=(1, 1, 1, 0.36),
        antialiased=True,
        shade=False,
    )
    surface.set_facecolors(facecolors)
    ax.set_title(title, pad=2, fontsize=fontsize)
    ax.view_init(elev=24, azim=-58)
    surface_nodes = nodes[np.unique(triangles)]
    lo = surface_nodes.min(axis=0)
    hi = surface_nodes.max(axis=0)
    pad = np.maximum((hi - lo) * 0.05, 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    _clean_3d_axis(ax, zoom=zoom)


def fig_qualitative_case(
    manifest: dict[str, Any],
    calibration: dict[str, Any],
    fem_payload: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
) -> None:
    """Render a case-level synthetic example from the four saved experiments."""
    patient_id = "lung_patient_004"
    patient, experiments = _load_case(dataset_root, manifest, patient_id)
    record = next(row for row in calibration["records"] if row["patient_id"] == patient_id)
    reference_method = fem_payload["reference_method"]
    candidate_method = next(
        name for name in fem_payload["methods"] if name != reference_method
    )
    fem_record = next(
        row
        for row in fem_payload["records"]
        if row["patient_id"] == patient_id
        and row["benchmark_method"] == candidate_method
    )
    sample = experiments[0]
    nodes = np.asarray(sample["nodes"])
    triangles = np.asarray(sample["surface_tris"], dtype=int)
    true_center = np.asarray(sample["inclusion_center"])
    true_radius = float(sample["inclusion_radius"])
    extent = nodes.max(axis=0) - nodes.min(axis=0)
    pred_center = nodes.min(axis=0) + np.asarray(record["center_fraction_estimated"]) * extent
    pred_radius = float(record["radius_fraction_estimated"]) * min(extent[0], extent[1])

    # Final 6.5 in layout. Vertical bands (inches from the top of a 6.9 in
    # page): A = three 3-D panels, a one-line screening note, B = header row
    # + 4 loads x (3 views x peak flow | 7-frame path), a legend strip, and
    # C = the load/response curves. Everything is placed in absolute inches
    # so that the point sizes below are the printed point sizes.
    n_views = 3
    n_track_cols = 2 * n_views
    page_height = 6.9
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, page_height))

    def band(top_in: float, bottom_in: float, **kwargs: Any):
        return fig.add_gridspec(
            top=1.0 - top_in / page_height,
            bottom=1.0 - bottom_in / page_height,
            **kwargs,
        )

    # 3-D axes draw their titles above the gridspec cell, so the A band starts
    # 0.56 in below the top edge to clear the suptitle.
    grid_a = band(0.56, 1.60, nrows=1, ncols=3, left=0.035, right=0.985, wspace=0.04)
    grid_note = band(1.62, 1.96, nrows=1, ncols=1, left=0.07, right=0.985)
    grid_b = band(
        2.04,
        5.12,
        nrows=5,
        ncols=n_track_cols,
        height_ratios=(0.34, 1, 1, 1, 1),
        left=0.07,
        right=0.985,
        wspace=0.10,
        hspace=0.14,
    )
    grid_legend = band(5.14, 5.34, nrows=1, ncols=1, left=0.07, right=0.985)
    grid_c = band(5.64, 6.48, nrows=1, ncols=1, left=0.085, right=0.905)

    ax_true = fig.add_subplot(grid_a[0, 0], projection="3d")
    ax_pred = fig.add_subplot(grid_a[0, 1], projection="3d")
    ax_mesh = fig.add_subplot(grid_a[0, 2], projection="3d")
    _surface_partition(
        ax_true,
        nodes,
        triangles,
        true_center,
        true_radius,
        f"Ground-truth partition\n$E_b$ = {record['E_background_true']/1000:.2f} kPa, $\\rho_E$ = {record['inclusion_ratio_true']:.2f}",
        fontsize=8.0,
        zoom=1.45,
    )
    _surface_partition(
        ax_pred,
        nodes,
        triangles,
        pred_center,
        pred_radius,
        f"Predicted partition\n$E_b$ = {record['E_background_estimated']/1000:.2f} kPa, $\\rho_E$ = {record['inclusion_ratio_estimated']:.2f}",
        fontsize=8.0,
        zoom=1.45,
    )
    u_sequence = np.asarray(sample["u_seq"]).reshape(7, -1, 3)
    peak_frame = int(np.argmax(np.linalg.norm(u_sequence, axis=2).max(axis=1)))
    u_peak = u_sequence[peak_frame]
    surface_ids = np.unique(triangles)
    deformation_display_scale = 4.0
    deformed = nodes + deformation_display_scale * u_peak
    for tri in triangles[::2]:
        loop = np.r_[tri, tri[0]]
        ax_mesh.plot(nodes[loop, 0], nodes[loop, 1], nodes[loop, 2], color=GRAY, alpha=0.32, lw=0.45)
        ax_mesh.plot(deformed[loop, 0], deformed[loop, 1], deformed[loop, 2], color=ORANGE, alpha=0.68, lw=0.55)
    stride_ids = surface_ids[:: max(1, len(surface_ids) // 20)]
    ax_mesh.quiver(
        nodes[stride_ids, 0],
        nodes[stride_ids, 1],
        nodes[stride_ids, 2],
        deformation_display_scale * u_peak[stride_ids, 0],
        deformation_display_scale * u_peak[stride_ids, 1],
        deformation_display_scale * u_peak[stride_ids, 2],
        color=ORANGE,
        linewidth=0.8,
        arrow_length_ratio=0.25,
    )
    ax_mesh.set_title(
        "Rest → peak-deformed surface\n(gray → orange; displacement ×4)",
        fontsize=8.0,
        pad=2,
    )
    ax_mesh.view_init(elev=24, azim=-58)
    _clean_3d_axis(ax_mesh, zoom=1.45)
    retained = fem_record["screening_rejected"] and not fem_record["refinement_accepted"]
    ax_note = fig.add_subplot(grid_note[0, 0])
    ax_note.axis("off")
    note_text = (
        "Frozen FEM screening decision — "
        f"screening rejected: {str(fem_record['screening_rejected']).lower()};  "
        f"relative cost reduction: {fem_record['relative_cost_reduction']:.2%}\n"
        f"{fem_record['function_evaluations']} evaluations;  "
        f"{'retained PCA–Ridge initial estimate' if retained else 'accepted physics refinement'}"
    )
    ax_note.text(
        0.0,
        0.5,
        note_text,
        transform=ax_note.transAxes,
        fontsize=7.4,
        color=ORANGE,
        ha="left",
        va="center",
        linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": PALE_ORANGE, "pad": 2.5},
    )
    fig.text(0.012, 1.0 - 0.30 / page_height, "A", fontsize=10, fontweight="bold", va="top")

    load_names = ("Press left", "Press right", "Shear x", "Shear y")
    track_axes = []
    frame_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 7))
    # Column headers for section B live in their own short row so that the
    # two-line titles cannot collide with the screening note above.
    for view in range(n_views):
        for kind_index, kind in enumerate(("peak flow", "7-frame path")):
            ax_header = fig.add_subplot(grid_b[0, 2 * view + kind_index])
            ax_header.axis("off")
            ax_header.text(
                0.5,
                0.0,
                f"View {view + 1}\n{kind}",
                transform=ax_header.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="semibold",
                linespacing=1.15,
            )
    for load_index, experiment in enumerate(experiments):
        u_load = np.asarray(experiment["u_seq"]).reshape(7, -1, 3)
        peak = int(np.argmax(np.linalg.norm(u_load, axis=2).max(axis=1)))
        rest_seq = np.asarray(experiment["image_uv_rest_multiview_seq"])
        deformed_seq = np.asarray(experiment["image_uv_deformed_multiview_seq"])
        confidence_seq = np.asarray(
            experiment["image_occlusion_confidence_multiview_seq"]
        )
        visible_seq = np.asarray(
            experiment["image_visibility_multiview_seq"]
        ).astype(bool)
        n_views = int(rest_seq.shape[1])
        for view in range(n_views):
            # Column pair per view: peak flow | seven-frame path.
            ax_flow = fig.add_subplot(grid_b[load_index + 1, 2 * view])
            ax_path = fig.add_subplot(grid_b[load_index + 1, 2 * view + 1])
            track_axes.extend((ax_flow, ax_path))

            rest = rest_seq[peak, view]
            deformed_uv = deformed_seq[peak, view]
            confidence = np.clip(confidence_seq[peak, view], 0, 1)
            valid = (
                visible_seq[peak, view]
                & np.isfinite(rest).all(axis=1)
                & np.isfinite(deformed_uv).all(axis=1)
            )
            ids = np.where(valid)[0]
            if len(ids) > 22:
                ids = ids[np.argsort(confidence[ids])[-22:]]
            track_display_scale = 35.0
            delta = (deformed_uv[ids] - rest[ids]) * track_display_scale
            display_deformed = rest[ids] + delta
            segments = np.stack([rest[ids], display_deformed], axis=1)
            collection = LineCollection(
                segments,
                cmap="viridis",
                norm=plt.Normalize(0, 1),
                linewidths=1.0,
            )
            collection.set_array(confidence[ids])
            ax_flow.add_collection(collection)
            ax_flow.quiver(
                rest[ids, 0],
                rest[ids, 1],
                delta[:, 0],
                delta[:, 1],
                confidence[ids],
                cmap="viridis",
                clim=(0, 1),
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.009,
                headwidth=3.4,
                headlength=4.0,
            )
            ax_flow.scatter(rest[ids, 0], rest[ids, 1], s=5, color=INK, alpha=0.55)
            all_uv = np.vstack([rest[ids], display_deformed])
            margin = max(np.ptp(all_uv[:, 0]), np.ptp(all_uv[:, 1]), 1e-6) * 0.08
            ax_flow.set_xlim(all_uv[:, 0].min() - margin, all_uv[:, 0].max() + margin)
            ax_flow.set_ylim(all_uv[:, 1].max() + margin, all_uv[:, 1].min() - margin)
            ax_flow.set_aspect("equal")

            # Seven-frame image-plane path for denser view evidence.
            for frame_index in range(7):
                mask = visible_seq[frame_index, view]
                if not np.any(mask):
                    continue
                ax_path.scatter(
                    deformed_seq[frame_index, view, mask, 0],
                    deformed_seq[frame_index, view, mask, 1],
                    s=6,
                    color=frame_colors[frame_index],
                    alpha=float(
                        np.clip(
                            np.mean(confidence_seq[frame_index, view, mask]),
                            0.2,
                            0.95,
                        )
                    ),
                    edgecolors="none",
                )
            for node_index in range(0, deformed_seq.shape[2], 3):
                mask = visible_seq[:, view, node_index]
                if np.count_nonzero(mask) >= 2:
                    ax_path.plot(
                        deformed_seq[mask, view, node_index, 0],
                        deformed_seq[mask, view, node_index, 1],
                        color=GRAY,
                        linewidth=0.45,
                        alpha=0.4,
                    )
            ax_path.invert_yaxis()
            ax_path.set_aspect("equal")

            if view == 0:
                ax_flow.set_ylabel(load_names[load_index], fontsize=7.5, labelpad=3)
            for ax in (ax_flow, ax_path):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(color=LIGHT)
    # One legend strip for both encodings of section B: path dots are
    # coloured by frame, peak-flow arrows by occlusion confidence.
    frame_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=frame_colors[index],
            markersize=5,
            label=f"Frame {index}",
        )
        for index in (0, 3, 6)
    ]
    confidence_handles = [
        Line2D([0], [0], color=plt.cm.viridis(value), lw=2.2, label=label)
        for value, label in (
            (0.2, "Arrow colour: low confidence"),
            (0.6, "medium"),
            (0.95, "high"),
        )
    ]
    ax_legend = fig.add_subplot(grid_legend[0, 0])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=frame_handles + confidence_handles,
        loc="center",
        ncol=6,
        fontsize=7.4,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.text(0.012, 1.0 - 1.98 / page_height, "B", fontsize=10, fontweight="bold", va="top")

    ax_response = fig.add_subplot(grid_c[0, 0])
    frames = np.arange(7)
    force_curves = []
    displacement_curves = []
    for experiment in experiments:
        force = np.asarray(experiment["true_forces"]).reshape(7, -1, 3)
        disp = np.asarray(experiment["u_seq"]).reshape(7, -1, 3)
        force_curves.append(np.linalg.norm(force, axis=2).sum(axis=1))
        displacement_curves.append(np.linalg.norm(disp, axis=2).max(axis=1) * 1000)
    force_curves = np.asarray(force_curves)
    displacement_curves = np.asarray(displacement_curves)
    line_colors = [BLUE, GREEN, ORANGE, PURPLE]
    for name, color, curve in zip(load_names, line_colors, force_curves, strict=True):
        ax_response.plot(frames, curve, "o-", color=color, lw=1.5, ms=3.5, label=f"{name} force")
    ax_response.set_xlabel("Frame", labelpad=1)
    ax_response.set_ylabel("Applied nodal\nforce sum (N)", fontsize=7.5)
    ax_response.set_xticks(frames)
    ax_response.grid(color=LIGHT)
    ax_disp = ax_response.twinx()
    for color, curve in zip(line_colors, displacement_curves, strict=True):
        ax_disp.plot(frames, curve, "--", color=color, lw=1.3, alpha=0.86)
    ax_disp.set_ylabel("Peak nodal\ndisplacement (mm)", fontsize=7.5)
    ax_disp.tick_params(labelsize=7.5)
    # Head-room so the four-entry legend never sits on the frame-3 peak.
    ax_response.set_ylim(0, float(force_curves.max()) * 1.55)
    ax_disp.set_ylim(0, float(displacement_curves.max()) * 1.55)
    ax_response.set_title(
        "Seven-frame load/response sequences (solid: force; dashed: displacement)",
        loc="left",
        fontsize=8.5,
        pad=4,
    )
    ax_response.legend(
        ncol=4,
        loc="upper left",
        fontsize=7.4,
        handlelength=1.6,
        columnspacing=1.2,
        borderaxespad=0.3,
    )
    fig.text(0.012, 1.0 - 5.45 / page_height, "C", fontsize=10, fontweight="bold", va="top")
    case_label = patient_id.replace("lung_patient_", "case ")
    fig.suptitle(
        f"Case-level qualitative result — synthetic {case_label}",
        y=0.993,
        fontsize=10.5,
    )
    save_figure(fig, output_dir, "fig_qualitative_case")


def fig_case_gallery(
    manifest: dict[str, Any],
    calibration: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
) -> None:
    """Show three predeclared frozen test cases spanning background-E error."""
    case_specs = (
        ("lung_patient_039", "LOW ERROR", GREEN),
        ("lung_patient_004", "MID ERROR", BLUE),
        ("lung_patient_134", "HIGH ERROR", ORANGE),
    )
    records = {row["patient_id"]: row for row in calibration["records"]}
    # Final 6.5 in layout: per case one row of GT surface | prediction | card.
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 5.9))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 1.0, 0.98),
        left=0.01,
        right=0.99,
        top=0.905,
        bottom=0.01,
        hspace=0.16,
        wspace=0.02,
    )

    for row_index, (patient_id, stratum, accent) in enumerate(case_specs):
        patient, experiments = _load_case(dataset_root, manifest, patient_id)
        if patient["split"] != "test":
            raise ValueError(f"Gallery case is not in the frozen test split: {patient_id}")
        if patient_id not in records:
            raise ValueError(f"Frozen PCA–Ridge prediction missing: {patient_id}")
        record = records[patient_id]
        sample = experiments[0]
        nodes = np.asarray(sample["nodes"])
        triangles = np.asarray(sample["surface_tris"], dtype=int)
        true_center = np.asarray(sample["inclusion_center"])
        true_radius = float(sample["inclusion_radius"])
        extent = nodes.max(axis=0) - nodes.min(axis=0)
        pred_center = nodes.min(axis=0) + np.asarray(record["center_fraction_estimated"]) * extent
        pred_radius = float(record["radius_fraction_estimated"]) * min(extent[0], extent[1])

        e_error = abs(record["E_background_estimated"] / record["E_background_true"] - 1.0)
        ratio_error = abs(record["inclusion_ratio_estimated"] / record["inclusion_ratio_true"] - 1.0)
        center_error = float(
            np.linalg.norm(
                np.asarray(record["center_fraction_estimated"])
                - np.asarray(record["center_fraction_true"])
            )
        )
        radius_error = abs(record["radius_fraction_estimated"] / record["radius_fraction_true"] - 1.0)

        ax_gt = fig.add_subplot(grid[row_index, 0], projection="3d")
        ax_pred = fig.add_subplot(grid[row_index, 1], projection="3d")
        case_label = patient_id.replace("lung_patient_", "case ")
        _surface_partition(
            ax_gt,
            nodes,
            triangles,
            true_center,
            true_radius,
            f"{stratum} · {case_label}\nGT partition",
            fontsize=8.5,
            zoom=1.35,
        )
        _surface_partition(
            ax_pred,
            nodes,
            triangles,
            pred_center,
            pred_radius,
            "Frozen PCA–Ridge\nprediction",
            fontsize=8.5,
            zoom=1.35,
        )
        ax_gt.title.set_color(accent)

        ax_card = fig.add_subplot(grid[row_index, 2])
        ax_card.axis("off")
        card = FancyBboxPatch(
            (0.03, 0.06),
            0.94,
            0.86,
            transform=ax_card.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#F7F9FB",
            edgecolor=accent,
            linewidth=1.4,
        )
        ax_card.add_patch(card)
        ax_card.text(
            0.09,
            0.86,
            "GT → prediction",
            transform=ax_card.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=accent,
            va="top",
        )
        ax_card.text(
            0.09,
            0.72,
            f"$E_b$: {record['E_background_true']/1000:.2f} → "
            f"{record['E_background_estimated']/1000:.2f} kPa\n"
            f"$\\rho_E$: {record['inclusion_ratio_true']:.2f} → "
            f"{record['inclusion_ratio_estimated']:.2f}",
            transform=ax_card.transAxes,
            va="top",
            fontsize=7.5,
            linespacing=1.5,
        )
        # Two aligned columns (name left, value right) keep every line >= 7 pt
        # inside a 2 in card without relying on padded spaces.
        error_rows = (
            ("$E_b$ relative error", e_error),
            ("ratio relative error", ratio_error),
            ("center normalized error", center_error),
            ("radius relative error", radius_error),
        )
        for line_index, (name, value) in enumerate(error_rows):
            y_text = 0.46 - 0.105 * line_index
            ax_card.text(0.09, y_text, name, transform=ax_card.transAxes, va="top", fontsize=7.5)
            ax_card.text(
                0.92,
                y_text,
                f"{value:.1%}",
                transform=ax_card.transAxes,
                va="top",
                ha="right",
                fontsize=7.5,
                fontweight="semibold",
            )

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=7, markerfacecolor=BLUE, markeredgecolor=INK, label="Background"),
        Line2D([0], [0], marker="s", linestyle="", markersize=7, markerfacecolor=ORANGE, markeredgecolor=INK, label="Inclusion"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.955),
        ncol=2,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.suptitle("Frozen test-case gallery — synthetic examples", y=0.99, fontsize=10.5)
    save_figure(fig, output_dir, "fig_case_gallery")


def _development_metrics(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    stage_specs = [
        ("40-case development", "stage40_round2"),
        ("100-case development", "stage100"),
        ("250-case development", "stage250"),
    ]
    # The stage-250 base reported in the supplement (development-variant table
    # and text: 28.77/31.75/12.78/15.78 %) is the round-0 training state of
    # ``stage250``; rounds 1-2 are later cycle states that were not tabulated.
    # The 40- and 100-case stages keep their round-2 (final-cycle) records.
    gnn_round = {"stage250": "round_0"}
    for stage_label, directory in stage_specs:
        for model, relative in (
            ("MLP", Path(directory) / "mlp" / "metrics_mlp.json"),
            ("MeshGNN", Path(directory) / gnn_round.get(directory, "round_2") / "metrics_gnn.json"),
        ):
            payload = load_json(results_root / relative)
            rows.append({"stage": stage_label, "model": model, **payload["validation"]})
    variants: list[dict[str, Any]] = []
    for directory in (
        "stage250",
        "stage250_physics_summary",
        "stage250_view_summary",
        "stage250_node_summary",
        "stage250_local_refine",
        "stage250_mean_supervised",
    ):
        candidates = [
            results_root / directory / "metrics_gnn.json",
            results_root / directory / gnn_round.get(directory, "round_2") / "metrics_gnn.json",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            continue
        payload = load_json(path)
        variants.append({"variant": directory.removeprefix("stage250_").replace("_", " "), **payload["validation"]})
    return rows, variants


def fig_development_ablation(results_root: Path, output_dir: Path) -> None:
    """Show validation-only development iterations without implying pure scaling."""
    rows, variants = _development_metrics(results_root)
    metrics = [
        ("E_background_median_relative_error", "background $E$"),
        ("inclusion_ratio_median_relative_error", "ratio"),
        ("center_error_normalized_median", "center"),
        ("radius_relative_error_median", "radius"),
    ]
    # Final 6.5 in layout: A | B on top, C across the full width underneath
    # so that the six variant names fit beside their markers at >= 7 pt.
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 5.7))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.92),
        width_ratios=(1.0, 1.18),
        left=0.118,
        right=0.985,
        top=0.885,
        bottom=0.085,
        hspace=0.62,
        wspace=0.50,
    )
    ax = fig.add_subplot(grid[0, 0])
    stages = list(dict.fromkeys(row["stage"] for row in rows))
    x = np.arange(len(stages))
    plotted: list[float] = []
    for offset, model, color, marker in ((-0.06, "MLP", GRAY, "s"), (0.06, "MeshGNN", BLUE, "o")):
        values = [
            np.mean([row[key] for key, _ in metrics[:2]]) * 100
            for stage in stages
            for row in rows
            if row["stage"] == stage and row["model"] == model
        ]
        plotted.extend(values)
        ax.plot(x + offset, values, marker=marker, lw=1.6, ms=4.5, color=color, label=model)
    ax.set_xticks(x, ["40\n(round 2)", "100", "250\n(round 0)"])
    ax.set_xlim(-0.35, 2.35)
    # Extra room below the curves keeps the caveat box clear of the lines.
    value_span = max(max(plotted) - min(plotted), 1e-6)
    ax.set_ylim(min(plotted) - 0.34 * value_span, max(plotted) + 0.10 * value_span)
    ax.set_xlabel("Development cohort size (cases)")
    ax.set_ylabel("Mean validation\nmaterial error (%)")
    ax.set_title("Development iterations", fontsize=9.5)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.6)
    ax.text(
        0.97,
        0.04,
        "Cohort size, training state and\narchitecture change together:\nnot a pure scaling law.",
        transform=ax.transAxes,
        fontsize=7.4,
        color=ORANGE,
        ha="right",
        va="bottom",
        linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 2.5},
    )
    panel_label(ax, "A", x=-0.30, y=1.14)

    ax = fig.add_subplot(grid[0, 1])
    variant_names = [row["variant"] for row in variants]
    y = np.arange(len(variant_names))
    width = 0.19
    metric_colors = [BLUE, GREEN, ORANGE, PURPLE]
    for index, ((key, label), color) in enumerate(zip(metrics, metric_colors, strict=True)):
        values = np.asarray([row[key] for row in variants]) * 100
        ax.barh(y + (index - 1.5) * width, values, height=width, color=color, label=label)
    ax.set_yticks(y, variant_names)
    ax.invert_yaxis()
    ax.set_xlabel("Validation median error (%)")
    ax.set_title("250-cohort variants", fontsize=9.5, pad=16)
    ax.grid(axis="x", color=LIGHT)
    ax.legend(
        ncol=4,
        fontsize=7.4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.1,
    )
    panel_label(ax, "B", x=-0.40, y=1.14)

    ax = fig.add_subplot(grid[1, :])
    sdf = np.asarray([row["node_sdf_mae_mean"] for row in variants])
    dice = np.asarray([row["partition_soft_dice_mean"] for row in variants])
    sizes = np.asarray([row["patient_count"] for row in variants])
    ax.scatter(
        sdf,
        dice,
        s=35 + sizes,
        c=np.arange(len(variants)),
        cmap="viridis",
        edgecolor=INK,
        linewidth=0.6,
        zorder=3,
    )
    # Per-variant label placement chosen so that no two names overlap at the
    # full 5.7 in panel width (checked with the PyMuPDF overlap diagnostic).
    placements = {
        "node summary": ((6, 1), "left", "center"),
        "local refine": ((6, 1), "left", "center"),
        "physics summary": ((0, -9), "center", "top"),
        "view summary": ((6, 0), "left", "center"),
        "stage250": ((6, 0), "left", "center"),
        "mean supervised": ((-7, 0), "right", "center"),
    }
    for index, name in enumerate(variant_names):
        offset, ha, va = placements.get(name, ((6, 1), "left", "center"))
        ax.annotate(
            name,
            (sdf[index], dice[index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            ha=ha,
            va=va,
        )
    span_x = float(np.ptp(sdf))
    span_y = float(np.ptp(dice))
    ax.set_xlim(sdf.min() - 0.08 * span_x, sdf.max() + 0.10 * span_x)
    ax.set_ylim(dice.min() - 0.18 * span_y, dice.max() + 0.22 * span_y)
    ax.set_xlabel("Validation node-SDF MAE (lower is better)")
    ax.set_ylabel("Validation soft Dice\n(higher is better)")
    ax.set_title("Node representation trade-off", fontsize=9.5)
    ax.grid(color=LIGHT)
    ax.text(
        0.985,
        0.95,
        "All choices shown here use validation metrics;\nthe held-out test set was not used for tuning.",
        transform=ax.transAxes,
        fontsize=7.4,
        color=GREEN,
        ha="right",
        va="top",
        linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 2.5},
    )
    panel_label(ax, "C", x=-0.125, y=1.14)
    fig.suptitle("Development and representation ablations (validation only)", y=0.985, fontsize=10.5)
    save_figure(fig, output_dir, "fig_development_ablation")


def _bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    """Deterministic patient bootstrap used only for figure annotations."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(5000, len(values)), replace=True)
    return tuple(np.quantile(np.median(samples, axis=1), (0.025, 0.975)))


def fig_main_results(payload: dict[str, Any], output_dir: Path) -> None:
    """Six-panel held-out accuracy figure at its printed width.

    A--D: case-level violin + box + points per endpoint with the
    development-fixed criterion, median and case-bootstrap 95% CI;
    E: endpoint attainment by case; F: true contrast versus ratio error.
    Every label is >= 7 pt on the 6.5 in page (nominal 7.1 pt after the
    0.03 in save pad).
    """
    gates = payload["gates"]["gates"]
    records = payload["calibration"]["records"]
    if len(records) != 50:
        raise ValueError(f"Expected 50 frozen test records, found {len(records)}")
    heterogeneous = [row for row in records if row["inclusion_ratio_true"] > 1.0 + 1e-8]
    metric_specs = [
        (
            "$E_b$ relative error",
            np.asarray([abs(row["E_background_estimated"] / row["E_background_true"] - 1) for row in records]),
            gates["image_E_error"]["threshold"],
            BLUE,
            "A",
        ),
        (
            r"Inclusion-ratio $\rho_E$ relative error",
            np.asarray([abs(row["inclusion_ratio_estimated"] / row["inclusion_ratio_true"] - 1) for row in records]),
            gates["image_ratio_error"]["threshold"],
            GREEN,
            "B",
        ),
        (
            "Center normalized error",
            np.asarray([
                np.linalg.norm(np.asarray(row["center_fraction_estimated"]) - np.asarray(row["center_fraction_true"]))
                for row in heterogeneous
            ]),
            gates["center_error"]["threshold"],
            PURPLE,
            "C",
        ),
        (
            "Radius relative error",
            np.asarray([abs(row["radius_fraction_estimated"] / row["radius_fraction_true"] - 1) for row in heterogeneous]),
            gates["radius_error"]["threshold"],
            ORANGE,
            "D",
        ),
    ]
    small_pt = 7.1
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 7.0))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.10,
        right=0.985,
        bottom=0.07,
        top=0.92,
        hspace=0.60,
        wspace=0.36,
    )
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(3)])
    for index, (ax, (title, values, threshold, color, label)) in enumerate(
        zip(axes.flat[:4], metric_specs, strict=True)
    ):
        percent = values * 100
        violin = ax.violinplot(percent, positions=[0], widths=0.74, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.18)
        ax.boxplot(
            percent,
            positions=[0],
            widths=0.22,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.1},
            medianprops={"color": INK, "linewidth": 1.5},
            whiskerprops={"color": color, "linewidth": 0.9},
            capprops={"color": color, "linewidth": 0.9},
        )
        rng = np.random.default_rng(3100 + index)
        jitter = rng.normal(0, 0.055, len(percent))
        ax.scatter(jitter, percent, s=9, color=color, alpha=0.66, edgecolor="white", linewidth=0.2, zorder=3)
        ax.axhline(
            threshold * 100,
            color=ORANGE,
            linestyle="--",
            linewidth=1.2,
            label=f"Development-fixed\ncriterion {threshold:.0%}",
        )
        median = float(np.median(values))
        ci = _bootstrap_median_ci(values, 4100 + index)
        ax.text(
            0.975,
            0.965,
            f"median {median:.2%}\nbootstrap 95% CI\n{ci[0]:.1%}–{ci[1]:.1%}\n$n={len(values)}$ cases",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=small_pt,
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 2.5},
        )
        ax.set_xlim(-0.58, 0.58)
        # Leave head-room for the legend and the statistics box.
        ax.set_ylim(min(0.0, percent.min()) - 0.03 * percent.max(), percent.max() * 1.42)
        ax.set_xticks([0], ["track-summary\nPCA–Ridge"], fontsize=small_pt)
        ax.tick_params(axis="y", labelsize=small_pt)
        ax.set_ylabel("Case-level error (%)", fontsize=small_pt + 0.4)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", color=LIGHT)
        ax.legend(loc="upper left", fontsize=small_pt, handlelength=1.6, borderpad=0.3, labelspacing=0.3)
        panel_label(ax, label, x=-0.22, y=1.16)
    # Show endpoint co-occurrence and dependence on contrast without replacing
    # the per-endpoint distributions above.
    ax = axes[2, 0]
    matrix = np.full((4, len(records)), np.nan)
    for row_index, (_, values, threshold, _, _) in enumerate(metric_specs):
        if row_index < 2:
            matrix[row_index] = values <= threshold
        else:
            heterogeneous_ids = [
                index
                for index, row in enumerate(records)
                if row["inclusion_ratio_true"] > 1.0 + 1e-8
            ]
            matrix[row_index, heterogeneous_ids] = values <= threshold
    cmap = mcolors.ListedColormap(("#DDE4EA", "#1B9E77"))
    cmap.set_bad("#F7F7F7")
    ax.pcolormesh(
        np.arange(len(records) + 1),
        np.arange(5),
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolors="white",
        linewidth=0.2,
    )
    ax.set_ylim(4, 0)
    ax.set_yticks(np.arange(4) + 0.5, ("$E_b$", r"$\rho_E$", "Center", "Radius"), fontsize=small_pt)
    ax.set_xticks((0.5, 24.5, 49.5), ("1", "25", "50"), fontsize=small_pt)
    ax.set_xlabel("Held-out case index", fontsize=small_pt + 0.4)
    ax.set_title("Endpoint attainment by case", fontsize=8.5)
    for row_index, (_, values, threshold, _, _) in enumerate(metric_specs):
        ax.text(
            len(records) + 1.0,
            row_index + 0.5,
            f"{np.mean(values <= threshold):.0%}",
            va="center",
            ha="left",
            fontsize=small_pt,
            color=INK,
            clip_on=False,
        )
    ax.text(
        0.0,
        -0.30,
        "green: meets criterion; blank: homogeneous case",
        transform=ax.transAxes,
        fontsize=small_pt,
        color=GREEN,
        va="top",
    )
    panel_label(ax, "E", x=-0.22, y=1.16)

    ax = axes[2, 1]
    true_ratio = np.asarray([row["inclusion_ratio_true"] for row in heterogeneous])
    ratio_error = np.asarray(
        [abs(row["inclusion_ratio_estimated"] / row["inclusion_ratio_true"] - 1) for row in heterogeneous]
    ) * 100
    ax.scatter(true_ratio, ratio_error, s=14, color=GREEN, alpha=0.72, edgecolor="white", linewidth=0.3)
    ax.axhline(
        metric_specs[1][2] * 100,
        color=ORANGE,
        linestyle="--",
        linewidth=1.2,
        label="Development-fixed criterion",
    )
    ax.set_xlabel(r"True inclusion-to-background ratio $\rho_E$", fontsize=small_pt + 0.4)
    ax.set_ylabel(r"$\rho_E$ error (%)", fontsize=small_pt + 0.4)
    ax.set_title("Contrast versus ratio error", fontsize=8.5)
    ax.tick_params(labelsize=small_pt)
    ax.set_ylim(top=ratio_error.max() * 1.25)
    ax.grid(color=LIGHT)
    ax.legend(loc="upper right", fontsize=small_pt, handlelength=1.6, borderpad=0.3)
    panel_label(ax, "F", x=-0.22, y=1.16)
    fig.suptitle(
        "Held-out case-level accuracy, endpoint attainment, and contrast dependence",
        y=0.985,
        fontsize=9.5,
    )
    save_figure(fig, output_dir, "fig_main_results")


def fig_fair_fem(payload: dict[str, Any], output_dir: Path) -> None:
    methods = payload["fair_fem"]["methods"]
    reference_method = payload["fair_fem"]["reference_method"]
    candidate_method = next(name for name in methods if name != reference_method)
    ai = methods[candidate_method]
    fixed = methods[reference_method]
    paired = payload["fair_fem"]["paired_statistics"][candidate_method]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.9))
    colors = [BLUE, GRAY]
    method_labels = ["Learned-prior\nscreened MAP", "Fixed-start FEM"]
    hatches = ["", "//"]

    for ax, metric, title in (
        (axes[0, 0], "E_background", "Background-modulus error"),
        (axes[0, 1], "inclusion_ratio", "Inclusion-ratio error"),
    ):
        key = f"{metric}_median_relative_error"
        ci_key = f"{metric}_median_bootstrap_95_ci"
        vals = np.asarray([ai[key], fixed[key]]) * 100
        cis = np.asarray([ai[ci_key], fixed[ci_key]]) * 100
        yerr = np.vstack((vals - cis[:, 0], cis[:, 1] - vals))
        bars = ax.bar(
            np.arange(2),
            vals,
            yerr=yerr,
            capsize=4,
            color=colors,
            edgecolor=INK,
            hatch=hatches,
        )
        for bar, value, upper in zip(bars, vals, yerr[1], strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + upper + 1, f"{value:.1f}%", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(2), method_labels)
        ax.set_ylabel("Median relative error (%)")
        ax.set_title(title)
        ax.set_ylim(0, (vals + yerr[1]).max() * 1.32)
        ax.grid(axis="y", color=LIGHT)
        delta = paired[f"{metric}_relative_error"]
        # Paired difference of two percentages: report in percentage points.
        delta_pp = f"{100 * delta['candidate_minus_reference_median']:.1f}".replace("-", "\u2212")
        ax.text(
            0.03,
            0.96,
            f"Paired median Δ = {delta_pp} pp\nLearned win rate = {delta['candidate_win_rate']:.0%}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 3},
        )

    ax = axes[1, 0]
    records_by_method = {
        method: [row for row in payload["fair_fem"]["records"] if row["benchmark_method"] == method]
        for method in (candidate_method, reference_method)
    }
    termination = np.asarray([
        np.mean([row["optimizer_success"] for row in records_by_method[method]])
        for method in records_by_method
    ]) * 100
    accepted = np.asarray([
        np.mean([row["refinement_accepted"] for row in records_by_method[method]])
        for method in records_by_method
    ]) * 100
    x = np.arange(2)
    bars_term = ax.bar(x - 0.19, termination, 0.38, color=BLUE, edgecolor=INK, label="Algorithmic termination / continuation stop")
    bars_accept = ax.bar(x + 0.19, accepted, 0.38, color=ORANGE, edgecolor=INK, hatch="//", label="Accepted refinement")
    for bars, values in ((bars_term, termination), (bars_accept, accepted)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, method_labels)
    # Head-room above the 100 % bar keeps the legend clear of every bar.
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Synthetic cases (%)")
    ax.set_title("Termination and acceptance")
    ax.grid(axis="y", color=LIGHT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=7.5, frameon=False)

    ax = axes[1, 1]
    x = np.arange(2)
    evals = np.asarray([ai["function_evaluations_median"], fixed["function_evaluations_median"]])
    times = np.asarray([ai["wall_time_seconds_median"], fixed["wall_time_seconds_median"]])
    bars = ax.bar(x - 0.18, evals, 0.36, color=BLUE, edgecolor=INK, label="Function evaluations")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.18, times, 0.36, color=ORANGE, edgecolor=INK, hatch="//", label="Wall time")
    for bar, value in zip(bars, evals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}", ha="center", va="bottom", fontsize=7.5)
    for bar, value in zip(bars2, times, strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 5, f"{value:.1f} s", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, method_labels)
    ax.set_ylabel("Median function evaluations", color=BLUE)
    ax2.set_ylabel("Median wall time (s)", color=ORANGE)
    ax2.tick_params(axis="y", labelsize=7.5)
    ax.set_ylim(0, max(evals) * 1.15)
    ax2.set_ylim(0, max(times) * 1.15)
    ax.set_title("Compute to solution")
    ax.grid(axis="y", color=LIGHT)
    handles = [bars[0], bars2[0]]
    ax.legend(handles, ["Evaluations", "Wall time"], frameon=False, fontsize=7.5, loc="upper left")
    ax.text(
        0.03,
        0.66,
        f"Evaluation\nreduction {payload['fair_fem']['evaluation_reduction']:.1%}",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
    )
    for label, axis in zip(("A", "B", "C", "D"), axes.flat, strict=True):
        panel_label(axis, label, x=-0.16)
    fig.suptitle("Configured FEM procedures (50 synthetic cases)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0, w_pad=1.5)
    save_figure(fig, output_dir, "fig_fair_fem")


def fig_calibration(payload: dict[str, Any], output_dir: Path) -> None:
    calibration = payload["calibration"]
    test = calibration["test"]
    gates = payload["gates"]["gates"]
    records = calibration["records"]
    # Drawn at the final 6.5 in width: two square calibration panels on top,
    # the region-error panel plus a shared legend below (all text >= 7 pt).
    fig = plt.figure(figsize=(6.5, 6.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.78), hspace=0.42, wspace=0.34)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
    ]
    legend_ax = fig.add_subplot(grid[1, 1])
    legend_ax.axis("off")
    scatter_specs = [
        (
            axes[0],
            np.log([row["E_background_true"] for row in records]),
            np.log([row["E_background_estimated"] for row in records]),
            test["log_E_uncertainty"]["half_width"],
            test["log_E_uncertainty"]["coverage"]["90"],
            r"$\log E_b$",
            BLUE,
            "A",
        ),
        (
            axes[1],
            np.log([row["inclusion_ratio_true"] for row in records]),
            np.log([row["inclusion_ratio_estimated"] for row in records]),
            test["log_ratio_uncertainty"]["half_width"],
            test["log_ratio_uncertainty"]["coverage"]["90"],
            r"$\log \rho_E$",
            GREEN,
            "B",
        ),
    ]
    legend_handles: list[Any] = []
    for ax, truth, prediction, half_width, coverage, parameter, color, label in scatter_specs:
        covered = np.abs(prediction - truth) <= half_width
        lo = min(truth.min(), prediction.min() - half_width)
        hi = max(truth.max(), prediction.max() + half_width)
        diagonal = np.linspace(lo, hi, 200)
        band = ax.fill_between(diagonal, diagonal - half_width, diagonal + half_width, color=GRAY, alpha=0.15)
        (line,) = ax.plot(diagonal, diagonal, "--", color=INK, linewidth=1.2)
        ax.errorbar(
            truth,
            prediction,
            yerr=half_width,
            fmt="none",
            ecolor=color,
            alpha=0.15,
            elinewidth=0.6,
            capsize=0,
            zorder=1,
        )
        dots = ax.scatter(truth[covered], prediction[covered], s=16, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        misses = ax.scatter(truth[~covered], prediction[~covered], s=24, facecolor="white", edgecolor=ORANGE, linewidth=1.1, marker="D")
        if not legend_handles:
            legend_handles = [band, line, dots, misses]
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"True {parameter}")
        ax.set_ylabel(f"Predicted {parameter}")
        ax.set_title(f"{parameter}: truth vs prediction", fontsize=9.5)
        ax.grid(color=LIGHT)
        ax.text(
            0.04,
            0.96,
            f"90% band ±{half_width:.3f}\ncoverage {coverage:.0%}, $n={len(records)}$ test cases",
            transform=ax.transAxes,
            va="top",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 2.5},
        )
        panel_label(ax, label, x=-0.20)
    legend_ax.legend(
        legend_handles,
        ["90% split-conformal band", "$y=x$", "Covered case", "Not covered"],
        loc="center left",
        fontsize=7.5,
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.0,
        title="Panels A–B",
        title_fontsize=8,
        alignment="left",
    )

    heterogeneous = [row for row in records if row["inclusion_ratio_true"] > 1.0 + 1e-8]
    center_errors = np.asarray([
        np.linalg.norm(np.asarray(row["center_fraction_estimated"]) - np.asarray(row["center_fraction_true"]))
        for row in heterogeneous
    ]) * 100
    radius_errors = np.asarray([
        abs(row["radius_fraction_estimated"] / row["radius_fraction_true"] - 1)
        for row in heterogeneous
    ]) * 100
    ax = axes[2]
    region_values = [center_errors, radius_errors]
    region_colors = [PURPLE, ORANGE]
    positions = [0, 1]
    violins = ax.violinplot(region_values, positions=positions, widths=0.72, showextrema=False)
    for body, color in zip(violins["bodies"], region_colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
    boxes = ax.boxplot(region_values, positions=positions, widths=0.22, showfliers=False, patch_artist=True)
    for box, color in zip(boxes["boxes"], region_colors, strict=True):
        box.set_facecolor("white")
        box.set_edgecolor(color)
    rng = np.random.default_rng(5220)
    for position, values, color in zip(positions, region_values, region_colors, strict=True):
        ax.scatter(position + rng.normal(0, 0.045, len(values)), values, s=10, color=color, alpha=0.65, edgecolor="white", linewidth=0.25)
    thresholds = [gates["center_error"]["threshold"] * 100, gates["radius_error"]["threshold"] * 100]
    for position, threshold, color in zip(positions, thresholds, region_colors, strict=True):
        ax.plot([position - 0.34, position + 0.34], [threshold, threshold], "--", color=color, linewidth=1.3)
    ax.set_xticks(positions, ["Center", "Radius"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(top=max(np.max(center_errors), np.max(radius_errors)) * 1.45)
    ax.set_ylabel("Case-level error (%)")
    ax.set_title("Heterogeneous-case region errors", fontsize=9.5)
    ax.grid(axis="y", color=LIGHT)
    ax.text(
        0.04,
        0.96,
        f"median center {np.median(center_errors):.1f}%, "
        f"radius {np.median(radius_errors):.1f}%\n"
        f"$n={len(heterogeneous)}$ heterogeneous cases; dashed = criteria",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 2.5},
    )
    panel_label(ax, "C", x=-0.20)
    fig.suptitle("Track-summary PCA–Ridge calibration and region accuracy", y=0.985)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.08, top=0.90)
    save_figure(fig, output_dir, "fig_calibration")


def fig_response_baselines(payload: dict[str, Any], output_dir: Path) -> None:
    """Compare train-selected response regressors on the same held-out cases.

    Drawn at the final 6.5 in width (three rows): endpoint medians with
    bootstrap intervals (A-D), ceiling attainment (E) and the case-level
    PCA-Ridge/Ridge comparison (F). Value labels are omitted because they
    collided with the markers; the medians are tabulated in the main text.
    """
    benchmark = payload["baselines"]
    methods = benchmark["methods"]
    abbrev = {
        "training_population_prior": "Pop",
        "ridge": "R",
        "pls": "PLS",
        "extra_trees": "ET",
        "pca_ridge": "PCR",
    }
    full_names = {
        "training_population_prior": "Training-population prior",
        "ridge": "Ridge",
        "pls": "Partial least squares",
        "extra_trees": "ExtraTrees",
        "pca_ridge": "PCA–Ridge (primary)",
    }
    colors = [GRAY, "#56B4E9", PURPLE, ORANGE, GREEN]
    specifications = (
        (
            "E_background_median_relative_error",
            "E_background_relative_error",
            "Background modulus",
            0.15,
        ),
        (
            "inclusion_ratio_median_relative_error",
            "inclusion_ratio_relative_error",
            "Inclusion ratio",
            0.25,
        ),
        (
            "center_error_normalized_median",
            "center_error_normalized",
            "Region center",
            0.10,
        ),
        (
            "radius_relative_error_median",
            "radius_relative_error",
            "Region radius",
            0.12,
        ),
    )
    fig = plt.figure(figsize=(6.5, 7.1))
    grid = fig.add_gridspec(
        3, 2, hspace=0.6, wspace=0.36, height_ratios=(1.0, 1.0, 1.08),
        left=0.10, right=0.975, top=0.925, bottom=0.11,
    )
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(3)])
    positions = np.arange(len(methods))
    tick_labels = [abbrev[row["name"]] for row in methods]
    for panel, (ax, (metric, ci_key, title, gate)) in enumerate(
        zip(axes.flat[:4], specifications, strict=True)
    ):
        values = np.asarray([row["test_metrics"][metric] for row in methods]) * 100
        intervals = np.asarray(
            [row["bootstrap_95_ci"][ci_key] for row in methods]
        ) * 100
        errors = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))
        ax.errorbar(
            positions,
            values,
            yerr=errors,
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.5,
            zorder=2,
        )
        ax.scatter(
            positions,
            values,
            s=34,
            c=colors,
            edgecolor=INK,
            linewidth=0.6,
            zorder=3,
        )
        ax.axhline(
            gate * 100,
            color=ORANGE,
            linestyle="--",
            linewidth=1.1,
            label=f"Development-fixed criterion {gate:.0%}",
        )
        ax.set_xticks(positions, tick_labels)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_ylim(0, max(intervals[:, 1].max(), gate * 100) * 1.28)
        ax.set_ylabel("Median case-level\nerror (%)")
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", color=LIGHT)
        ax.legend(loc="upper right", fontsize=7.0)
        panel_label(ax, chr(ord("A") + panel), x=-0.20)

    # Companion panels: case-level ceiling attainment and the unresolved
    # PCA--Ridge/Ridge comparison.
    ax = axes[2, 0]
    gate_matrix = np.zeros((len(specifications), len(methods)))
    for row_index, (metric, ci_key, _, gate) in enumerate(specifications):
        for column_index, method in enumerate(methods):
            values = np.asarray(
                [
                    row[ci_key]
                    for row in method["records"]
                    if row.get(ci_key) is not None
                ]
            )
            gate_matrix[row_index, column_index] = np.mean(values <= gate)
    mesh = ax.pcolormesh(
        np.arange(len(methods) + 1),
        np.arange(len(specifications) + 1),
        gate_matrix * 100,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        edgecolors="white",
        linewidth=0.35,
    )
    ax.set_ylim(len(specifications), 0)
    for row_index in range(gate_matrix.shape[0]):
        for column_index in range(gate_matrix.shape[1]):
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{gate_matrix[row_index, column_index]:.0%}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if gate_matrix[row_index, column_index] > 0.58 else INK,
            )
    # pcolormesh cells span [k, k+1]; centre the tick labels and cell text.
    ax.set_xticks(np.arange(len(methods)) + 0.5, tick_labels, fontsize=8)
    ax.set_yticks(np.arange(4) + 0.5, ("$E_b$", "Ratio", "Center", "Radius"))
    ax.tick_params(length=0)
    ax.set_title("Cases meeting each criterion", fontsize=9.5)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    colorbar.solids.set_rasterized(False)  # keep the PDF fully vector
    colorbar.set_label("Cases within criterion (%)", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7)
    panel_label(ax, "E", x=-0.20)

    ax = axes[2, 1]
    records_by_method = {
        method["name"]: {row["patient_id"]: row for row in method["records"]}
        for method in methods
    }
    ridge_records = records_by_method["ridge"]
    primary_records = records_by_method["pca_ridge"]
    for metric, color, label in (
        ("E_background_relative_error", BLUE, "$E_b$"),
        ("inclusion_ratio_relative_error", GREEN, "Ratio"),
    ):
        x_values = np.asarray(
            [ridge_records[patient_id][metric] * 100 for patient_id in primary_records]
        )
        y_values = np.asarray(
            [primary_records[patient_id][metric] * 100 for patient_id in primary_records]
        )
        ax.scatter(
            x_values,
            y_values,
            s=14,
            color=color,
            alpha=0.65,
            edgecolor="white",
            linewidth=0.3,
            label=label,
        )
    limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot((0, limit), (0, limit), "--", color=INK, linewidth=0.9, label="Equal error")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ridge case error (%)")
    ax.set_ylabel("PCA–Ridge case error (%)")
    ax.set_title("PCA–Ridge vs Ridge", fontsize=9.5)
    ax.grid(color=LIGHT)
    ax.legend(loc="upper left", fontsize=7.0)
    panel_label(ax, "F", x=-0.20)
    fig.suptitle(
        "Common-input response regressors on 50 held-out synthetic cases",
        y=0.985,
        fontsize=10.5,
    )
    _abbreviation_key(
        fig,
        [(abbrev[row["name"]], full_names[row["name"]]) for row in methods],
        y=0.012,
        title="Key",
        left=0.10,
        inline=True,
    )
    save_figure(fig, output_dir, "fig_response_baselines")


def _sample_ion_vertices(path: Path, count: int, seed: int) -> np.ndarray:
    """Load and deterministically subsample one de-identified surface."""
    with np.load(path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"], dtype=np.float32)
    # Match the local adapter and reduce anatomical identifiability: retain only
    # the central 20% distance patch, never the complete patient-derived surface.
    full_center = np.mean(vertices, axis=0)
    distance = np.linalg.norm(vertices - full_center, axis=1)
    vertices = vertices[distance <= np.quantile(distance, 0.20)]
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(vertices), size=min(count, len(vertices)), replace=False)
    sample = vertices[indices]
    center = np.median(sample, axis=0)
    sample = sample - center
    scale = np.quantile(np.linalg.norm(sample, axis=1), 0.98)
    return sample / max(float(scale), 1e-8)


def fig_real_ion_cases(payload: dict[str, Any], output_dir: Path) -> None:
    """Show de-identified CT-derived geometry while preserving the no-material-truth boundary.

    The per-case QC numbers (vertex/face counts, Jacobians) that used to sit
    in a text box below each cloud are tabulated in the supplement instead,
    so the figure is a single row of darker, denser point clouds.
    """
    geometry_ids = payload["ood"]["geometry_ids"]
    fig = plt.figure(figsize=(6.5, 2.75))
    grid = fig.add_gridspec(1, len(geometry_ids), wspace=0.05, left=0.01, right=0.99, bottom=0.0, top=0.86)
    for index, geometry_id in enumerate(geometry_ids):
        detail = load_json(ION_FORMAL_ROOT / f"{geometry_id}.json")
        if not detail["model_output_stability"]["finite_outputs"]:
            raise ValueError(f"ION geometry {geometry_id} did not produce finite model output")
        vertices = _sample_ion_vertices(
            ION_MESH_ROOT / f"{geometry_id}.npz",
            count=9000,
            seed=7300 + index,
        )
        # Rotated 3-D point-surface view; only de-identified geometry is shown.
        ax = fig.add_subplot(grid[0, index], projection="3d")
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            s=1.6,
            color="#0B4F7A",
            alpha=0.85,
            linewidths=0,
            rasterized=False,
        )
        lower = np.quantile(vertices, 0.01, axis=0)
        upper = np.quantile(vertices, 0.99, axis=0)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        extent = upper - lower
        ax.set_box_aspect(np.maximum(extent, 1e-3), zoom=1.25)
        ax.view_init(elev=18, azim=-62)
        ax.set_axis_off()
        ax.set_title(f"De-identified geometry {index + 1}", fontsize=8.5, pad=2)
        panel_label(ax, chr(ord("A") + index), x=0.0, y=1.10)
    fig.suptitle(
        "Real ION CT-derived geometry: numerical feasibility only (no material truth)",
        y=0.995,
        fontsize=10,
    )
    save_figure(fig, output_dir, "fig_real_ion_cases")


def fig_multiseed_ood(payload: dict[str, Any], output_dir: Path) -> None:
    multi = payload["multiseed"]
    cohort_qc = load_json(ION_COHORT_QC)
    per_seed = multi["per_seed"]
    seeds = [row["seed"] for row in per_seed]
    e_errors = np.asarray([row["E_background_median_relative_error"] for row in per_seed]) * 100
    r_errors = np.asarray([row["inclusion_ratio_median_relative_error"] for row in per_seed]) * 100
    fig = plt.figure(figsize=(6.5, 3.1))
    # Panel C is a text box: its widest line ("FEM construction: 100% success")
    # needs ~1.7 in at 7.2 pt, so the slot is ~1.95 in wide.
    grid = fig.add_gridspec(
        1, 3, width_ratios=(1.0, 1.0, 1.45), wspace=0.40,
        left=0.085, right=0.985, bottom=0.17, top=0.80,
    )
    ax_e, ax_r, ax_ood = [fig.add_subplot(grid[0, index]) for index in range(3)]

    for ax, values, mean, std, title, color, ensemble, ci in (
        (
            ax_e,
            e_errors,
            multi["E_background_seed_mean"] * 100,
            multi["E_background_seed_std"] * 100,
            "Background modulus",
            BLUE,
            multi["ensemble_patient_E_median"] * 100,
            np.asarray(multi["ensemble_patient_E_bootstrap_95_ci"]) * 100,
        ),
        (
            ax_r,
            r_errors,
            multi["inclusion_ratio_seed_mean"] * 100,
            multi["inclusion_ratio_seed_std"] * 100,
            "Inclusion ratio",
            GREEN,
            multi["ensemble_patient_ratio_median"] * 100,
            np.asarray(multi["ensemble_patient_ratio_bootstrap_95_ci"]) * 100,
        ),
    ):
        ax.plot(seeds, values, "o-", color=color, linewidth=1.4, markersize=4, label="Per-seed median")
        ax.axhspan(mean - std, mean + std, color=color, alpha=0.14, label="Seed mean ± SD")
        ax.axhline(mean, color=color, linestyle="--", linewidth=1.0)
        # The ensemble marker sits one seed step after the last seed so that
        # its rotated tick label is spaced like the seed labels.
        ax.errorbar(
            seeds[-1] + 1.0,
            ensemble,
            yerr=[[ensemble - ci[0]], [ci[1] - ensemble]],
            fmt="D",
            markersize=4,
            color=ORANGE,
            capsize=3,
            label="Ensemble (95% CI)",
        )
        ax.set_xticks([*seeds, seeds[-1] + 1.0], [*map(str, seeds), "Ens."])
        ax.tick_params(axis="x", labelsize=7.5, labelrotation=40)
        ax.set_xlabel("Seed", fontsize=7.5)
        ax.set_ylabel("Median relative error (%)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", color=LIGHT)
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + 0.42 * (upper - lower))
        ax.legend(frameon=False, fontsize=7.0, loc="upper left", handlelength=1.4, borderaxespad=0.3)
    panel_label(ax_e, "A", x=-0.26)
    panel_label(ax_r, "B", x=-0.26)

    ax_ood.axis("off")
    geometry_count = int(cohort_qc["geometry_count"])
    passed_count = int(cohort_qc["passed_count"])
    minimum_deformation_jacobian = min(
        row["finite_element"]["minimum_deformation_jacobian"]
        for row in cohort_qc["records"]
    )
    lines = [
        "ION geometry OOD boundary",
        "",
        f"Geometry count: {geometry_count}",
        "Source: de-identified ION CT",
        f"FEM construction: {passed_count / geometry_count:.0%} success",
        f"Forward pass: {passed_count / geometry_count:.0%} success",
        f"Min. deform. Jacobian: {minimum_deformation_jacobian:.3f}",
        "",
        "Material truth: NOT REPORTED",
        "Scope: geometry stability only",
    ]
    box = FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        transform=ax_ood.transAxes,
        boxstyle="round,pad=0.02",
        facecolor="#F7F8FA",
        edgecolor=PURPLE,
        linewidth=1.4,
    )
    ax_ood.add_patch(box)
    ax_ood.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax_ood.transAxes,
        va="top",
        fontsize=7.2,
        linespacing=1.45,
    )
    if geometry_count < 5:
        ax_ood.text(
            0.05,
            0.04,
            "Small geometry-only sample; no population-level OOD claim.",
            transform=ax_ood.transAxes,
            fontsize=7.2,
            color=ORANGE,
            fontweight="bold",
        )
    panel_label(ax_ood, "C", x=-0.06)
    fig.suptitle(
        f"Five-seed stability and bounded OOD evidence (n={multi['patient_count']} held-out cases)",
        fontsize=10.5,
        fontweight="bold",
        y=0.985,
    )
    save_figure(fig, output_dir, "fig_multiseed_ood")


def fig_ion_ct_synthetic_pipeline(output_dir: Path) -> None:
    """Show privacy-minimized CT-to-local-FEM processing for all geometries."""
    from lung_inverse_rendering.ct_geometry import build_scene_from_ct_mesh

    privacy = load_json(ION_FIGURE_ASSETS / "privacy_manifest.json")
    records = sorted(privacy["records"], key=lambda item: item["geometry_id"])
    # Final 6.5 in layout: rows stay one geometry each (as in the caption);
    # the six ~1.1 in rows plus header fit in < 7 in of printed height.
    row_height = 0.98
    figure_height = 0.62 + row_height * len(records)
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, figure_height))
    grid = fig.add_gridspec(
        len(records),
        5,
        left=0.06,
        right=0.995,
        top=1.0 - 0.60 / figure_height,
        bottom=0.012,
        wspace=0.05,
        hspace=0.05,
    )
    column_titles = (
        "Windowed CT",
        "HU mask",
        "Mask overlay",
        "CT-derived\nsurface",
        "Local tetrahedral\nmesh",
    )
    # Column headers as figure text on one baseline: image axes shrink to
    # their aspect ratio and 3-D axes do not, so per-axes titles would sit at
    # different heights.
    header_y = grid[0, 0].get_position(fig).y1 + 0.004
    for column, title in enumerate(column_titles):
        cell = grid[0, column].get_position(fig)
        fig.text(
            0.5 * (cell.x0 + cell.x1),
            header_y,
            title,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="semibold",
            linespacing=1.15,
        )
    for row_index, record in enumerate(records):
        geometry_id = record["geometry_id"]
        for column, key in enumerate(("ct", "mask", "overlay")):
            ax = fig.add_subplot(grid[row_index, column])
            image = np.asarray(Image.open(ION_FIGURE_ASSETS / record["files"][key]))
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.axis("off")
            if column == 0:
                # Rotated row label; "Geometry n" (~0.65 in tall at 7.5 pt)
                # leaves a clear gap between the ~0.95 in rows, whereas the
                # longer "Test geometry n" ran into its neighbours.
                ax.text(
                    -0.06,
                    0.5,
                    f"Geometry {row_index + 1}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="semibold",
                )
        mesh_path = ION_MESH_ROOT / f"{geometry_id}.npz"
        with np.load(mesh_path, allow_pickle=False) as archive:
            vertices = np.asarray(archive["vertices"], dtype=float)
        selection = np.random.default_rng(2026 + row_index).choice(
            len(vertices), size=min(1800, len(vertices)), replace=False
        )
        surface = vertices[selection]
        surface = (surface - surface.mean(axis=0)) / np.maximum(
            np.ptp(surface, axis=0), 1e-8
        )
        ax_surface = fig.add_subplot(grid[row_index, 3], projection="3d")
        ax_surface.scatter(
            surface[:, 0], surface[:, 1], surface[:, 2], s=0.3, color=BLUE, alpha=0.55
        )
        ax_surface.view_init(18, -68)
        ax_surface.set_box_aspect((1.0, 1.0, 0.8), zoom=1.45)
        ax_surface.set_axis_off()

        scene = build_scene_from_ct_mesh(mesh_path, geometry_id=geometry_id, E_true=5000.0)
        nodes = scene["nodes"].detach().cpu().numpy()
        elems = scene["elems"].detach().cpu().numpy()
        ax_tet = fig.add_subplot(grid[row_index, 4], projection="3d")
        edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        segments = set()
        for tet in elems:
            for first, second in edge_pairs:
                segments.add(tuple(sorted((int(tet[first]), int(tet[second])))))
        for first, second in list(segments)[:: max(1, len(segments) // 700)]:
            ax_tet.plot(
                nodes[[first, second], 0],
                nodes[[first, second], 1],
                nodes[[first, second], 2],
                color=PURPLE,
                linewidth=0.25,
                alpha=0.55,
            )
        ax_tet.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=0.7, color=ORANGE)
        ax_tet.view_init(22, -60)
        ax_tet.set_box_aspect((1.0, 1.0, 0.8), zoom=1.45)
        ax_tet.set_axis_off()
    fig.suptitle(
        f"Privacy-minimized real-geometry protocol ({len(records)} locked test geometries)",
        y=0.992,
        fontsize=10.5,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "fig_ion_ct_synthetic_pipeline")


def _material_map(
    positions: np.ndarray,
    center: np.ndarray,
    radius: float,
    background: float,
    ratio: float,
) -> np.ndarray:
    occupancy = np.linalg.norm(positions - center[None, :], axis=1) <= radius
    return np.log(background * (1.0 + (ratio - 1.0) * occupancy))


def fig_ion_ct_material_recovery(output_dir: Path) -> None:
    """Visualize held-out hybrid estimates on a fixed heterogeneous scenario.

    The four imaged products of the original 6 x 4 design are kept as rows
    (synthetic truth, uncalibrated hybrid initializer, |hybrid - truth| in
    log E, and rest versus peak-deformed nodes) for the six held-out
    CT-derived geometries (columns). The two log-E rows share one colour
    scale and one colorbar, the error row has its own shared colorbar, and
    all text is >= 7 pt at 6.5 in width.
    """
    import torch

    benchmark = load_json(ION_SYNTH_RESULTS / "benchmark.json")
    manifest = load_json(ION_SYNTH_DATASET / "manifest.json")
    scenario_id = "scenario_010"
    deformation_display_scale = 2.5
    records = [
        row
        for row in benchmark["records"]
        if row["method"] == "ours_hybrid_initializer"
        and row["scenario_id"] == scenario_id
    ]
    records.sort(key=lambda row: row["geometry_id"])
    if len(records) != 6:
        raise ValueError(f"Expected six held-out geometries for {scenario_id}, found {len(records)}")
    patients = {row["geometry_id"]: row for row in manifest["patients"] if row["scenario_id"] == scenario_id}

    cases: list[dict[str, Any]] = []
    for record in records:
        patient = patients[record["geometry_id"]]
        experiment_path = ION_SYNTH_DATASET / patient["experiments"][0]["relative_path"]
        experiment = torch.load(experiment_path, map_location="cpu", weights_only=False)
        nodes = experiment["nodes"].detach().cpu().numpy()
        minimum, extent = nodes.min(axis=0), np.maximum(np.ptp(nodes, axis=0), 1e-8)
        positions = (nodes - minimum) / extent
        # Peak-frame nodal displacement of the first (press-left) load, drawn
        # magnified so that the deformation is visible at the page width.
        u_seq = experiment["u_seq"].detach().cpu().numpy().reshape(len(experiment["u_seq"]), -1, 3)
        peak = int(np.argmax(np.linalg.norm(u_seq, axis=2).max(axis=1)))
        deformed = (nodes + deformation_display_scale * u_seq[peak] - minimum) / extent
        truth = _material_map(
            positions,
            np.asarray(record["center_fraction_true"]),
            float(record["radius_fraction_true"]),
            float(record["E_background_true"]),
            float(record["inclusion_ratio_true"]),
        )
        prediction = _material_map(
            positions,
            np.asarray(record["center_fraction_estimated"]),
            float(record["radius_fraction_estimated"]),
            float(record["E_background_estimated"]),
            float(record["inclusion_ratio_estimated"]),
        )
        cases.append(
            {
                "positions": positions,
                "deformed": deformed,
                "truth": truth,
                "prediction": prediction,
                "error": np.abs(prediction - truth),
                "ratio_error": 100.0 * float(record["inclusion_ratio_relative_error"]),
                "E_error": 100.0 * float(record["E_background_relative_error"]),
            }
        )
    log_low = min(min(case["truth"].min(), case["prediction"].min()) for case in cases)
    log_high = max(max(case["truth"].max(), case["prediction"].max()) for case in cases)
    error_high = max(max(float(case["error"].max()), 1e-8) for case in cases)

    n_rows = 4
    fig = plt.figure(figsize=(6.5, 5.35))
    grid = fig.add_gridspec(
        n_rows,
        len(cases) + 1,
        width_ratios=[1.0] * len(cases) + [0.07],
        hspace=0.10,
        wspace=0.08,
        left=0.085,
        right=0.93,
        top=0.875,
        bottom=0.012,
    )
    axes = np.asarray(
        [[fig.add_subplot(grid[row, col]) for col in range(len(cases))] for row in range(n_rows)]
    )
    row_titles = (
        "Synthetic truth\nlog E",
        "Uncalibrated\nhybrid initializer",
        "|hybrid − truth|\n(log E)",
        f"Rest → peak\n(×{deformation_display_scale:g} displ.)",
    )
    log_mappable = error_mappable = None
    for column, case in enumerate(cases):
        positions = case["positions"]
        log_mappable = axes[0, column].scatter(
            positions[:, 0], positions[:, 1], c=case["truth"], s=4,
            cmap="viridis", vmin=log_low, vmax=log_high, edgecolors="none",
        )
        axes[1, column].scatter(
            positions[:, 0], positions[:, 1], c=case["prediction"], s=4,
            cmap="viridis", vmin=log_low, vmax=log_high, edgecolors="none",
        )
        error_mappable = axes[2, column].scatter(
            positions[:, 0], positions[:, 1], c=case["error"], s=4,
            cmap="magma", vmin=0.0, vmax=error_high, edgecolors="none",
        )
        axes[3, column].scatter(
            positions[:, 0], positions[:, 1], s=3, color="#B8C2CC", edgecolors="none", zorder=2,
        )
        axes[3, column].scatter(
            case["deformed"][:, 0], case["deformed"][:, 1], s=3, color=BLUE, alpha=0.85,
            edgecolors="none", zorder=3,
        )
        for row in range(n_rows):
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_xlim(-0.04, 1.04)
            axes[row, column].set_ylim(-0.04, 1.04)
        axes[3, column].set_xlim(
            min(-0.04, case["deformed"][:, 0].min() - 0.02), max(1.04, case["deformed"][:, 0].max() + 0.02)
        )
        axes[3, column].set_ylim(
            min(-0.04, case["deformed"][:, 1].min() - 0.02), max(1.04, case["deformed"][:, 1].max() + 0.02)
        )
        axes[0, column].set_title(
            f"Geometry {column + 1}\nratio err {case['ratio_error']:.1f}%\n$E_b$ err {case['E_error']:.1f}%",
            fontsize=7.2,
            pad=3,
            linespacing=1.25,
        )
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=7.5, linespacing=1.2)
    log_cax = fig.add_subplot(grid[0:2, len(cases)])
    log_cbar = fig.colorbar(log_mappable, cax=log_cax)
    log_cbar.set_label("log E (shared scale)", fontsize=7.5)
    log_cbar.ax.tick_params(labelsize=7.5)
    error_cax = fig.add_subplot(grid[2, len(cases)])
    error_cbar = fig.colorbar(error_mappable, cax=error_cax)
    error_cbar.set_label("|Δ log E|", fontsize=7.5)
    error_cbar.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        "Held-out synthetic material on CT-derived geometry (scenario 10)",
        y=0.995,
        fontsize=10,
    )
    save_figure(fig, output_dir, "fig_ion_ct_material_recovery")


def fig_ion_ct_method_comparison(output_dir: Path) -> None:
    """Per-geometry medians for every estimator on the 30-scene controlled subset.

    Four stacked full-width panels (one per endpoint) share the estimator
    axis; categories use the short labels in ``CT_METHOD_ABBREV`` and the
    full names are printed in an in-figure key, so no rotated tick text.
    """
    benchmark = load_json(ION_PHYSICS_RESULTS / "benchmark.json")
    # Same estimator order and evidence-tier separators as the main-text
    # scene-distribution figure (``fig_ion_ct_error_distributions``):
    # R, PLS, ET, PCR, Pop, GNN, Hyb, FEM-f, FEM-m, MAP, OOD, Ref*, with
    # dotted lines after the common-input tier and before the reference arm.
    methods = _ct_available_methods(benchmark)
    tier_common_count = sum(
        method in benchmark["protocol"]["evidence_tiers"]["common_input"]
        for method in methods
    )
    metrics = (
        ("E_background_relative_error", "Background E\nerror (%)", 15.0),
        ("inclusion_ratio_relative_error", "Modulus-ratio\nerror (%)", 25.0),
        ("center_error_normalized", "Center error\n(% extent)", 10.0),
        ("radius_relative_error", "Radius\nerror (%)", 12.0),
    )
    geometry_ids = sorted(
        {row["geometry_id"] for row in benchmark["geometry_aggregates"]}
    )
    lookup = {
        (row["method"], row["geometry_id"]): row
        for row in benchmark["geometry_aggregates"]
    }
    # Panel B (modulus ratio) carries three geometry medians above 100 %
    # that would otherwise crush the remaining points into the bottom third;
    # it is drawn on a broken axis (0-90 % | 100-140 %) so every median stays
    # visible at its true value.
    broken_metric = "inclusion_ratio_relative_error"
    broken_lower, broken_upper = (0.0, 90.0), (100.0, 142.0)
    fig = plt.figure(figsize=(6.5, 7.0))
    grid = fig.add_gridspec(
        len(metrics), 1, hspace=0.27, left=0.115, right=0.985, top=0.875, bottom=0.205,
    )
    colors = plt.colormaps["tab10"](
        np.linspace(0.0, 0.8, len(geometry_ids))
    )
    offsets = np.linspace(-0.22, 0.22, len(geometry_ids))
    x = np.arange(len(methods))

    def draw(ax: plt.Axes, metric: str, gate: float) -> None:
        all_values: list[list[float]] = []
        for geometry_index, geometry_id in enumerate(geometry_ids):
            values = np.asarray(
                [
                    100
                    * lookup[(method, geometry_id)]["metrics"][metric].get(
                        "median", np.nan
                    )
                    for method in methods
                ],
                dtype=float,
            )
            all_values.append(values.tolist())
            ax.scatter(
                x + offsets[geometry_index],
                values,
                color=colors[geometry_index],
                edgecolor="white",
                linewidth=0.3,
                s=16,
                alpha=0.9,
                zorder=3,
                label=f"Geometry {geometry_index + 1}",
            )
        medians = np.nanmedian(np.asarray(all_values, dtype=float), axis=0)
        ax.scatter(
            x,
            medians,
            marker="D",
            s=22,
            color=INK,
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
            label="Across-geometry median",
        )
        ax.axhline(gate, color=INK, linestyle="--", linewidth=0.9, label="Development-fixed gate")
        boundaries = [tier_common_count - 0.5]
        if "fem_oracle_region_force" in methods:
            boundaries.append(methods.index("fem_oracle_region_force") - 0.5)
        for boundary in boundaries:
            ax.axvline(boundary, color=GRAY, linestyle=":", linewidth=0.9)
        ax.set_xticks(x, [CT_METHOD_ABBREV[method] for method in methods])
        ax.tick_params(axis="x", labelsize=8, labelbottom=True)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.grid(axis="y", color=LIGHT)

    axes: list[plt.Axes] = []
    first_axis: plt.Axes | None = None
    for index, (metric, ylabel, gate) in enumerate(metrics):
        letter = chr(ord("A") + index)
        if metric == broken_metric:
            upper, lower = _broken_axes(fig, grid[index], sharex=first_axis, upper_ratio=0.4)
            draw(upper, metric, gate)
            draw(lower, metric, gate)
            upper.set_ylim(*broken_upper)
            upper.set_yticks([100, 140])
            lower.set_ylim(*broken_lower)
            lower.set_yticks([0, 25, 50, 75])
            _broken_ylabel(upper, lower, ylabel)
            _slot_panel_label(fig, grid[index], letter, x=-0.10, y=1.16)
            axes.append(lower)
            first_axis = first_axis or lower
            continue
        ax = fig.add_subplot(grid[index], sharex=first_axis)
        first_axis = first_axis or ax
        draw(ax, metric, gate)
        ax.set_ylabel(ylabel)
        panel_label(ax, letter, x=-0.10, y=1.16)
        axes.append(ax)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("Estimator, 30-scene controlled subset (key below)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=GRAY, linestyle=":", lw=0.9))
    legend_labels.append("Evidence-tier boundary")
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        fontsize=7.5,
        ncol=5,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.suptitle(
        "Held-out geometry comparison (5 controlled FEM scenes per CT geometry)",
        y=0.99,
        fontsize=10.5,
    )
    _abbreviation_key(fig, _ct_key_pairs(methods), y=0.008, ncol=2, left=0.115)
    save_figure(fig, output_dir, "fig_ion_ct_method_comparison")


def _ct_available_methods(benchmark: dict[str, Any]) -> list[str]:
    return [method for method in CT_METHOD_ORDER if method in benchmark["methods"]]


def _ct_boxplot(
    ax: plt.Axes,
    benchmark: dict[str, Any],
    methods: list[str],
    metric: str,
    *,
    scale: float = 1.0,
) -> None:
    records = benchmark["records"]
    values = [
        [
            scale * float(row[metric])
            for row in records
            if row["method"] == method and row.get(metric) is not None
        ]
        for method in methods
    ]
    boxes = ax.boxplot(
        values,
        positions=np.arange(len(methods)),
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": INK, "linewidth": 1.2},
        whiskerprops={"color": GRAY, "linewidth": 0.7},
        capprops={"color": GRAY, "linewidth": 0.7},
        flierprops={
            "marker": ".",
            "markersize": 2.5,
            "markerfacecolor": GRAY,
            "markeredgecolor": "none",
            "alpha": 0.4,
        },
    )
    tiers = benchmark["protocol"]["evidence_tiers"]
    tier_lookup = {
        method: tier for tier, tier_methods in tiers.items() for method in tier_methods
    }
    for patch, method in zip(boxes["boxes"], methods):
        patch.set_facecolor(CT_TIER_COLORS[tier_lookup[method]])
        patch.set_alpha(0.72)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.6)
    # Short horizontal category labels (>= 7 pt); the full names are written
    # in the figure key by the caller.
    ax.set_xticks(
        np.arange(len(methods)),
        [CT_METHOD_ABBREV[method] for method in methods],
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    common_count = sum(
        method in benchmark["protocol"]["evidence_tiers"]["common_input"]
        for method in methods
    )
    oracle_index = (
        methods.index("fem_oracle_region_force")
        if "fem_oracle_region_force" in methods
        else len(methods)
    )
    ax.axvline(common_count - 0.5, color=GRAY, linestyle=":", linewidth=0.9)
    ax.axvline(oracle_index - 0.5, color=GRAY, linestyle=":", linewidth=0.9)
    ax.grid(axis="y", color=LIGHT)


def fig_ion_ct_error_distributions(output_dir: Path) -> None:
    """Show all scene-level errors instead of geometry medians alone.

    Four stacked full-width panels (one per endpoint) so that the twelve
    estimator boxes get ~0.5 in each and horizontal >= 7 pt short labels;
    the full estimator names are printed in the in-figure key.
    """
    benchmark = load_json(ION_PHYSICS_RESULTS / "benchmark.json")
    scene_count = (
        int(benchmark["protocol"]["geometry_cluster_count"])
        * int(benchmark["protocol"]["scenarios_per_test_geometry"])
    )
    metrics = (
        ("E_background_relative_error", "Background E\nerror (%)", 15.0),
        ("inclusion_ratio_relative_error", "Modulus-ratio\nerror (%)", 25.0),
        ("center_error_normalized", "Center error\n(% extent)", 10.0),
        ("radius_relative_error", "Radius\nerror (%)", 12.0),
    )
    ct_methods = _ct_available_methods(benchmark)
    # Panel B (modulus ratio): a handful of fliers reach ~300 % and would
    # squeeze every box into the bottom third, so the panel is drawn on a
    # broken axis (0-100 % | 100-310 %); boxes, whiskers and fliers keep
    # their true values on both segments.
    broken_metric = "inclusion_ratio_relative_error"
    broken_lower, broken_upper = (0.0, 100.0), (100.0, 315.0)
    fig = plt.figure(figsize=(6.5, 6.6))
    grid = fig.add_gridspec(
        len(metrics), 1, hspace=0.3, left=0.115, right=0.985, top=0.895, bottom=0.205,
    )

    def draw(ax: plt.Axes, metric: str, gate: float) -> None:
        _ct_boxplot(ax, benchmark, ct_methods, metric, scale=100.0)
        ax.tick_params(axis="x", labelbottom=True)
        ax.axhline(
            gate,
            color=INK,
            linestyle="--",
            linewidth=0.9,
            label="Development-fixed gate",
        )

    axes: list[plt.Axes] = []
    first_axis: plt.Axes | None = None
    for index, (metric, ylabel, gate) in enumerate(metrics):
        letter = chr(ord("A") + index)
        if metric == broken_metric:
            upper, lower = _broken_axes(fig, grid[index], sharex=first_axis, upper_ratio=0.4)
            draw(upper, metric, gate)
            draw(lower, metric, gate)
            upper.set_ylim(*broken_upper)
            upper.set_yticks([200, 300])
            lower.set_ylim(*broken_lower)
            lower.set_yticks([0, 25, 50, 75, 100])
            _broken_ylabel(upper, lower, ylabel)
            _slot_panel_label(fig, grid[index], letter, x=-0.10, y=1.16)
            axes.append(lower)
            first_axis = first_axis or lower
            continue
        ax = fig.add_subplot(grid[index], sharex=first_axis)
        first_axis = first_axis or ax
        draw(ax, metric, gate)
        ax.set_ylabel(ylabel)
        panel_label(ax, letter, x=-0.10, y=1.16)
        axes.append(ax)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("Estimator (full names in the key below)")
    handles = [
        Line2D([0], [0], color=CT_TIER_COLORS["common_input"], lw=6, alpha=0.72),
        Line2D([0], [0], color=CT_TIER_COLORS["secondary"], lw=6, alpha=0.72),
        Line2D([0], [0], color=CT_TIER_COLORS["oracle"], lw=6, alpha=0.72),
        Line2D([0], [0], color=INK, linestyle="--", lw=0.9),
        Line2D([0], [0], color=GRAY, linestyle=":", lw=0.9),
        Line2D([0], [0], marker="s", color=INK, markerfacecolor="white", lw=0, markersize=5),
    ]
    fig.legend(
        handles,
        ("Common input", "Secondary", "Ref*", "Development-fixed gate", "Evidence-tier boundary", "Box: scene distribution"),
        loc="upper center",
        ncol=3,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.972),
        frameon=False,
        columnspacing=1.6,
    )
    fig.suptitle(
        f"Controlled CT-derived synthetic mechanics: {scene_count} scenes, six geometries",
        y=0.99,
        fontsize=10.5,
    )
    _abbreviation_key(fig, _ct_key_pairs(ct_methods), y=0.008, ncol=2, left=0.115)
    save_figure(fig, output_dir, "fig_ion_ct_error_distributions")


def fig_ion_ct_node_material_metrics(output_dir: Path) -> None:
    """Expose node-map quality and the class-imbalance failure mode."""
    benchmark = load_json(ION_PHYSICS_RESULTS / "benchmark.json")
    methods = _ct_available_methods(benchmark)
    metrics = (
        ("node_log_E_mae", "Node log-E MAE", 1.0),
        ("material_map_correlation", "Material-map correlation", 1.0),
        ("partition_soft_dice", "Partition soft Dice", 1.0),
    )
    # Three stacked full-width panels so the twelve short estimator labels
    # stay horizontal and >= 7 pt; full names are in the in-figure key.
    fig, axes = plt.subplots(
        3, 1, figsize=(6.5, 6.4), sharex=True,
        gridspec_kw={"hspace": 0.28, "left": 0.115, "right": 0.985, "top": 0.93, "bottom": 0.2},
    )
    for index, (ax, (metric, ylabel, scale)) in enumerate(zip(axes, metrics)):
        _ct_boxplot(ax, benchmark, methods, metric, scale=scale)
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylabel(ylabel)
        if metric != "node_log_E_mae":
            ax.set_ylim(-0.08, 1.02)
        panel_label(ax, chr(ord("A") + index), x=-0.10, y=1.18)
    axes[-1].set_xlabel("Estimator (full names in the key below)")
    fig.suptitle(
        "Node-level material-map recovery on held-out CT geometries",
        y=0.99,
        fontsize=10.5,
    )
    _abbreviation_key(fig, _ct_key_pairs(methods), y=0.008, ncol=2, left=0.115)
    save_figure(fig, output_dir, "fig_ion_ct_node_material_metrics")


def fig_ion_ct_physics_tradeoff(output_dir: Path) -> None:
    """Compare physics accuracy, computational cost, and force coupling."""
    benchmark = load_json(ION_PHYSICS_RESULTS / "benchmark.json")
    physics_methods = [
        method
        for method in (
            "fem_fixed_init",
            "fem_deterministic_multistart",
            "fem_learned_screened_map",
            "ours_ood_screened_fem",
            "fem_oracle_region_force",
        )
        if method in benchmark["methods"]
    ]
    colors = {
        "fem_fixed_init": BLUE,
        "fem_deterministic_multistart": GREEN,
        "fem_learned_screened_map": ORANGE,
        "ours_ood_screened_fem": "#CC79A7",
        "fem_oracle_region_force": PURPLE,
    }
    markers = {
        "fem_fixed_init": "o",
        "fem_deterministic_multistart": "s",
        "fem_learned_screened_map": "^",
        "ours_ood_screened_fem": "D",
        "fem_oracle_region_force": "P",
    }
    # Final 6.5 in layout: 2 x 2 panels, one shared legend strip on top (short
    # estimator labels) and the full-name estimator key underneath.
    # The bottom margin holds the C/D x-labels plus the three-row estimator
    # key with its title; 0.20 of a 6.4 in figure keeps a clear gap between
    # the key title and the panel C x-label.
    fig, axes_grid = plt.subplots(
        2,
        2,
        figsize=(FINAL_WIDTH_IN, 6.4),
        gridspec_kw={
            "left": 0.10,
            "right": 0.985,
            "top": 0.84,
            "bottom": 0.20,
            "hspace": 0.42,
            "wspace": 0.34,
        },
    )
    axes = axes_grid.flat
    geometry_rows = benchmark["geometry_aggregates"]
    for method in physics_methods:
        rows = [row for row in geometry_rows if row["method"] == method]
        x = [
            100 * row["metrics"]["E_background_relative_error"]["median"]
            for row in rows
        ]
        y = [
            100 * row["metrics"]["inclusion_ratio_relative_error"]["median"]
            for row in rows
        ]
        axes[0].scatter(
            x,
            y,
            s=30,
            marker=markers[method],
            color=colors[method],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            label=CT_METHOD_ABBREV[method],
        )
    axes[0].axvline(15, color=INK, linestyle="--", linewidth=0.9)
    axes[0].axhline(25, color=INK, linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("Background E error (%, geometry median)")
    axes[0].set_ylabel("Modulus-ratio error (%, geometry median)")
    axes[0].grid(color=LIGHT)

    nonoracle = [method for method in physics_methods if "oracle" not in method]
    summaries = benchmark["methods"]
    evaluations = [
        summaries[method]["nested_descriptive"]["geometry_level_medians"][
            "function_evaluations"
        ]["median"]
        for method in nonoracle
    ]
    times = [
        summaries[method]["nested_descriptive"]["geometry_level_medians"][
            "wall_time_seconds"
        ]["median"]
        for method in nonoracle
    ]
    for method, evaluation_count, wall_time in zip(
        nonoracle, evaluations, times
    ):
        axes[1].scatter(
            evaluation_count,
            wall_time,
            s=46,
            marker=markers[method],
            color=colors[method],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    # Every method shares the 16-evaluation budget, so the markers stack at
    # one abscissa; write the wall-time values beside them, grouping any
    # methods whose medians would otherwise print on top of each other.
    time_span = max(max(times) - min(times), 1e-6)
    groups: list[list[int]] = []
    for index in sorted(range(len(nonoracle)), key=lambda i: times[i]):
        if groups and times[index] - times[groups[-1][-1]] < 0.08 * time_span:
            groups[-1].append(index)
        else:
            groups.append([index])
    for group in groups:
        text = "  ·  ".join(
            f"{CT_METHOD_ABBREV[nonoracle[i]]} {times[i]:.0f} s" for i in group
        )
        axes[1].annotate(
            text,
            (evaluations[group[0]], float(np.mean([times[i] for i in group]))),
            xytext=(9, 0),
            textcoords="offset points",
            fontsize=7.2,
            va="center",
            ha="left",
        )
    unique_evaluations = sorted(set(float(value) for value in evaluations))
    axes[1].set_xticks(unique_evaluations, [f"{value:g}" for value in unique_evaluations])
    axes[1].set_xlim(min(unique_evaluations) - 1.0, max(unique_evaluations) + 7.0)
    axes[1].set_ylim(0, max(times) * 1.18)
    axes[1].set_xlabel("Function evaluations (geometry median)")
    axes[1].set_ylabel("Wall time (s, geometry median)")
    axes[1].grid(color=LIGHT)

    records = benchmark["records"]
    for method in nonoracle:
        rows = [row for row in records if row["method"] == method]
        axes[2].scatter(
            [100 * row["force_scale_relative_error"] for row in rows],
            [100 * row["E_background_relative_error"] for row in rows],
            s=12,
            marker=markers[method],
            color=colors[method],
            alpha=0.55,
            edgecolor="none",
        )
    axes[2].set_xlabel("Force-scale error (%, per scene)")
    axes[2].set_ylabel("Background E error (%, per scene)")
    axes[2].grid(color=LIGHT)

    diagnostic_methods = physics_methods
    termination = []
    accepted = []
    screened = []
    for method in diagnostic_methods:
        rows = [row for row in records if row["method"] == method]
        termination.append(
            100 * np.mean([float(row["algorithmic_termination"]) for row in rows])
        )
        accepted.append(
            100 * np.mean([float(row["refinement_accepted"]) for row in rows])
        )
        screened.append(
            100 * np.mean([float(row["screening_rejected"]) for row in rows])
        )
    x = np.arange(len(diagnostic_methods))
    width = 0.26
    axes[3].bar(
        x - width,
        termination,
        width,
        color=BLUE,
        alpha=0.85,
        label="Algorithmic termination",
    )
    axes[3].bar(
        x,
        accepted,
        width,
        color=GREEN,
        alpha=0.85,
        hatch="///",
        edgecolor="white",
        linewidth=0.3,
        label="Refinement accepted",
    )
    axes[3].bar(
        x + width,
        screened,
        width,
        color=ORANGE,
        alpha=0.85,
        hatch="...",
        edgecolor="white",
        linewidth=0.3,
        label="Screening rejected",
    )
    axes[3].set_xticks(x, [CT_METHOD_ABBREV[method] for method in diagnostic_methods])
    axes[3].set_ylim(0, 108)
    axes[3].set_ylabel("Scenes (%)")
    axes[3].set_xlabel("Estimator (see key)")
    axes[3].grid(axis="y", color=LIGHT)
    method_handles, method_labels = axes[0].get_legend_handles_labels()
    decision_handles, decision_labels = axes[3].get_legend_handles_labels()
    # Two one-row legends: estimators (A-C markers) above, D's bar types below.
    fig.legend(
        method_handles,
        method_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.957),
        ncol=len(method_handles),
        fontsize=7.4,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    fig.legend(
        decision_handles,
        decision_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.922),
        ncol=len(decision_handles),
        fontsize=7.4,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    for index, ax in enumerate(axes):
        panel_label(ax, chr(ord("A") + index), x=-0.24, y=1.12)
    _abbreviation_key(
        fig,
        _ct_key_pairs(physics_methods),
        y=0.012,
        ncol=2,
        fontsize=7.3,
        title="Estimator key (all estimators share the same evaluation budget)",
        left=0.06,
        right=0.99,
    )
    fig.suptitle(
        "Inverse-FEM accuracy, efficiency, and force calibration",
        y=0.99,
        fontsize=10.5,
    )
    save_figure(fig, output_dir, "fig_ion_ct_physics_tradeoff")


def fig_ion_ct_multiview_tracks(output_dir: Path) -> None:
    """Display four loads × three views, each as path + peak-frame columns."""
    import torch

    manifest = load_json(ION_SYNTH_DATASET / "manifest.json")
    patient = next(
        row
        for row in manifest["patients"]
        if row["scenario_id"] == "scenario_010"
    )
    # Protocol has three calibrated cameras; densify columns as path | peak per view.
    n_views = 3
    n_loads = len(patient["experiments"])
    # Final 6.5 in layout: header row for two-line column titles, then one
    # row per load; the legend strip sits between the title and the header.
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 4.45))
    grid = fig.add_gridspec(
        n_loads + 1,
        2 * n_views,
        height_ratios=(0.36,) + (1.0,) * n_loads,
        left=0.055,
        right=0.99,
        top=0.865,
        bottom=0.012,
        wspace=0.10,
        hspace=0.12,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[row + 1, col]) for col in range(2 * n_views)]
            for row in range(n_loads)
        ]
    )
    for view_index in range(n_views):
        for kind_index, kind in enumerate(("7-frame path", "peak flow")):
            ax_header = fig.add_subplot(grid[0, 2 * view_index + kind_index])
            ax_header.axis("off")
            ax_header.text(
                0.5,
                0.0,
                f"View {view_index + 1}\n{kind}",
                transform=ax_header.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="semibold",
                linespacing=1.15,
            )
    frame_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 7))
    for load_index, experiment_row in enumerate(patient["experiments"]):
        experiment = torch.load(
            ION_SYNTH_DATASET / experiment_row["relative_path"],
            map_location="cpu",
            weights_only=False,
        )
        uv = experiment["image_uv_deformed_multiview_seq"].detach().cpu().numpy()
        uv_rest = experiment["image_uv_rest_multiview_seq"].detach().cpu().numpy()
        visible = (
            experiment["image_visibility_multiview_seq"].detach().cpu().numpy()
        )
        confidence = (
            experiment["image_occlusion_confidence_multiview_seq"]
            .detach()
            .cpu()
            .numpy()
        )
        # Peak frame from 3-D motion magnitude (same rule as qualitative figure).
        u_seq = np.asarray(experiment["u_seq"]).reshape(7, -1, 3)
        peak = int(np.argmax(np.linalg.norm(u_seq, axis=2).max(axis=1)))
        for view_index in range(n_views):
            ax_path = axes[load_index, 2 * view_index]
            ax_peak = axes[load_index, 2 * view_index + 1]
            for frame_index in range(7):
                mask = visible[frame_index, view_index]
                ax_path.scatter(
                    uv[frame_index, view_index, mask, 0],
                    uv[frame_index, view_index, mask, 1],
                    s=7,
                    color=frame_colors[frame_index],
                    alpha=np.clip(
                        float(np.mean(confidence[frame_index, view_index, mask]))
                        if np.any(mask)
                        else 0.1,
                        0.15,
                        0.9,
                    ),
                    edgecolors="none",
                )
            for node_index in range(0, uv.shape[2], 4):
                mask = visible[:, view_index, node_index]
                if np.count_nonzero(mask) >= 2:
                    ax_path.plot(
                        uv[mask, view_index, node_index, 0],
                        uv[mask, view_index, node_index, 1],
                        color=GRAY,
                        linewidth=0.45,
                        alpha=0.38,
                    )
            # Peak-frame rest→deformed evidence for the same view.
            peak_mask = visible[peak, view_index]
            rest = uv_rest[peak, view_index]
            deformed = uv[peak, view_index]
            conf = np.clip(confidence[peak, view_index], 0, 1)
            ids = np.where(
                peak_mask
                & np.isfinite(rest).all(axis=1)
                & np.isfinite(deformed).all(axis=1)
            )[0]
            if len(ids) > 28:
                ids = ids[np.argsort(conf[ids])[-28:]]
            scale = 28.0
            delta = (deformed[ids] - rest[ids]) * scale
            ax_peak.scatter(rest[ids, 0], rest[ids, 1], s=5, color=INK, alpha=0.5)
            ax_peak.quiver(
                rest[ids, 0],
                rest[ids, 1],
                delta[:, 0],
                delta[:, 1],
                conf[ids],
                cmap="viridis",
                clim=(0, 1),
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.009,
                headwidth=3.4,
                headlength=4.0,
            )
            # Share the data window of the path panel with the peak-flow panel
            # of the same view (union of both autoscaled ranges), so that a
            # view with only a few visible peak-frame nodes is not squeezed
            # into a thin band by the equal-aspect autoscale.
            path_x, path_y = ax_path.get_xlim(), ax_path.get_ylim()
            peak_x, peak_y = ax_peak.get_xlim(), ax_peak.get_ylim()
            shared_x = (min(path_x[0], peak_x[0]), max(path_x[1], peak_x[1]))
            shared_y = (min(path_y[0], peak_y[0]), max(path_y[1], peak_y[1]))
            for ax in (ax_path, ax_peak):
                ax.set_xlim(*shared_x)
                ax.set_ylim(*shared_y)
                ax.invert_yaxis()
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
            if view_index == 0:
                ax_path.set_ylabel(
                    experiment_row["name"].replace("_", " ").capitalize(),
                    fontsize=7.5,
                    labelpad=3,
                )
    frame_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=frame_colors[index],
            markersize=5,
            label=f"Frame {index}",
        )
        for index in (0, 3, 6)
    ]
    encoding_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GRAY,
            alpha=0.55,
            markersize=6,
            label="Dot opacity = occlusion confidence",
        ),
        Line2D(
            [0],
            [0],
            color=GRAY,
            linewidth=1.0,
            label="Gray line = same-node trajectory",
        ),
        Line2D(
            [0],
            [0],
            color=plt.cm.viridis(0.85),
            linewidth=1.6,
            label="Arrow colour = confidence (peak flow)",
        ),
    ]
    # Two legend rows: frame colours, then the three encodings.
    fig.legend(
        handles=frame_handles + encoding_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=7.2,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    fig.suptitle(
        "Synchronized multiview trajectories under four load directions",
        y=0.992,
        fontsize=10,
    )
    save_figure(fig, output_dir, "fig_ion_ct_multiview_tracks")


def fig_ion_ct_physics_material_maps(output_dir: Path) -> None:
    """Compare learned, measured-force FEM, and oracle nodal material fields."""
    import torch

    benchmark = load_json(ION_PHYSICS_RESULTS / "benchmark.json")
    manifest = load_json(ION_SYNTH_DATASET / "manifest.json")
    scenario_id = "scenario_000"
    methods = ("pca_ridge", "fem_fixed_init", "fem_oracle_region_force")
    lookup = {
        (row["geometry_id"], row["method"]): row
        for row in benchmark["records"]
        if row["scenario_id"] == scenario_id and row["method"] in methods
    }
    available_geometry_ids = {
        geometry_id
        for geometry_id, method in lookup
        if method == "pca_ridge"
    }
    patients = sorted(
        (
            row
            for row in manifest["patients"]
            if row["scenario_id"] == scenario_id
            and row["geometry_id"] in available_geometry_ids
        ),
        key=lambda row: row["geometry_id"],
    )[:3]
    columns = (
        "Synthetic truth",
        "PCA–Ridge",
        "Fixed-start FEM",
        "Reference*",
        "Fixed-FEM error\n|Δ log E|",
    )
    # Build every map first. The four log-E columns keep one colour scale per
    # geometry (as before, so the PCA-Ridge scale miss stays visible on every
    # row) while the error column shares a single colorbar across rows.
    cases: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for patient in patients:
        experiment = torch.load(
            ION_SYNTH_DATASET / patient["experiments"][0]["relative_path"],
            map_location="cpu",
            weights_only=False,
        )
        nodes = experiment["nodes"].detach().cpu().numpy()
        minimum = nodes.min(axis=0)
        extent = np.maximum(np.ptp(nodes, axis=0), 1e-8)
        positions = (nodes - minimum) / extent
        truth = _material_map(
            positions,
            np.asarray(patient["inclusion_center_fraction"]),
            float(patient["inclusion_radius_fraction"]),
            float(patient["E_background"]),
            float(patient["inclusion_ratio"]),
        )
        maps = [truth]
        for method in methods:
            record = lookup[(patient["geometry_id"], method)]
            maps.append(
                _material_map(
                    positions,
                    np.asarray(record["center_fraction_estimated"]),
                    float(record["radius_fraction_estimated"]),
                    float(record["E_background_estimated"]),
                    float(record["inclusion_ratio_estimated"]),
                )
            )
        maps.append(np.abs(maps[2] - truth))
        cases.append((positions, maps))
    error_high = max(max(float(maps[4].max()), 1e-8) for _, maps in cases)

    # Final 6.5 in layout per row: four log-E maps | row colorbar | gap |
    # error map; one error colorbar spans all rows at the far right.
    map_columns = 4
    cbar_col = map_columns
    error_col = map_columns + 2
    error_cbar_col = map_columns + 3
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 4.3))
    grid = fig.add_gridspec(
        len(patients),
        map_columns + 4,
        width_ratios=(1.0,) * map_columns + (0.065, 0.42, 1.0, 0.065),
        left=0.04,
        right=0.935,
        top=0.85,
        bottom=0.105,
        wspace=0.07,
        hspace=0.10,
    )
    axes = np.asarray(
        [
            [
                fig.add_subplot(grid[row, col])
                for col in list(range(map_columns)) + [error_col]
            ]
            for row in range(len(patients))
        ]
    )
    error_mappable = None
    for row_index, (positions, maps) in enumerate(cases):
        low = min(values.min() for values in maps[:4])
        high = max(values.max() for values in maps[:4])
        log_mappable = None
        for column_index, values in enumerate(maps):
            is_error = column_index == 4
            scatter = axes[row_index, column_index].scatter(
                positions[:, 0],
                positions[:, 1],
                c=values,
                s=5,
                cmap="magma" if is_error else "viridis",
                vmin=0.0 if is_error else low,
                vmax=error_high if is_error else high,
                edgecolors="none",
            )
            if is_error:
                error_mappable = scatter
            else:
                log_mappable = scatter
            axes[row_index, column_index].set_aspect("equal")
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            axes[row_index, column_index].set_xlim(-0.04, 1.04)
            axes[row_index, column_index].set_ylim(-0.04, 1.04)
        row_cax = fig.add_subplot(grid[row_index, cbar_col])
        row_cbar = fig.colorbar(log_mappable, cax=row_cax)
        row_cbar.ax.tick_params(labelsize=7.2, length=2, pad=1.5)
        row_cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        axes[row_index, 0].set_ylabel(f"Geometry {row_index + 1}", fontsize=7.5)
        if row_index == 0:
            for column_index, title in enumerate(columns):
                axes[row_index, column_index].set_title(
                    title, fontsize=7.8, pad=3, linespacing=1.15
                )
            row_cax.set_title("log E", fontsize=7.5, pad=3)
    error_cax = fig.add_subplot(grid[:, error_cbar_col])
    error_cbar = fig.colorbar(error_mappable, cax=error_cax)
    error_cbar.set_label("|Δ log E| (shared scale)", fontsize=7.5)
    error_cbar.ax.tick_params(labelsize=7.2, length=2, pad=1.5)
    fig.text(
        0.04,
        0.012,
        "Each row's four log E maps share that row's colour scale; the error column shares one scale.\n"
        "* true-region/true-force reference under the shared 16-evaluation budget "
        "(diagnostic comparison only).",
        fontsize=7.3,
        color=GRAY,
        va="bottom",
        ha="left",
        linespacing=1.3,
    )
    fig.suptitle(
        "Synthetic material truth, held-out recovery, physics refinement, and reference",
        y=0.99,
        fontsize=10,
    )
    save_figure(fig, output_dir, "fig_ion_ct_physics_material_maps")


def fig_network_fem(output_dir: Path) -> None:
    """Draw MeshMaterialGNN via PlotNeuralNet (PlotNeuralNet checkout given by PLOTNEURALNET_DIR)."""
    script = Path(__file__).resolve().parent / "plot_meshmaterial_gnn.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing PlotNeuralNet script: {script}")
    # Ensure assets land in the requested output directory.
    import importlib.util

    spec = importlib.util.spec_from_file_location("plot_meshmaterial_gnn", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CURRENT_DIR = Path(output_dir).resolve()
    module.main()
    pdf_path = Path(output_dir) / "fig_network_fem.pdf"
    png_path = Path(output_dir) / "fig_network_fem.png"
    if not pdf_path.is_file() or not png_path.is_file():
        raise RuntimeError("PlotNeuralNet did not write fig_network_fem.pdf/png")


def fig_aiim_sensitivity(results_root: Path, output_dir: Path) -> None:
    """Noise x force-bias robustness and the no-screen counterfactual (R1/R2)."""
    sweep_dir = results_root / "aiim_sensitivity"
    summary = load_json(sweep_dir / "sensitivity_summary.json")
    reference = summary["reference"]
    configs = summary["configurations"]

    def series(prefix: str, ref_x: float) -> tuple[list[float], dict[str, list[float]]]:
        xs, ys = [ref_x], {
            "E": [100 * reference["E_background_median_relative_error"]],
            "ratio": [100 * reference["inclusion_ratio_median_relative_error"]],
            "accept": [100 * reference["refinement_acceptance_rate"]],
        }
        for name, row in sorted(
            ((n, c) for n, c in configs.items() if n.startswith(prefix)),
            key=lambda item: item[1]["spec"].get(
                "track_noise_px", item[1]["spec"].get("force_scale_error", 0.0)
            ),
        ):
            spec = row["spec"]
            xs.append(
                spec["track_noise_px"] if prefix == "noise" else spec["force_scale_error"]
            )
            ys["E"].append(100 * row["E_background_median_relative_error"])
            ys["ratio"].append(100 * row["inclusion_ratio_median_relative_error"])
            ys["accept"].append(100 * row["refinement_acceptance_rate"])
        order = np.argsort(xs)
        return [float(np.asarray(xs)[order][i]) for i in range(len(xs))], {
            key: [float(np.asarray(value)[order][i]) for i in range(len(value))]
            for key, value in ys.items()
        }

    noise_x, noise_y = series("noise", 0.05)
    force_x, force_y = series("force", 0.0)

    noscreen = load_json(sweep_dir / "sensitivity_noscreen.json")
    screened = load_json(
        results_root
        / "fem_multiview_screened50_strict"
        / "metrics_image_tracks_force_map_sigma_0.05_px_0.05_nodes_allframes_"
        "temporalreg_smooth2_multiview_conf0.20_regionweight8.0_"
        "frozen50_screen05_strict_gnn_init_predicted_region.json"
    )
    ref_records = {row["patient_id"]: row for row in screened["records"]}
    pairs = [
        (ref_records[row["patient_id"]], row)
        for row in noscreen["records"]
        if row["patient_id"] in ref_records
    ]

    # Height reduced (7.0 -> 6.2 in) so that the printed figure plus caption
    # no longer runs into the page footer. Panel C is split into two
    # side-by-side scatter plots with their own scales: ratio errors reach
    # ~100 % while background-E errors stay below ~25 %, and a shared axis
    # (or a log axis, which cannot show the exact-zero ratio errors) crushed
    # the background-E points.
    fig = plt.figure(figsize=(6.5, 6.2))
    grid = fig.add_gridspec(3, 1)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    bottom = grid[2].subgridspec(1, 2, wspace=0.30)
    ax_e, ax_r = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])
    for axis, xs, ys, xlabel, title, logx in (
        axes[0], noise_x, noise_y, "Track noise (px)", "A  Observation-noise robustness", True,
    ), (
        axes[1], force_x, force_y, "Force-calibration bias",
        "B  Force-prior misspecification", False,
    ):
        axis.plot(xs, ys["E"], "-o", color=BLUE, markersize=4, label=r"Background $E$")
        axis.plot(xs, ys["ratio"], "-s", color=ORANGE, markersize=4, label="Ratio")
        axis.plot(xs, ys["accept"], "--^", color=GRAY, markersize=4, label="Acceptance rate")
        axis.set_xlabel(xlabel)
        if logx:
            axis.set_xscale("log")
            axis.set_xticks(xs)
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_ylabel("Median error / rate (%)")
        axis.set_title(title, loc="left")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="center right", fontsize=7.5)

    e_ref = [100 * a["E_background_relative_error"] for a, _ in pairs]
    e_new = [100 * b["E_background_relative_error"] for _, b in pairs]
    r_ref = [100 * a["inclusion_ratio_relative_error"] for a, _ in pairs]
    r_new = [100 * b["inclusion_ratio_relative_error"] for _, b in pairs]
    for axis, ref, new, color, sub in (
        (ax_e, e_ref, e_new, BLUE, r"background $E$"),
        (ax_r, r_ref, r_new, ORANGE, "modulus ratio"),
    ):
        lim = max(ref + new) * 1.08
        axis.plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls=":")
        axis.scatter(ref, new, s=16, color=color, alpha=0.75)
        axis.set_xlim(-0.02 * lim, lim)
        axis.set_ylim(-0.02 * lim, lim)
        axis.set_xlabel(f"Screened MAP error (%), {sub}")
        axis.set_ylabel("Unscreened FEM error (%)")
        axis.grid(alpha=0.25)
    ax_e.set_title("C  Screen counterfactual (per case)", loc="left")
    nfev_ref = float(np.median([a["function_evaluations"] for a, _ in pairs]))
    nfev_new = float(np.median([b["function_evaluations"] for _, b in pairs]))
    worse = sum(1 for a, b in pairs if b["E_background_relative_error"]
                > a["E_background_relative_error"] + 1e-9)
    # Summary numbers sit in the title row of the right-hand sub-panel, i.e.
    # outside both plotted regions (two short lines so that they stay within
    # that sub-panel's width).
    ax_r.set_title(
        f"median evals {nfev_ref:.0f} → {nfev_new:.0f}\n"
        f"unscreened worse ($E$): {worse}/{len(pairs)} cases",
        loc="right",
        fontsize=7.5,
        fontweight="normal",
        linespacing=1.2,
    )
    fig.tight_layout(h_pad=1.2)
    save_figure(fig, output_dir, "fig_aiim_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing the frozen lung_mesh_ai_v2 JSON artifacts.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing manifest.json and patient gt.pt tensors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination for PDF and PNG assets.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="STEM",
        help="Regenerate only the listed figure stems (e.g. fig_fair_fem).",
    )
    args = parser.parse_args()
    configure_style()
    payload = load_sources(args.results_root.resolve())
    dataset_root = args.dataset_root.resolve()
    manifest = load_json(dataset_root / "manifest.json")
    results_root = args.results_root.resolve()
    builders: dict[str, Any] = {
        "fig_pipeline": lambda: fig_pipeline(args.output_dir),
        "fig_network_fem": lambda: fig_network_fem(args.output_dir),
        "fig_dataset_protocol": lambda: fig_dataset_protocol(manifest, args.output_dir),
        "fig_qualitative_case": lambda: fig_qualitative_case(
            manifest,
            payload["calibration"],
            payload["fair_fem"],
            dataset_root,
            args.output_dir,
        ),
        "fig_case_gallery": lambda: fig_case_gallery(
            manifest, payload["calibration"], dataset_root, args.output_dir
        ),
        "fig_development_ablation": lambda: fig_development_ablation(results_root, args.output_dir),
        "fig_main_results": lambda: fig_main_results(payload, args.output_dir),
        "fig_response_baselines": lambda: fig_response_baselines(payload, args.output_dir),
        "fig_fair_fem": lambda: fig_fair_fem(payload, args.output_dir),
        "fig_calibration": lambda: fig_calibration(payload, args.output_dir),
        "fig_real_ion_cases": lambda: fig_real_ion_cases(payload, args.output_dir),
        "fig_multiseed_ood": lambda: fig_multiseed_ood(payload, args.output_dir),
        "fig_ion_ct_synthetic_pipeline": lambda: fig_ion_ct_synthetic_pipeline(args.output_dir),
        "fig_ion_ct_material_recovery": lambda: fig_ion_ct_material_recovery(args.output_dir),
        "fig_ion_ct_method_comparison": lambda: fig_ion_ct_method_comparison(args.output_dir),
        "fig_ion_ct_error_distributions": lambda: fig_ion_ct_error_distributions(args.output_dir),
        "fig_ion_ct_node_material_metrics": lambda: fig_ion_ct_node_material_metrics(args.output_dir),
        "fig_ion_ct_physics_tradeoff": lambda: fig_ion_ct_physics_tradeoff(args.output_dir),
        "fig_ion_ct_multiview_tracks": lambda: fig_ion_ct_multiview_tracks(args.output_dir),
        "fig_ion_ct_physics_material_maps": lambda: fig_ion_ct_physics_material_maps(args.output_dir),
        "fig_aiim_sensitivity": lambda: fig_aiim_sensitivity(results_root, args.output_dir),
    }
    stems = tuple(builders)
    if args.only:
        unknown = [stem for stem in args.only if stem not in builders]
        if unknown:
            raise SystemExit(f"Unknown figure stems: {unknown}; choose from {list(builders)}")
        stems = tuple(stem for stem in builders if stem in set(args.only))
    for stem in stems:
        print(f"generating {stem} ...")
        builders[stem]()
    verify_assets(args.output_dir, stems)
    print(f"Generated {2 * len(stems)} assets in {args.output_dir.resolve()}")
    for stem in stems:
        print(f"  {stem}.pdf")
        print(f"  {stem}.png")


if __name__ == "__main__":
    main()
