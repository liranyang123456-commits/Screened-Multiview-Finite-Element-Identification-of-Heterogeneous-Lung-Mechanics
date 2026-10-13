#!/usr/bin/env python3
"""Generate the Array/Elsevier graphical abstract for the revised manuscript.

The abstract merges the Figure 1 framework illustration with the main
results: title, the six illustrated pipeline stages of Figure 1 (drawn by
``generate_aiim_figures.draw_pipeline_stages`` so that Figure 1 and the
abstract share one drawing), the prior-matched sequential EKF side path, two
result cards and one takeaway line. The canvas is drawn at its display size
(13 cm wide, the Elsevier guide width) so that the point sizes written here
are the point sizes a reader sees; it is rasterised at ~519 dpi to a
2656 px wide upload. Every number in the results row is read from the frozen
JSON artifacts listed in ``SOURCES`` and cross-checked against the manuscript
values; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_aiim_figures import draw_pipeline_stages  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ATTACHMENTS = ROOT / "figures"
# The EKF bars show the prior-matched run: material-prior width multiplier 1,
# the width used by the CT120 official-prior audit and by the frozen primary
# synthetic screened-MAP run (the width-2 runs, 3.12 % / 2.68 %, are the
# sensitivity reported in Supplement Sec. 9.5). Switching EKF_PRIOR_WIDTH to
# 2.0 selects the width-2 directories and their manuscript values instead.
EKF_PRIOR_WIDTH = 1.0
EKF_DIRS = {1.0: ("ekf_sequential_synthetic50_w1", "ekf_sequential_ct120_all_w1"), 2.0: ("ekf_sequential_synthetic50", "ekf_sequential_ct120_all")}
SOURCES = {
    "calibrator": ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json",
    "ekf_synthetic": ROOT / "results" / "lung_mesh_ai_v2" / EKF_DIRS[EKF_PRIOR_WIDTH][0] / "summary.json",
    "audit_ct": ROOT / "results" / "ion_ct_official_prior_screened_fem120_v1" / "summary.json",
    "ekf_ct": ROOT / "results" / "ct_reviewer_drivers_v6" / EKF_DIRS[EKF_PRIOR_WIDTH][1] / "summary.json",
}
# Manuscript values (percent) that the JSON-derived numbers must reproduce.
EXPECTED_EKF = {1.0: {"synthetic": 2.73, "ct": 4.74}, 2.0: {"synthetic": 2.68, "ct": 3.12}}
EXPECTED = {
    "synthetic": {"initializer": 3.82, "audit": 4.11, "ekf": EXPECTED_EKF[EKF_PRIOR_WIDTH]["synthetic"]},
    "ct": {"initializer": 8.55, "audit": 11.70, "ekf": EXPECTED_EKF[EKF_PRIOR_WIDTH]["ct"]},
}

TITLE = (
    "Calibrated Multiview Inverse Finite-Element Identification of "
    "Heterogeneous Lung Mechanics with Screened Physics Auditing"
)

INK = "#17212B"
GRAY = "#5B6573"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#6F4E7C"
CYAN = "#0098A6"
PALE = {
    "gray": "#F4F6F8",
    "blue": "#E8F2F8",
    "cyan": "#E6F6F7",
    "green": "#E6F4EF",
    "orange": "#FAEDE8",
    "purple": "#F0EBF3",
}

# Display size of the abstract (13 x 6.9 cm) and the pixel target. Elsevier
# asks for >= 531 x 1328 px (h x w) legible at 5 x 13 cm; the extra height
# carries the illustrated pipeline row.
WIDTH_IN, HEIGHT_IN = 13.0 / 2.54, 6.9 / 2.54
PIXEL_WIDTH = 2656
DPI = PIXEL_WIDTH / WIDTH_IN

# Point sizes at the 13 cm display width (the figure is drawn at that width,
# so fontsize_pt * 13 cm / figure_width == fontsize_pt). Minimum 6 pt.
FONT = {"title": 8.6, "box_title": 6.2, "box_detail": 6.0, "icon_label": 6.0, "strip": 6.0, "note": 6.0}
MIN_FONT_PT = min(FONT.values())

# Pipeline row: the Figure 1 boxes are drawn on an inset whose data unit is
# PIPELINE_SCALE display inches, so the icon geometry shrinks uniformly while
# fonts, marker areas and line widths are set explicitly for the small size.
PIPELINE_SCALE = 0.46
GA_STAGES = (
    ("1  Geometry &\nconstraints", "mesh, boundary,\ncontacts", GRAY, PALE["gray"]),
    ("2  Multiview\nmotion", "4 loads, 3 views\n7 frames + force", BLUE, PALE["blue"]),
    ("3  Response\nencoding", "visibility mask\nresponse tokens", BLUE, "#F2F7FB"),
    ("4  Learned\ninitializer", "PCA–Ridge or\nMeshGNN hybrid", GREEN, PALE["green"]),
    ("5  Physics\ndecision", "7 first-round\nevaluations", ORANGE, PALE["orange"]),
    ("6  Audited\noutput", "accept / retain\nfield + interval", PURPLE, PALE["purple"]),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _check_ekf_prior_width(ekf_summary: dict, batch_arm: str, cohort: str) -> None:
    """The EKF run must carry the selected prior width in every record and, for
    width 1, reproduce the batch arm's material-prior SD scene by scene."""
    if ekf_summary.get("prior_width_scale") is None:
        if EKF_PRIOR_WIDTH != 2.0:
            raise ValueError(f"{cohort}: EKF summary predates prior-width bookkeeping (legacy width-2 run)")
        return
    width = float(ekf_summary["prior_width_scale"])
    in_records = [float(value) for value in ekf_summary["prior_width_scale_in_records"]]
    if width != EKF_PRIOR_WIDTH or in_records != [EKF_PRIOR_WIDTH]:
        raise ValueError(f"{cohort}: EKF summary reports prior width {width} / {in_records}, expected {EKF_PRIOR_WIDTH}")
    if ekf_summary["prior_width_scale_recorded_per_record"] != ekf_summary["n"]:
        raise ValueError(f"{cohort}: prior width not recorded in every EKF record")
    check = ekf_summary["paired_vs_batch"][batch_arm]["matched_protocol_checks"]["prior_width"]
    if EKF_PRIOR_WIDTH == 1.0 and not (check["width_scale_equal"] and check["prior_sd_all_equal"]):
        raise ValueError(f"{cohort}: width-1 EKF prior SD does not match the batch arm {batch_arm}: {check}")


def load_numbers() -> dict[str, dict[str, float]]:
    """Median background-modulus errors (percent) from the frozen artifacts."""
    calibrator = _load(SOURCES["calibrator"])
    ekf_synthetic = _load(SOURCES["ekf_synthetic"])
    audit_ct = _load(SOURCES["audit_ct"])
    ekf_ct = _load(SOURCES["ekf_ct"])

    synthetic = {
        "initializer": 100.0 * calibrator["test"]["E_background_median_relative_error"],
        "audit": ekf_synthetic["paired_vs_batch"]["frozen_primary_screened_map_locked50_v1"][
            "endpoints"
        ]["E_background_relative_error"]["batch_median_percent"],
        "ekf": ekf_synthetic["median_background_error_percent"],
    }
    ct = {
        "initializer": 100.0 * audit_ct["reference_official_hybrid_ct"]["E_background_median_relative_error"],
        "audit": 100.0 * audit_ct["screened_fem_locked"]["E_background_median_relative_error"]["median"],
        "ekf": ekf_ct["median_background_error_percent"],
    }
    if calibrator["test_patient_count"] != 50 or ekf_synthetic["n"] != 50:
        raise ValueError("Synthetic sources do not describe the 50 held-out cases")
    if audit_ct["screened_fem_locked"]["n"] != 120 or ekf_ct["n"] != 120:
        raise ValueError("CT sources do not describe the 120-scene cohort")
    _check_ekf_prior_width(ekf_ct, "official_prior_screened_fem120_v1", "ct")
    _check_ekf_prior_width(ekf_synthetic, "frozen_primary_screened_map_locked50_v1", "synthetic")
    numbers = {"synthetic": synthetic, "ct": ct}
    for cohort, values in numbers.items():
        for key, value in values.items():
            if abs(round(value, 2) - EXPECTED[cohort][key]) > 0.005:
                raise ValueError(f"{cohort}/{key}: JSON gives {value:.4f}%, manuscript states {EXPECTED[cohort][key]}%")
    return numbers


def _fit_lines(fig: plt.Figure, text: str, fontsize: float, weight: str, max_width_pt: float) -> str:
    """Greedy word wrap using measured glyph widths (not character counts)."""
    renderer = fig.canvas.get_renderer()
    prop = matplotlib.font_manager.FontProperties(size=fontsize, weight=weight)

    def width_pt(fragment: str) -> float:
        width, _height, _descent = renderer.get_text_width_height_descent(fragment, prop, ismath=False)
        return width * 72.0 / fig.dpi

    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and width_pt(candidate) > max_width_pt:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    for line in lines:
        if width_pt(line) > max_width_pt:
            raise RuntimeError(f"Word too wide for its box at {fontsize} pt: {line!r}")
    return "\n".join(lines)


def _bbox_inside(inner: matplotlib.transforms.Bbox, outer: matplotlib.transforms.Bbox, tol: float = 0.5) -> bool:
    return (
        inner.x0 >= outer.x0 - tol
        and inner.x1 <= outer.x1 + tol
        and inner.y0 >= outer.y0 - tol
        and inner.y1 <= outer.y1 + tol
    )


def draw_abstract(numbers: dict[str, dict[str, float]]) -> tuple[plt.Figure, float]:
    """Lay out the abstract in display inches: title / pipeline / results / takeaway.

    Returns the figure and the smallest font size used (points at 13 cm)."""
    W, H = WIDTH_IN, HEIGHT_IN
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.canvas.draw()  # renderer needed for measured wrapping
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    width_pt = W * 72.0
    margin_x = 0.015

    # ---- vertical layout (inches from the bottom) ------------------------
    note_y = 0.05  # takeaway baseline (va=bottom)
    card_y0, card_h = 0.41, 0.66
    ekf_y = card_y0 + card_h + 0.115  # EKF side-path shaft and label centre
    box_y0, box_h = ekf_y + 0.115, 0.98
    title_y = H - 0.05  # title top (va=top)

    # ---- row 1: title ------------------------------------------------------
    title = ax.text(
        W / 2,
        title_y,
        _fit_lines(fig, TITLE, FONT["title"], "bold", 0.94 * width_pt),
        ha="center",
        va="top",
        fontsize=FONT["title"],
        fontweight="bold",
        color=INK,
        linespacing=1.15,
    )

    # ---- row 2: the six Figure 1 stage boxes --------------------------------
    n = len(GA_STAGES)
    gap = 0.09
    box_w = (W - 2 * margin_x - (n - 1) * gap) / n
    lefts = [margin_x + i * (box_w + gap) for i in range(n)]
    s = PIPELINE_SCALE
    slack = 0.06  # inset margin so box borders are not clipped
    inset_h = box_h + 2 * slack
    pipeline_ax = ax.inset_axes([0.0, box_y0 - slack, W, inset_h], transform=ax.transData)
    pipeline_ax.set_xlim(0, W / s)
    pipeline_ax.set_ylim(0, inset_h / s)
    pipeline_ax.axis("off")
    pipeline_ax.patch.set_alpha(0.0)
    boxes = draw_pipeline_stages(
        pipeline_ax,
        [(left / s, slack / s) for left in lefts],
        stages=GA_STAGES,
        box_w=box_w / s,
        box_h=box_h / s,
        icon_dy=0.50 / s,
        title_dy=(0.09 / s, 0.135 / s),
        detail_dy=0.165 / s,
        title_pt=FONT["box_title"],
        detail_pt=FONT["box_detail"],
        label_pt=FONT["icon_label"],
        check_pt=8.0,
        box_lw=1.0,
        rounding_size=0.05 / s,
        marker_scale=0.26,
        lw_scale=0.75,
        arrow_scale=4.0,
        decision_area=170.0,
        decision_arm=(0.17, 0.12),
        fem_lattice=False,
    )
    mid_y = box_y0 + box_h / 2
    for i in range(n - 1):
        ax.add_patch(
            FancyArrowPatch(
                (lefts[i] + box_w + 0.008, mid_y),
                (lefts[i + 1] - 0.008, mid_y),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.9,
                color=INK,
                shrinkA=0,
                shrinkB=0,
            )
        )
    # EKF side path: from the learned initializer (stage 4) past the physics
    # decision to the output row, drawn in the EKF colour.
    x_start = lefts[3] + box_w / 2
    x_end = lefts[5] + box_w / 2
    ax.plot([x_start, x_start], [box_y0, ekf_y], color=PURPLE, lw=0.9, ls=(0, (2.5, 1.5)), solid_capstyle="butt")
    ax.add_patch(
        FancyArrowPatch(
            (x_start, ekf_y),
            (x_end, ekf_y),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.9,
            color=PURPLE,
            linestyle=(0, (2.5, 1.5)),
            shrinkA=0,
            shrinkB=0,
        )
    )
    ekf_label = ax.text(
        x_start - 0.08,
        ekf_y,
        "Prior-matched sequential EKF: all-frame updates, no screening",
        ha="right",
        va="center",
        fontsize=FONT["icon_label"],
        color=PURPLE,
    )

    # ---- row 3: result cards ------------------------------------------------
    card_gap = 0.12
    card_w = (W - 2 * margin_x - card_gap) / 2
    cards = [
        ("Synthetic: 50 held-out cases", numbers["synthetic"], margin_x),
        ("CT-derived: 120 scenes, 6 held-out geometries", numbers["ct"], margin_x + card_w + card_gap),
    ]
    ekf_bar_label = "Prior-matched EKF" if EKF_PRIOR_WIDTH == 1.0 else "Sequential EKF (w2)"
    rows = [("Learned initializer", "initializer", GREEN), ("Screened FEM audit", "audit", ORANGE), (ekf_bar_label, "ekf", PURPLE)]
    card_patches: list[tuple[FancyBboxPatch, list]] = []
    for label, values, left in cards:
        card = FancyBboxPatch(
            (left, card_y0),
            card_w,
            card_h,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor="white",
            edgecolor="#C9D1D9",
            linewidth=0.8,
        )
        ax.add_patch(card)
        head = ax.text(
            left + 0.07,
            card_y0 + card_h - 0.055,
            label,
            ha="left",
            va="top",
            fontsize=FONT["strip"],
            fontweight="bold",
            color=INK,
        )
        # Horizontal bars: median background-modulus error (%), shared scale.
        inset = ax.inset_axes([left + 0.95, card_y0 + 0.07, 1.22, card_h - 0.24], transform=ax.transData)
        y = [2, 1, 0]
        vals = [values[key] for _, key, _ in rows]
        inset.barh(y, vals, color=[c for _, _, c in rows], height=0.62)
        value_texts = []
        for yi, val in zip(y, vals):
            value_texts.append(
                inset.text(val + 0.3, yi, f"{val:.2f}%", va="center", ha="left", fontsize=FONT["strip"], color=INK)
            )
        inset.set_yticks(y, [name for name, _, _ in rows], fontsize=FONT["strip"])
        inset.set_xlim(0, 16.5)
        inset.set_ylim(-0.55, 2.55)
        inset.set_xticks([])
        inset.tick_params(axis="y", length=0, pad=2)
        for spine in ("top", "right", "bottom"):
            inset.spines[spine].set_visible(False)
        inset.spines["left"].set_color("#C9D1D9")
        inset.set_facecolor("none")
        card_patches.append((card, [head, *value_texts, *inset.get_yticklabels()]))
    bars_caption = ax.text(
        W / 2,
        card_y0 - 0.025,
        "Bars: median background-modulus error (%), lower is better",
        ha="center",
        va="top",
        fontsize=FONT["note"],
        color=GRAY,
    )

    # ---- row 4: takeaway ----------------------------------------------------
    ct = numbers["ct"]
    note = (
        "Screening controls FEM cost but does not detect model mismatch: on CT-derived geometry the audit raises "
        f"the median error ({ct['initializer']:.2f}% → {ct['audit']:.2f}%); the prior-matched sequential EKF "
        f"({ct['ekf']:.2f}%) carries the identification evidence."
    )
    note_lines = _fit_lines(fig, note, FONT["note"], "normal", 0.96 * width_pt)
    if note_lines.count("\n") > 1:
        raise RuntimeError("Takeaway wraps to more than two lines; shorten it")
    note_text = ax.text(
        W / 2,
        note_y,
        note_lines,
        ha="center",
        va="bottom",
        fontsize=FONT["note"],
        color=ORANGE,
        linespacing=1.15,
    )

    # ---- self-checks --------------------------------------------------------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # (a) every box/card text stays inside its box and the texts of one box do not collide
    for patch, texts in boxes + card_patches:
        box_bbox = patch.get_window_extent(renderer)
        extents = [text.get_window_extent(renderer) for text in texts]
        for text, text_bbox in zip(texts, extents):
            if not _bbox_inside(text_bbox, box_bbox):
                raise RuntimeError(f"Text overflows its box: {text.get_text()!r}")
        for first in range(len(extents)):
            for second in range(first + 1, len(extents)):
                if extents[first].overlaps(extents[second]):
                    raise RuntimeError(
                        f"Texts overlap inside a box: {texts[first].get_text()!r} / {texts[second].get_text()!r}"
                    )
    # (b) free-standing lines must not touch the boxes, the cards or each other
    free_lines = [title, ekf_label, bars_caption, note_text]
    free_extents = [text.get_window_extent(renderer) for text in free_lines]
    for first in range(len(free_lines)):
        for second in range(first + 1, len(free_lines)):
            if free_extents[first].overlaps(free_extents[second]):
                raise RuntimeError(f"Lines overlap: {free_lines[first].get_text()!r} / {free_lines[second].get_text()!r}")
    for patch, _texts in boxes + card_patches:
        patch_bbox = patch.get_window_extent(renderer)
        for text, extent in zip(free_lines, free_extents):
            if extent.overlaps(patch_bbox):
                raise RuntimeError(f"Text overlaps a box or card: {text.get_text()!r}")
    # the EKF label must end before the EKF shaft starts
    if ekf_label.get_window_extent(renderer).x1 > ax.transData.transform((x_start, ekf_y))[0]:
        raise RuntimeError("EKF label runs into the EKF arrow")
    # (c) no two visible texts anywhere may overlap, and none may leave the canvas
    all_texts = [
        text for text in fig.findobj(matplotlib.text.Text) if text.get_text().strip() and text.get_visible()
    ]
    all_extents = [text.get_window_extent(renderer) for text in all_texts]
    fig_bbox = fig.bbox
    for text, extent in zip(all_texts, all_extents):
        if extent.x0 < -0.5 or extent.x1 > fig_bbox.width + 0.5 or extent.y0 < -0.5 or extent.y1 > fig_bbox.height + 0.5:
            raise RuntimeError(f"Text leaves the canvas: {text.get_text()!r}")
    for first in range(len(all_texts)):
        for second in range(first + 1, len(all_texts)):
            if all_extents[first].overlaps(all_extents[second]):
                raise RuntimeError(f"Texts overlap: {all_texts[first].get_text()!r} / {all_texts[second].get_text()!r}")
    # (d) icon labels (prior / MAP / dJ / check mark) stay inside their stage box
    stage_bboxes = [patch.get_window_extent(renderer) for patch, _ in boxes]
    for text, extent in zip(all_texts, all_extents):
        if text.axes is pipeline_ax and not any(_bbox_inside(extent, bbox) for bbox in stage_bboxes):
            raise RuntimeError(f"Pipeline text outside every stage box: {text.get_text()!r}")
    smallest = min(text.get_fontsize() for text in all_texts)
    if smallest < MIN_FONT_PT - 1e-9:
        raise RuntimeError(f"Font below {MIN_FONT_PT} pt: {smallest}")
    return fig, smallest


def main() -> None:
    numbers = load_numbers()
    fig, min_font = draw_abstract(numbers)
    ATTACHMENTS.mkdir(parents=True, exist_ok=True)
    png = ATTACHMENTS / "graphical_abstract_array.png"
    fig.savefig(png, dpi=DPI, facecolor="white", pad_inches=0)
    fig.savefig(png.with_suffix(".pdf"), facecolor="white", pad_inches=0)
    plt.close(fig)
    from PIL import Image

    with Image.open(png) as image:
        display_scale = 13.0 / (WIDTH_IN * 2.54)
        print(
            f"wrote {png} ({image.width} x {image.height} px, {WIDTH_IN * 2.54:.1f} x {HEIGHT_IN * 2.54:.1f} cm, "
            f"min font {min_font * display_scale:.2f} pt at 13 cm width)"
        )
    print(f"wrote {png.with_suffix('.pdf')}")
    for cohort, values in numbers.items():
        print(cohort, {key: round(value, 2) for key, value in values.items()})


if __name__ == "__main__":
    main()
