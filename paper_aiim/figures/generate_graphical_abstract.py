#!/usr/bin/env python3
"""Generate the Array/Elsevier graphical abstract for the revised manuscript.

The banner is drawn at its display size (13 x 5.2 cm, the Elsevier guide
size) so that the point sizes written here are the point sizes a reader sees;
it is rasterised at ~519 dpi to the 2656 x 1062 px upload. Every number in
the results strip is read from the frozen JSON artifacts listed in
``SOURCES`` and cross-checked against the manuscript values; nothing is typed
in by hand.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

# Display size of the banner (13 x 5.2 cm) and the pixel target.
WIDTH_IN, HEIGHT_IN = 13.0 / 2.54, 5.2 / 2.54
PIXEL_WIDTH = 2656
DPI = PIXEL_WIDTH / WIDTH_IN

FONT = {"title": 8.6, "box_title": 6.2, "box_detail": 6.0, "strip": 6.0, "note": 6.0}
MIN_FONT_PT = min(FONT.values())


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


def draw_abstract(numbers: dict[str, dict[str, float]]) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
    fig.canvas.draw()  # renderer needed for measured wrapping
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    width_pt = WIDTH_IN * 72.0

    # ---- title ---------------------------------------------------------
    ax.text(
        0.5,
        0.97,
        _fit_lines(fig, TITLE, FONT["title"], "bold", 0.94 * width_pt),
        ha="center",
        va="top",
        fontsize=FONT["title"],
        fontweight="bold",
        color=INK,
        linespacing=1.15,
    )

    # ---- pipeline -------------------------------------------------------
    stages = [
        ("Synthetic / CT-derived geometry", "implanted two-level stiffness field", GRAY, PALE["gray"]),
        ("Calibrated multiview tracks", "4 loads × 3 views × 7 frames; force and camera calibration", BLUE, PALE["blue"]),
        ("Learned initializer", "PCA–Ridge (synthetic) / hybrid MeshGNN (CT); conformal intervals", GREEN, PALE["green"]),
        ("Screened FEM audit", "accept if the development-fixed objective improves, else retain", ORANGE, PALE["orange"]),
        ("Sequential EKF diagnostic", "all-frame updates of E_bg and q; prior matched to the audit; no screening", PURPLE, PALE["purple"]),
    ]
    n = len(stages)
    x_margin, gap = 0.015, 0.022
    box_w = (1.0 - 2 * x_margin - (n - 1) * gap) / n
    box_y0, box_h = 0.50, 0.33
    inner_pt = (box_w - 0.012) * width_pt
    boxes: list[tuple[FancyBboxPatch, list]] = []
    for i, (title, detail, color, fill) in enumerate(stages):
        left = x_margin + i * (box_w + gap)
        box = FancyBboxPatch(
            (left, box_y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.1,
        )
        ax.add_patch(box)
        texts = [
            ax.text(
                left + box_w / 2,
                box_y0 + box_h - 0.03,
                _fit_lines(fig, title, FONT["box_title"], "bold", inner_pt),
                ha="center",
                va="top",
                fontsize=FONT["box_title"],
                fontweight="bold",
                color=INK,
                linespacing=1.1,
            ),
            ax.text(
                left + box_w / 2,
                box_y0 + 0.03,
                _fit_lines(fig, detail, FONT["box_detail"], "normal", inner_pt),
                ha="center",
                va="bottom",
                fontsize=FONT["box_detail"],
                color=GRAY,
                linespacing=1.1,
            ),
        ]
        boxes.append((box, texts))
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (left + box_w + 0.003, box_y0 + box_h / 2),
                    (left + box_w + gap - 0.003, box_y0 + box_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=6,
                    linewidth=0.8,
                    color=INK,
                )
            )

    # ---- results strip --------------------------------------------------
    strip_y0, strip_h = 0.165, 0.29
    cards = [
        ("Synthetic: 50 held-out cases", numbers["synthetic"], 0.025, 0.46),
        ("CT-derived geometry: 120 held-out scenes", numbers["ct"], 0.515, 0.46),
    ]
    ekf_label = "Prior-matched EKF" if EKF_PRIOR_WIDTH == 1.0 else "Sequential EKF (w2)"
    rows = [("Learned initializer", "initializer", GREEN), ("Screened FEM audit", "audit", ORANGE), (ekf_label, "ekf", PURPLE)]
    card_patches = []
    for label, values, left, width in cards:
        card = FancyBboxPatch(
            (left, strip_y0),
            width,
            strip_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="white",
            edgecolor="#C9D1D9",
            linewidth=0.8,
        )
        ax.add_patch(card)
        head = ax.text(
            left + 0.012,
            strip_y0 + strip_h - 0.03,
            label,
            ha="left",
            va="top",
            fontsize=FONT["strip"],
            fontweight="bold",
            color=INK,
        )
        card_patches.append((card, [head]))
        # Horizontal bars: median background-modulus error (%), shared scale.
        inset = ax.inset_axes([left + 0.19, strip_y0 + 0.035, width - 0.26, strip_h - 0.11], transform=ax.transData)
        y = [2, 1, 0]
        vals = [values[key] for _, key, _ in rows]
        inset.barh(y, vals, color=[c for _, _, c in rows], height=0.62)
        for yi, val in zip(y, vals):
            inset.text(val + 0.3, yi, f"{val:.2f}%", va="center", ha="left", fontsize=FONT["strip"], color=INK)
        inset.set_yticks(y, [name for name, _, _ in rows], fontsize=FONT["strip"])
        inset.set_xlim(0, 16.5)
        inset.set_xticks([])
        inset.tick_params(axis="y", length=0, pad=2)
        for spine in ("top", "right", "bottom"):
            inset.spines[spine].set_visible(False)
        inset.spines["left"].set_color("#C9D1D9")
        inset.set_facecolor("none")
    bars_caption = ax.text(
        0.5,
        strip_y0 - 0.015,
        "Bars: median background-modulus error (%), lower is better",
        ha="center",
        va="top",
        fontsize=FONT["note"],
        color=GRAY,
    )

    # ---- note -----------------------------------------------------------
    ct = numbers["ct"]
    note = (
        "The accept/retain screen controls cost but does not detect model mismatch: on CT-derived geometry the audit "
        f"raises the median error ({ct['initializer']:.2f}% → {ct['audit']:.2f}%), whereas the prior-matched "
        f"sequential EKF reaches {ct['ekf']:.2f}%."
    )
    note_text = ax.text(
        0.5,
        0.015,
        _fit_lines(fig, note, FONT["note"], "normal", 0.96 * width_pt),
        ha="center",
        va="bottom",
        fontsize=FONT["note"],
        color=ORANGE,
        linespacing=1.15,
    )

    # ---- self-check: no text may leave its box --------------------------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, texts in boxes + card_patches:
        box_bbox = patch.get_window_extent(renderer)
        extents = [text.get_window_extent(renderer) for text in texts]
        for text, text_bbox in zip(texts, extents):
            if not (
                text_bbox.x0 >= box_bbox.x0 - 0.5
                and text_bbox.x1 <= box_bbox.x1 + 0.5
                and text_bbox.y0 >= box_bbox.y0 - 0.5
                and text_bbox.y1 <= box_bbox.y1 + 0.5
            ):
                raise RuntimeError(f"Text overflows its box: {text.get_text()!r}")
        for first in range(len(extents)):
            for second in range(first + 1, len(extents)):
                if extents[first].overlaps(extents[second]):
                    raise RuntimeError(
                        f"Texts overlap inside a box: {texts[first].get_text()!r} / {texts[second].get_text()!r}"
                    )
    # the free-standing lines below the cards must not run into each other or the cards
    free_lines = [bars_caption, note_text]
    free_extents = [text.get_window_extent(renderer) for text in free_lines]
    if free_extents[0].overlaps(free_extents[1]):
        raise RuntimeError("Note overlaps the bar caption; shorten the note")
    for card, _texts in card_patches:
        for text, extent in zip(free_lines, free_extents):
            if extent.overlaps(card.get_window_extent(renderer)):
                raise RuntimeError(f"Text overlaps a results card: {text.get_text()!r}")
    fig_bbox = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_text().strip() or not text.get_visible():
            continue
        extent = text.get_window_extent(renderer)
        if extent.x0 < -0.5 or extent.x1 > fig_bbox.width + 0.5 or extent.y0 < -0.5 or extent.y1 > fig_bbox.height + 0.5:
            raise RuntimeError(f"Text leaves the canvas: {text.get_text()!r}")
    return fig


def main() -> None:
    numbers = load_numbers()
    fig = draw_abstract(numbers)
    ATTACHMENTS.mkdir(parents=True, exist_ok=True)
    png = ATTACHMENTS / "graphical_abstract_array.png"
    fig.savefig(png, dpi=DPI, facecolor="white", pad_inches=0)
    fig.savefig(png.with_suffix(".pdf"), facecolor="white", pad_inches=0)
    plt.close(fig)
    from PIL import Image

    with Image.open(png) as image:
        print(f"wrote {png} ({image.width} x {image.height} px, min font {MIN_FONT_PT} pt at 13 cm width)")
    print(f"wrote {png.with_suffix('.pdf')}")
    for cohort, values in numbers.items():
        print(cohort, {key: round(value, 2) for key, value in values.items()})


if __name__ == "__main__":
    main()
