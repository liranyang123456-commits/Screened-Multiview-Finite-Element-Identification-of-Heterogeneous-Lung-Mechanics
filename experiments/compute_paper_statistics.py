"""Compute auditable scene-level statistics used by the TBME manuscript.

Only checked-in per-scene Markdown tables are analyzed. The 40-scene
multi-anatomy and 24-scene C3VD-derived summaries do not contain per-scene
records, so this script deliberately does not manufacture confidence intervals
for those experiments.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 10_000


def markdown_tables(path: Path) -> list[tuple[list[str], list[dict[str, str]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tables: list[tuple[list[str], list[dict[str, str]]]] = []
    index = 0
    while index + 1 < len(lines):
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        if (
            header.startswith("|")
            and separator.startswith("|")
            and re.fullmatch(r"\|?[\s:|-]+\|?", separator)
        ):
            columns = [cell.strip() for cell in header.strip("|").split("|")]
            rows: list[dict[str, str]] = []
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                values = [
                    cell.strip()
                    for cell in lines[index].strip().strip("|").split("|")
                ]
                if len(values) == len(columns):
                    rows.append(dict(zip(columns, values)))
                index += 1
            tables.append((columns, rows))
            continue
        index += 1
    return tables


def table_with_column(path: Path, column: str) -> list[dict[str, str]]:
    matches = [rows for columns, rows in markdown_tables(path) if column in columns]
    if len(matches) != 1:
        raise ValueError(f"Expected one table containing {column!r} in {path}")
    return matches[0]


def number(value: str) -> float:
    return float(value.replace("%", "").replace("**", "").strip())


def bootstrap_ci(
    values: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    samples = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True)
    if statistic == "mean":
        estimates = samples.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def exact_paired_permutation_pvalue(differences: np.ndarray) -> float:
    """Two-sided exact sign-flip test for paired differences."""
    observed = abs(float(differences.mean()))
    means: list[float] = []
    for mask in range(1 << len(differences)):
        signs = np.array(
            [1.0 if mask & (1 << i) else -1.0 for i in range(len(differences))]
        )
        means.append(abs(float((differences * signs).mean())))
    return float(np.mean(np.asarray(means) >= observed - 1e-12))


def fmt_ci(interval: tuple[float, float], decimals: int = 1) -> str:
    return f"[{interval[0]:.{decimals}f}, {interval[1]:.{decimals}f}]"


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sim_rows = table_with_column(RESULTS / "sim_eval.md", "scene")
    baseline_rows = table_with_column(RESULTS / "baseline_cmp.md", "E_gt")

    metric_columns = {
        "E relative error (%)": "E_rel%",
        "Albedo relative error (%)": "alb_rel%",
        "Roughness relative error (%)": "rough_rel%",
        "PSNR (dB)": "PSNR",
        "SSIM": "SSIM",
    }

    summary_lines = [
        "# Auditable statistical summary",
        "",
        (
            f"Bootstrap confidence intervals use {N_BOOTSTRAP:,} scene-level "
            f"resamples with seed {BOOTSTRAP_SEED}."
        ),
        "",
        "## Sixty-scene stiffness sweep",
        "",
        "| Metric | Mean ± SD | Median | 95% bootstrap CI of median |",
        "|---|---:|---:|---:|",
    ]

    for label, column in metric_columns.items():
        values = np.asarray([number(row[column]) for row in sim_rows], dtype=float)
        ci = bootstrap_ci(values, "median", rng)
        decimals = 3 if column == "SSIM" else 1
        summary_lines.append(
            f"| {label} | {values.mean():.3f} ± {values.std(ddof=1):.3f} | "
            f"{np.median(values):.3f} | {fmt_ci(ci, decimals)} |"
        )

    grouped: dict[float, list[float]] = defaultdict(list)
    for row in sim_rows:
        grouped[number(row["scene"]) / 1000.0].append(number(row["E_rel%"]))

    summary_lines.extend(
        [
            "",
            "### Young's-modulus error by ground-truth stiffness",
            "",
            "| E GT (kPa) | n | Mean ± SD (%) | Median (%) | 95% CI of mean (%) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for stiffness, grouped_values in sorted(grouped.items()):
        values = np.asarray(grouped_values, dtype=float)
        ci = bootstrap_ci(values, "mean", rng)
        summary_lines.append(
            f"| {stiffness:g} | {len(values)} | "
            f"{values.mean():.1f} ± {values.std(ddof=1):.1f} | "
            f"{np.median(values):.1f} | {fmt_ci(ci)} |"
        )

    ours = np.asarray(
        [number(row["Ours E err%"]) for row in baseline_rows], dtype=float
    )
    decoupled = np.asarray(
        [number(row["Decoupled E err%"]) for row in baseline_rows], dtype=float
    )
    differences = decoupled - ours
    difference_ci = bootstrap_ci(differences, "mean", rng)
    permutation_p = exact_paired_permutation_pvalue(differences)

    summary_lines.extend(
        [
            "",
            "## Four-scene paired baseline comparison",
            "",
            f"- Unified: {ours.mean():.1f} ± {ours.std(ddof=1):.1f}% E error.",
            (
                f"- Decoupled: {decoupled.mean():.1f} ± "
                f"{decoupled.std(ddof=1):.1f}% E error."
            ),
            (
                f"- Paired reduction: {differences.mean():.1f} percentage points; "
                f"bootstrap 95% CI {fmt_ci(difference_ci)}."
            ),
            (
                f"- Exact two-sided paired sign-flip test: p={permutation_p:.3f}. "
                "The comparison is exploratory and underpowered at n=4."
            ),
            "",
            "## Evidence limitations",
            "",
            (
                "- The 40-scene multi-anatomy and 24-scene C3VD-derived files "
                "contain only aggregate/group summaries; scene-level SDs and CIs "
                "cannot be reconstructed."
            ),
            (
                "- The optimization is deterministic for a fixed initialization; "
                "no multi-initialization study is available."
            ),
            (
                "- The free-node comparison contains one scene and is descriptive "
                "only; no inferential statistic is valid."
            ),
            "",
        ]
    )

    force_path = RESULTS / "small_bowel_force" / "summary.json"
    if force_path.exists():
        force_rows = json.loads(force_path.read_text(encoding="utf-8"))
        summary_lines.extend(
            [
                "## Public small-bowel video-force benchmark",
                "",
                "| Protocol | Window | RMSE mean ± fold SD (N) | Pearson r | R² |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in force_rows:
            summary_lines.append(
                f"| {row['protocol']} | {row['window']} | "
                f"{row['rmse_n']['mean']:.3f} ± {row['rmse_n']['sd']:.3f} | "
                f"{row['pearson_r']['mean']:.3f} ± {row['pearson_r']['sd']:.3f} | "
                f"{row['r2']['mean']:.3f} ± {row['r2']['sd']:.3f} |"
            )
        summary_lines.extend(
            [
                "",
                "Folds are recording-disjoint; this benchmark validates the force "
                "front end and is not a Young's-modulus experiment.",
                "",
            ]
        )

    ets_path = RESULTS / "ets_phantom_sensitivity.json"
    if ets_path.exists():
        ets = json.loads(ets_path.read_text(encoding="utf-8"))
        summary_lines.extend(
            [
                "## ÉTS phantom sensitivity analysis",
                "",
                f"- Median ratio range: {ets['median_ratio_range'][0]:.3f}–"
                f"{ets['median_ratio_range'][1]:.3f}.",
                f"- Relative-error range: {ets['relative_error_range_percent'][0]:.1f}%–"
                f"{ets['relative_error_range_percent'][1]:.1f}%.",
                "",
            ]
        )

    sim_v3_path = RESULTS / "sim_v3" / "eval.json"
    if sim_v3_path.exists():
        sim_v3 = json.loads(sim_v3_path.read_text(encoding="utf-8"))
        summary_lines.extend(
            [
                "## sim_v3 frozen-protocol evaluation",
                "",
                "| Force condition | E median (95% CI) | Albedo median | "
                "Roughness median | PSNR median | SSIM median |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, group in sim_v3["groups"].items():
            e = group["E_rel"]
            summary_lines.append(
                f"| {condition} | {e['median']*100:.1f}% "
                f"({e['bootstrap_95_ci_of_median'][0]*100:.1f}–"
                f"{e['bootstrap_95_ci_of_median'][1]*100:.1f}%) | "
                f"{group['albedo_rel']['median']*100:.1f}% | "
                f"{group['roughness_rel']['median']*100:.1f}% | "
                f"{group['psnr']['median']:.2f} | {group['ssim']['median']:.3f} |"
            )
        summary_lines.extend(
            [
                "",
                "The force conditions are paired on the same test scenes. "
                "sim_v3 is controlled simulation evidence, separate from the "
                "ÉTS stiffness-contrast and real-video force experiments.",
                "",
            ]
        )

    output = RESULTS / "paper_statistics.md"
    output.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
