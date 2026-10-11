"""Per-geometry failure breakdown for the held-out CT cohort (AIIM R4/R5).

Reads results/ion_ct_expanded120_final/benchmark.json and produces:
  1. geometry_failure_breakdown.json - per-geometry medians for the four gated
     metrics (hybrid initializer and PCA-Ridge), gate pass/fail, and scene-level
     Spearman correlations of ratio error with inclusion size, OOD score, and
     partition quality.
  2. paper_aiim/supplementary/ion_geometry_failure_table.tex - longtable
     fragment included by supplementary.tex.
  3. Console summary used for the main-text paragraph.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
OUT_JSON = ROOT / "results" / "ion_ct_expanded120_final" / "geometry_failure_breakdown.json"
OUT_TEX = ROOT / "paper_aiim" / "supplementary" / "ion_geometry_failure_table.tex"

METRICS = {
    "E_background_relative_error": ("Background $E$", 0.15),
    "inclusion_ratio_relative_error": ("Ratio", 0.25),
    "center_error_normalized": ("Center", 0.10),
    "radius_relative_error": ("Radius", 0.12),
}
METHODS = {"ours_hybrid_initializer": "Hybrid initializer", "pca_ridge": "PCA--Ridge"}


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            mean_rank = 0.5 * (index + stop) + 1.0
            for position in range(index, stop + 1):
                result[order[position]] = mean_rank
            index = stop + 1
        return result

    rx, ry = ranks(x), ranks(y)
    n = len(rx)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / (var_x * var_y) ** 0.5


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def main() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    records = payload["records"]
    geometries = sorted(
        {row["geometry_id"] for row in records if row["method"] == "ours_hybrid_initializer"}
    )

    breakdown: dict[str, dict] = {}
    for method_key, method_label in METHODS.items():
        per_geometry = {}
        for geometry in geometries:
            rows = [
                row
                for row in records
                if row["method"] == method_key and row["geometry_id"] == geometry
            ]
            metrics = {}
            for metric, (_, gate) in METRICS.items():
                values = [
                    float(row[metric]) for row in rows if row.get(metric) is not None
                ]
                value = median(values)
                metrics[metric] = {
                    "median": value,
                    "gate": gate,
                    "pass": value <= gate,
                    "count": len(values),
                }
            per_geometry[geometry] = {"scenes": len(rows), "metrics": metrics}
        breakdown[method_key] = {"label": method_label, "per_geometry": per_geometry}

    hybrid_rows = [row for row in records if row["method"] == "ours_hybrid_initializer"]
    correlations = {
        "radius_fraction_true": spearman(
            [float(r["inclusion_ratio_relative_error"]) for r in hybrid_rows],
            [float(r["radius_fraction_true"]) for r in hybrid_rows],
        ),
        "ood_score": spearman(
            [float(r["inclusion_ratio_relative_error"]) for r in hybrid_rows],
            [float(r["ood_score"]) for r in hybrid_rows],
        ),
        "partition_soft_dice": spearman(
            [float(r["inclusion_ratio_relative_error"]) for r in hybrid_rows],
            [float(r["partition_soft_dice"]) for r in hybrid_rows],
        ),
    }

    cluster = payload["methods"]["ours_hybrid_initializer"]["geometry_cluster_bootstrap"]
    summary = {
        "source": str(BENCHMARK.relative_to(ROOT)),
        "geometries": geometries,
        "methods": breakdown,
        "ratio_error_scene_correlations_spearman": correlations,
        "hybrid_geometry_cluster_bootstrap": cluster,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "\\begin{center}\\footnotesize",
        "\\begin{longtable}{llcccc}",
        "\\toprule",
        "Geometry & Method & Background $E$ & Ratio & Center & Radius\\\\",
        "& & (gate 15\\%) & (gate 25\\%) & (gate 10\\%) & (gate 12\\%)\\\\",
        "\\midrule",
        "\\endhead",
    ]
    for geometry in geometries:
        short = geometry.replace("geom_", "")[:8]
        for method_key, method_label in METHODS.items():
            entry = breakdown[method_key]["per_geometry"][geometry]
            cells = []
            for metric in METRICS:
                item = entry["metrics"][metric]
                marker = "" if item["pass"] else "$^{\\dagger}$"
                cells.append(f"{item['median'] * 100:.2f}\\%{marker}")
            label = method_label if method_key == "ours_hybrid_initializer" else "PCA--Ridge"
            lines.append(f"\\texttt{{{short}}} & {label} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\midrule"
    boot = cluster
    cells = []
    for metric in METRICS:
        ci = boot[metric]["bootstrap_95_ci"]
        cells.append(
            f"{boot[metric]['median'] * 100:.2f}\\% "
            f"[{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"
        )
    lines.append(
        "\\multicolumn{2}{l}{All scenes (cluster bootstrap 95\\% CI)} & "
        + " & ".join(cells)
        + "\\\\"
    )
    lines += ["\\bottomrule", "\\end{longtable}", "\\end{center}"]
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "correlations": correlations,
        "worst_geometry_ratio": max(
            (
                (g, breakdown["ours_hybrid_initializer"]["per_geometry"][g]["metrics"]
                 ["inclusion_ratio_relative_error"]["median"])
                for g in geometries
            ),
            key=lambda item: item[1],
        ),
        "bootstrap_ratio": boot["inclusion_ratio_relative_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
