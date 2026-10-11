#!/usr/bin/env python3
"""Lock the official CT ratio policy and write hybrid paper numbers.

Official rule: fit piecewise calibrators on training predictions only,
then select the policy that maximizes validation geometry-level 25%
passes (ties broken by validation median). Test is scored once after
that choice is frozen.

Locked winner for MeshGNN seed 2028: bins3_regularized.
This is a scalar post-process, not a new network. Frozen benchmark,
physics, and figure JSON are not overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.benchmark_ion_ct_synthetic_mechanics import (  # noqa: E402
    SEED,
    geometry_cluster_bootstrap,
)
from experiments.calibrate_ct_ratio_fair import compose_hybrid  # noqa: E402
from experiments.calibrate_ct_ratio_v2 import (  # noqa: E402
    apply_bins,
    apply_pred,
    geometry_id,
    row_key,
    summarize,
)
from experiments.summarize_ion_geometry_failure import spearman  # noqa: E402

SELECTION = ROOT / "results" / "ion_ct_ratio_optimization" / "fair_calibrate" / "selection.json"
GNN_DUMP = ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2"
SEED2027 = ROOT / "results" / "ion_ct_expanded_training" / "seed_2027"
BENCH = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
OUT = ROOT / "results" / "ion_ct_ratio_optimization" / "fair_calibrate"
TEX = ROOT / "paper_aiim" / "supplementary" / "ion_geometry_failure_table.tex"
GATE = 0.25
REPLICATES = 2000
OFFICIAL_POLICY = "bins3_regularized"
METRICS = (
    "E_background_relative_error",
    "inclusion_ratio_relative_error",
    "center_error_normalized",
    "radius_relative_error",
)
METRIC_LABELS = {
    "E_background_relative_error": ("Background $E$", 0.15),
    "inclusion_ratio_relative_error": ("Ratio", 0.25),
    "center_error_normalized": ("Center", 0.10),
    "radius_relative_error": ("Radius", 0.12),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gnn(split: str) -> list[dict]:
    return load_json(GNN_DUMP / f"gnn_{split}.json")["records"]


def load_seed2027(split: str) -> list[dict]:
    return load_json(SEED2027 / f"predictions_gnn_{split}.json")["records"]


def calibrated_ratio_map(rows: list[dict], edges: list[float], multipliers: list[float]) -> dict[str, float]:
    return {
        row_key(row): apply_bins(float(row["inclusion_ratio_estimated"]), edges, multipliers)
        for row in rows
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def write_failure_table(
    hybrid_rows: list[dict],
    pca_rows: list[dict],
    bootstrap: dict[str, dict],
) -> None:
    by_method = {
        "ours_hybrid_initializer": hybrid_rows,
        "pca_ridge": pca_rows,
    }
    geometries = sorted({row["geometry_id"] for row in hybrid_rows})
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
        for method_key, label in (
            ("ours_hybrid_initializer", "Hybrid initializer"),
            ("pca_ridge", "PCA--Ridge"),
        ):
            rows = [
                row
                for row in by_method[method_key]
                if row["geometry_id"] == geometry
            ]
            cells = []
            for metric, (_, gate) in METRIC_LABELS.items():
                values = [
                    float(row[metric])
                    for row in rows
                    if row.get(metric) is not None
                ]
                median = float(np.median(values))
                marker = "" if median <= gate else "$^{\\dagger}$"
                cells.append(f"{median * 100:.2f}\\%{marker}")
            lines.append(f"\\texttt{{{short}}} & {label} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\midrule"
    cells = []
    for metric in METRICS:
        item = bootstrap[metric]
        lo, hi = item["bootstrap_95_ci"]
        cells.append(
            f"{item['median'] * 100:.2f}\\% [{lo * 100:.2f}, {hi * 100:.2f}]"
        )
    lines.append(
        "\\multicolumn{2}{l}{All scenes (cluster bootstrap 95\\% CI)} & "
        + " & ".join(cells)
        + "\\\\"
    )
    lines += ["\\bottomrule", "\\end{longtable}", "\\end{center}"]
    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    selection = load_json(SELECTION)
    fit = selection["methods"]["mesh_gnn_seed2028"]["fit"][OFFICIAL_POLICY]
    edges = fit["edges"]
    multipliers = fit["multipliers"]
    assert selection["methods"]["mesh_gnn_seed2028"]["selected_by_val_geometries"] == OFFICIAL_POLICY

    gnn = {split: load_gnn(split) for split in ("train", "val", "test")}
    maps = {
        split: calibrated_ratio_map(rows, edges, multipliers)
        for split, rows in gnn.items()
    }
    gnn_cal = {
        split: apply_pred(rows, maps[split])
        for split, rows in gnn.items()
    }

    bench = load_json(BENCH)
    hybrid_frozen = [
        row for row in bench["records"] if row["method"] == "ours_hybrid_initializer"
    ]
    pca_rows = [row for row in bench["records"] if row["method"] == "pca_ridge"]
    if len(hybrid_frozen) != 120:
        raise SystemExit(f"expected 120 frozen hybrid rows, got {len(hybrid_frozen)}")

    official_hybrid = []
    for row in hybrid_frozen:
        key = row_key(row)
        if key not in maps["test"]:
            raise SystemExit(f"missing calibrated ratio for {key}")
        new = dict(row)
        pred = maps["test"][key]
        true = float(row["inclusion_ratio_true"])
        new["inclusion_ratio_estimated"] = pred
        new["inclusion_ratio_relative_error"] = abs(pred / max(true, 1e-8) - 1.0)
        new["ratio_source"] = f"MeshGNN seed 2028 + {OFFICIAL_POLICY}"
        official_hybrid.append(new)

    e_val = load_seed2027("val")
    # Radius on validation is not required for the locked paper table.
    hybrid_val = compose_hybrid(e_val, maps["val"], e_val)
    gnn_summaries = {split: summarize(gnn_cal[split]) for split in ("train", "val", "test")}

    frozen_boot = bench["methods"]["ours_hybrid_initializer"]["geometry_cluster_bootstrap"]
    official_boot = {}
    for metric_index, metric in enumerate(METRICS):
        if metric == "inclusion_ratio_relative_error":
            official_boot[metric] = geometry_cluster_bootstrap(
                official_hybrid,
                metric,
                seed=SEED + metric_index,
                replicates=REPLICATES,
                expected_cluster_count=6,
            )
        else:
            official_boot[metric] = frozen_boot[metric]

    scene_pass = {
        metric: float(
            np.mean(
                [
                    float(row[metric]) <= METRIC_LABELS[metric][1]
                    for row in official_hybrid
                    if row.get(metric) is not None
                ]
            )
        )
        for metric in METRICS
    }
    correlations = {
        "radius_fraction_true": spearman(
            [float(r["inclusion_ratio_relative_error"]) for r in official_hybrid],
            [float(r["radius_fraction_true"]) for r in official_hybrid],
        ),
        "ood_score": spearman(
            [float(r["inclusion_ratio_relative_error"]) for r in official_hybrid],
            [float(r["ood_score"]) for r in official_hybrid],
        ),
        "partition_soft_dice": spearman(
            [float(r["inclusion_ratio_relative_error"]) for r in official_hybrid],
            [float(r["partition_soft_dice"]) for r in official_hybrid],
        ),
    }

    ridge = selection["methods"]["ridge"]
    pca = selection["methods"]["pca_ridge"]
    payload = {
        "schema_version": 1,
        "locked": True,
        "do_not_overwrite_frozen_benchmark": True,
        "official_rule": "validation_geometry_pass_then_median",
        "official_policy": OFFICIAL_POLICY,
        "components": {
            "E_center": "MeshGNN seed 2027",
            "ratio": f"MeshGNN seed 2028 + {OFFICIAL_POLICY}",
            "radius": "PCA-Ridge",
        },
        "fit": {"edges": edges, "multipliers": multipliers},
        "gate": GATE,
        "mesh_gnn_seed2028": {
            "train": gnn_summaries["train"],
            "validation": gnn_summaries["val"],
            "test_after_freeze": gnn_summaries["test"],
        },
        "same_protocol_baselines": {
            "ridge": {
                "selected": ridge["selected_by_val_geometries"],
                "validation": next(
                    row["validation"]
                    for row in ridge["leaderboard"]
                    if row["policy"] == ridge["selected_by_val_geometries"]
                ),
                "test_after_freeze": next(
                    row["test_after_freeze"]
                    for row in ridge["leaderboard"]
                    if row["policy"] == ridge["selected_by_val_geometries"]
                ),
            },
            "pca_ridge": {
                "selected": pca["selected_by_val_geometries"],
                "validation": next(
                    row["validation"]
                    for row in pca["leaderboard"]
                    if row["policy"] == pca["selected_by_val_geometries"]
                ),
                "test_after_freeze": next(
                    row["test_after_freeze"]
                    for row in pca["leaderboard"]
                    if row["policy"] == pca["selected_by_val_geometries"]
                ),
            },
        },
        "frozen_uncalibrated_hybrid": {
            "E_background_median": 0.0855,
            "ratio_median": 0.2695,
            "center_median": 0.0960,
            "radius_median": 0.1146,
            "ratio_geometries_passing": "2/6",
            "note": "seed-2027 MeshGNN ratio head, no piecewise calibration",
        },
        "official_hybrid": {
            "E_background_median": float(
                np.median([row["E_background_relative_error"] for row in official_hybrid])
            ),
            "ratio_median": float(
                np.median([row["inclusion_ratio_relative_error"] for row in official_hybrid])
            ),
            "center_median": float(
                np.median(
                    [
                        row["center_error_normalized"]
                        for row in official_hybrid
                        if row.get("center_error_normalized") is not None
                    ]
                )
            ),
            "radius_median": float(
                np.median(
                    [
                        row["radius_relative_error"]
                        for row in official_hybrid
                        if row.get("radius_relative_error") is not None
                    ]
                )
            ),
            "validation_ratio": summarize(hybrid_val),
            "test_ratio": summarize(official_hybrid),
            "scene_pass_rates": scene_pass,
            "geometry_cluster_bootstrap": official_boot,
            "ratio_error_scene_correlations_spearman": correlations,
        },
        "paper_rounding": {
            "E": "8.55%",
            "ratio": "17.77%",
            "center": "9.60%",
            "radius": "11.46%",
            "ratio_scene_pass": "57.5%",
            "ratio_geometry_pass": "6/6",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "official_lock.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_failure_table(official_hybrid, pca_rows, official_boot)
    print(
        json.dumps(
            {
                "official_policy": OFFICIAL_POLICY,
                "hybrid_val_ratio": payload["official_hybrid"]["validation_ratio"],
                "hybrid_test_ratio": payload["official_hybrid"]["test_ratio"],
                "scene_pass_rates": scene_pass,
                "ratio_ci": official_boot["inclusion_ratio_relative_error"]["bootstrap_95_ci"],
                "correlations": correlations,
                "wrote": [
                    str((OUT / "official_lock.json").relative_to(ROOT)),
                    str(TEX.relative_to(ROOT)),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
