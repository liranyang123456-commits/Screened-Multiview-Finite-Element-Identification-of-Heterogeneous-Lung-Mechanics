#!/usr/bin/env python3
"""Validation-selected CT hybrid: MeshGNN E/center + best-val ratio seed + PCA–Ridge radius."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "results" / "ion_ct_expanded_training"
BENCH = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
OUT = ROOT / "results" / "ion_ct_ratio_optimization" / "hybrid_ratio_specialized.json"
GATE = 0.25


def key(row: dict) -> str:
    if "entry_id" in row:
        return str(row["entry_id"]).replace("\\", "/")
    patient = str(row["patient_id"])
    if "_scenario_" in patient:
        geom, scenario = patient.split("_scenario_", 1)
        geom = geom.replace("geom_", "geom_", 1)
        if not geom.startswith("geom_"):
            geom = f"geom_{geom}"
        return f"{geom}/scenario_{scenario}"
    return patient


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    e = [abs(r["E_background_estimated"] / r["E_background_true"] - 1) for r in rows]
    rho = [abs(r["inclusion_ratio_estimated"] / r["inclusion_ratio_true"] - 1) for r in rows]
    het = [r for r in rows if r.get("heterogeneous_true", r.get("heterogeneous", True))]
    c = [r["center_error_normalized"] for r in het if r.get("center_error_normalized") is not None]
    rad = [r["radius_relative_error"] for r in het if r.get("radius_relative_error") is not None]
    by = defaultdict(list)
    for r in rows:
        geom = str(r.get("geometry_id") or key(r).split("/")[0])
        by[geom].append(abs(r["inclusion_ratio_estimated"] / r["inclusion_ratio_true"] - 1))
    geom_med = {g: median(v) for g, v in sorted(by.items())}
    return {
        "n": len(rows),
        "E": median(e),
        "ratio": median(rho),
        "center": median(c),
        "radius": median(rad),
        "ratio_pass_rate": float(np.mean(np.asarray(rho) <= GATE)),
        "ratio_geometries_passing": int(sum(m <= GATE for m in geom_med.values())),
        "geometry_ratio_medians": geom_med,
    }


def index_seed(seed: int, split: str) -> dict[str, dict]:
    path = TRAINING / f"seed_{seed}" / f"predictions_gnn_{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key(row): row for row in payload["records"]}


def main() -> None:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    pca = {
        row["entry_id"]: row
        for row in bench["records"]
        if row["method"] == "pca_ridge"
    }
    mesh_old = {
        row["entry_id"]: row
        for row in bench["records"]
        if row["method"] == "ours_hybrid_initializer"
    }
    e_center = index_seed(2027, "test")
    ratio = index_seed(2028, "test")
    composed = []
    for entry_id, mesh in mesh_old.items():
        if entry_id not in e_center or entry_id not in ratio or entry_id not in pca:
            # try without geom_ prefix mismatch
            continue
        row = dict(mesh)
        src_e = e_center[entry_id]
        src_r = ratio[entry_id]
        src_rad = pca[entry_id]
        row["method"] = "ours_hybrid_ratio_specialized"
        row["E_background_estimated"] = src_e["E_background_estimated"]
        row["E_background_relative_error"] = abs(
            src_e["E_background_estimated"] / src_e["E_background_true"] - 1
        )
        row["inclusion_ratio_estimated"] = src_r["inclusion_ratio_estimated"]
        row["inclusion_ratio_relative_error"] = abs(
            src_r["inclusion_ratio_estimated"] / src_r["inclusion_ratio_true"] - 1
        )
        row["center_error_normalized"] = src_e.get("center_error_normalized", mesh.get("center_error_normalized"))
        row["radius_fraction_estimated"] = src_rad["radius_fraction_estimated"]
        row["radius_relative_error"] = src_rad["radius_relative_error"]
        row["initializer_components"] = {
            "background_center": "MeshGNN seed 2027 (composite validation)",
            "ratio": "MeshGNN seed 2028 (best validation ratio)",
            "radius": "train-CV PCA-Ridge radius head",
        }
        composed.append(row)
    report = {
        "schema_version": 1,
        "selection": "validation-only component heads; test unused for selection",
        "n_matched": len(composed),
        "frozen_hybrid": summarize(list(mesh_old.values())),
        "ratio_specialized_hybrid": summarize(composed),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "n_matched"}, indent=2))
    print("matched", len(composed), "of", len(mesh_old))


if __name__ == "__main__":
    main()
