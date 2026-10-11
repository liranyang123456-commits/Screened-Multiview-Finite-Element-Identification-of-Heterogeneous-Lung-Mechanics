"""Pairwise comparison of the PINN inverse arm against frozen baselines."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.pinn_inverse import official_bins3_regularized


def _normalize_id(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def _relative_error(estimate: float, truth: float) -> float:
    truth = float(truth)
    if abs(truth) < 1e-12:
        return abs(float(estimate) - truth)
    return abs(float(estimate) / truth - 1.0)


def _pair_stats(pinn: list[float], reference: list[float]) -> dict:
    delta = np.asarray(pinn, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "n": int(len(delta)),
        "pinn_median": float(np.median(pinn)),
        "ref_median": float(np.median(reference)),
        "median_delta_pp": float(np.median(delta) * 100.0),
        "pinn_wins": int(np.sum(delta < -1e-12)),
        "ref_wins": int(np.sum(delta > 1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_synthetic50() -> dict:
    pinn = _load(ROOT / "results" / "lung_mesh_ai_v2" / "pinn_inverse50_v1" / "summary.json")
    screened = _load(
        ROOT / "results" / "lung_mesh_ai_v2" / "fair_multiview_fem50_strict" / "fair_multiview_summary.json"
    )
    pinn_index = {_normalize_id(row["patient_id"]): row for row in pinn["records"]}
    map_index = {
        _normalize_id(row["patient_id"]): row
        for row in screened["records"]
        if row.get("benchmark_method") == "calibrated_multiview_screened_map_strict"
    }
    ids = sorted(set(pinn_index) & set(map_index))
    pinn_e = [pinn_index[key]["E_background_relative_error"] for key in ids]
    map_e = [map_index[key]["E_background_relative_error"] for key in ids]
    pinn_r = [pinn_index[key]["inclusion_ratio_relative_error"] for key in ids]
    map_r = [map_index[key]["inclusion_ratio_relative_error"] for key in ids]
    start_e = [
        _relative_error(pinn_index[key]["initial_E_background"], pinn_index[key]["E_background_true"])
        for key in ids
    ]
    start_r = [
        _relative_error(pinn_index[key]["initial_ratio"], pinn_index[key]["inclusion_ratio_true"])
        for key in ids
    ]
    centers = [
        pinn_index[key]["center_error_normalized"]
        for key in ids
        if pinn_index[key]["center_error_normalized"] is not None
    ]
    radii = [
        pinn_index[key]["radius_relative_error"]
        for key in ids
        if pinn_index[key]["radius_relative_error"] is not None
    ]
    return {
        "n": len(ids),
        "E_vs_screened_map": _pair_stats(pinn_e, map_e),
        "ratio_vs_screened_map": _pair_stats(pinn_r, map_r),
        "E_vs_own_start": _pair_stats(pinn_e, start_e),
        "ratio_vs_own_start": _pair_stats(pinn_r, start_r),
        "center_pinn_median": float(np.median(centers)),
        "radius_pinn_median": float(np.median(radii)),
        "wall_pinn_median": float(np.median([pinn_index[key]["wall_time_seconds"] for key in ids])),
        "wall_map_median": float(np.median([map_index[key]["wall_time_seconds"] for key in ids])),
    }


def compare_ct120(summary_path: Path) -> dict:
    pinn = _load(summary_path)
    bench = _load(ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json")
    pinn_index = {_normalize_id(row["patient_id"]): row for row in pinn["records"]}
    hybrid = {
        _normalize_id(row["patient_id"]): row
        for row in bench["records"]
        if row.get("method") == "ours_hybrid_initializer"
    }
    ids = sorted(set(pinn_index) & set(hybrid))
    pinn_e = [pinn_index[key]["E_background_relative_error"] for key in ids]
    hybrid_e = [hybrid[key]["E_background_relative_error"] for key in ids]
    pinn_r = [pinn_index[key]["inclusion_ratio_relative_error"] for key in ids]
    gnn_2028 = {
        _normalize_id(row["patient_id"]): row
        for row in _load(ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2" / "gnn_test.json")[
            "records"
        ]
    }
    hybrid_r_raw = [hybrid[key]["inclusion_ratio_relative_error"] for key in ids]
    hybrid_r_cal = [
        _relative_error(
            official_bins3_regularized(gnn_2028[key]["inclusion_ratio_estimated"]),
            hybrid[key]["inclusion_ratio_true"],
        )
        for key in ids
    ]
    start_e = [
        _relative_error(pinn_index[key]["initial_E_background"], pinn_index[key]["E_background_true"])
        for key in ids
    ]
    start_r = [
        _relative_error(pinn_index[key]["initial_ratio"], pinn_index[key]["inclusion_ratio_true"])
        for key in ids
    ]
    geometry_pinn: dict[str, list[float]] = defaultdict(list)
    geometry_official: dict[str, list[float]] = defaultdict(list)
    for key in ids:
        geometry_id = pinn_index[key].get("geometry_id") or key.split("_scenario_")[0]
        geometry_pinn[geometry_id].append(pinn_index[key]["inclusion_ratio_relative_error"])
        geometry_official[geometry_id].append(
            _relative_error(
                official_bins3_regularized(gnn_2028[key]["inclusion_ratio_estimated"]),
                hybrid[key]["inclusion_ratio_true"],
            )
        )
    pinn_geom = {key: float(np.median(values)) for key, values in geometry_pinn.items()}
    official_geom = {key: float(np.median(values)) for key, values in geometry_official.items()}
    gate = 0.25
    centers = [
        pinn_index[key]["center_error_normalized"]
        for key in ids
        if pinn_index[key]["center_error_normalized"] is not None
    ]
    radii = [
        pinn_index[key]["radius_relative_error"]
        for key in ids
        if pinn_index[key]["radius_relative_error"] is not None
    ]
    return {
        "n": len(ids),
        "E_vs_official_hybrid": _pair_stats(pinn_e, hybrid_e),
        "ratio_vs_official_calibrated": _pair_stats(pinn_r, hybrid_r_cal),
        "ratio_vs_uncalibrated_hybrid": _pair_stats(pinn_r, hybrid_r_raw),
        "E_vs_own_start": _pair_stats(pinn_e, start_e),
        "ratio_vs_own_start": _pair_stats(pinn_r, start_r),
        "official_calibrated_ratio_median": float(np.median(hybrid_r_cal)),
        "uncalibrated_hybrid_ratio_median": float(np.median(hybrid_r_raw)),
        "pinn_ratio_gt_0.9": int(np.sum(np.asarray(pinn_r) > 0.9)),
        "pinn_geometry_ratio_medians": pinn_geom,
        "pinn_geometry_pass_0.25": f"{sum(value < gate for value in pinn_geom.values())}/{len(pinn_geom)}",
        "official_geometry_ratio_medians": official_geom,
        "official_geometry_pass_0.25": f"{sum(value < gate for value in official_geom.values())}/{len(official_geom)}",
        "center_pinn_median": float(np.median(centers)),
        "radius_pinn_median": float(np.median(radii)),
        "wall_pinn_median": float(np.median([pinn_index[key]["wall_time_seconds"] for key in ids])),
    }


def main() -> None:
    payload = {
        "note": (
            "PINN inverse arm vs frozen screened MAP (synthetic 50) and official "
            "hybrid (CT 120). Same generator, same four endpoints. Not a public SOTA claim."
        ),
        "synthetic50": compare_synthetic50(),
    }
    ct_dirs = {
        "hybrid_initializer_bins3": ROOT / "results" / "ion_ct_pinn_inverse120_v1" / "summary.json",
        "official_2028_bins3": ROOT / "results" / "ion_ct_pinn_inverse120_officialstart_v1" / "summary.json",
    }
    for label, path in ct_dirs.items():
        if path.exists():
            payload[f"ct120_{label}"] = compare_ct120(path)
            path.with_name("comparison.json").write_text(
                json.dumps({"note": payload["note"], "label": label, "ct120": payload[f"ct120_{label}"]}, indent=2),
                encoding="utf-8",
            )
    out_syn = ROOT / "results" / "lung_mesh_ai_v2" / "pinn_inverse50_v1" / "comparison.json"
    out_syn.write_text(json.dumps({"note": payload["note"], "synthetic50": payload["synthetic50"]}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
