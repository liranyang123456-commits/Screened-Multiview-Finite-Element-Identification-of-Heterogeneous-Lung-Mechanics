"""Full four-endpoint scoreboard across frozen and new inverse arms."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "inverse_arm_scoreboard_20260825.json"

GATES = {
    "E": 0.15,
    "ratio": 0.25,
    "center": 0.10,
    "radius": 0.12,
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nid(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def _median_ci(values: list[float], seed: int = 2026) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"median": math.nan, "ci95": [math.nan, math.nan], "n": 0}
    rng = np.random.default_rng(seed)
    samples = [float(np.median(rng.choice(array, size=len(array), replace=True))) for _ in range(5000)]
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"median": float(np.median(array)), "ci95": [float(low), float(high)], "n": int(len(array))}


def _pass_rate(values: list[float], gate: float) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return float(np.mean(array <= gate))


def _geom_pass(rows: list[dict], field: str, gate: float) -> dict | None:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        geometry = row.get("geometry_id")
        if not geometry:
            patient = str(row.get("patient_id", ""))
            if "_scenario_" in patient:
                geometry = patient.split("_scenario_")[0]
        if not geometry or row.get(field) is None:
            continue
        buckets[str(geometry)].append(float(row[field]))
    if not buckets:
        return None
    medians = {key: float(np.median(values)) for key, values in buckets.items()}
    passing = sum(value <= gate for value in medians.values())
    return {
        "pass": f"{passing}/{len(medians)}",
        "medians": medians,
    }


def _from_records(rows: list[dict], *, label: str, status: str, notes: str = "") -> dict:
    e = [float(row["E_background_relative_error"]) for row in rows]
    ratio = [float(row["inclusion_ratio_relative_error"]) for row in rows]
    hetero = [row for row in rows if row.get("heterogeneous_true")]
    center = [float(row["center_error_normalized"]) for row in hetero if row.get("center_error_normalized") is not None]
    radius = [float(row["radius_relative_error"]) for row in hetero if row.get("radius_relative_error") is not None]
    force = [float(row["force_scale_relative_error"]) for row in rows if row.get("force_scale_relative_error") is not None]
    wall = [float(row["wall_time_seconds"]) for row in rows if row.get("wall_time_seconds") is not None]
    evals = [float(row["function_evaluations"]) for row in rows if row.get("function_evaluations") is not None]
    payload = {
        "label": label,
        "status": status,
        "n": len(rows),
        "heterogeneous_n": len(hetero),
        "E": _median_ci(e),
        "ratio": _median_ci(ratio),
        "center": _median_ci(center),
        "radius": _median_ci(radius),
        "force": _median_ci(force) if force else None,
        "wall_s": float(np.median(wall)) if wall else None,
        "evals": float(np.median(evals)) if evals else None,
        "scene_pass": {
            "E_15": _pass_rate(e, GATES["E"]),
            "ratio_25": _pass_rate(ratio, GATES["ratio"]),
            "center_10": _pass_rate(center, GATES["center"]),
            "radius_12": _pass_rate(radius, GATES["radius"]),
        },
        "geometry_ratio_pass_25": _geom_pass(rows, "inclusion_ratio_relative_error", 0.25),
        "accept_rate": float(np.mean([bool(row.get("refinement_accepted")) for row in rows]))
        if any("refinement_accepted" in row for row in rows)
        else None,
        "screen_reject_rate": float(np.mean([bool(row.get("screening_rejected")) for row in rows]))
        if any("screening_rejected" in row for row in rows)
        else None,
        "notes": notes,
    }
    return payload


def main() -> None:
    syn_map = _load(ROOT / "results" / "lung_mesh_ai_v2" / "fair_multiview_fem50_strict" / "fair_multiview_summary.json")
    syn_map_rows = [
        row
        for row in syn_map["records"]
        if row.get("benchmark_method") == "calibrated_multiview_screened_map_strict"
    ]
    cal = _load(ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json")
    pinn50 = _load(ROOT / "results" / "lung_mesh_ai_v2" / "pinn_inverse50_v1" / "summary.json")
    locked50 = _load(ROOT / "results" / "lung_mesh_ai_v2" / "screened_fem_locked50_v1" / "summary.json")
    multi = _load(ROOT / "results" / "lung_mesh_ai_v2" / "fem_multistart50_v1" / "summary.json")
    official = _load(ROOT / "results" / "ion_ct_ratio_optimization" / "fair_calibrate" / "official_lock.json")
    pinn120_off = _load(ROOT / "results" / "ion_ct_pinn_inverse120_officialstart_v1" / "summary.json")
    pinn120_hyb = _load(ROOT / "results" / "ion_ct_pinn_inverse120_v1" / "summary.json")
    stack = _load(ROOT / "results" / "ion_ct_ratio_optimization" / "stack_covariate_v1" / "selection.json")
    bench = _load(ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json")
    hybrid_rows = [row for row in bench["records"] if row.get("method") == "ours_hybrid_initializer"]
    gnn_test = _load(ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2" / "gnn_test.json")["records"]
    from lung_inverse_rendering.pinn_inverse import official_bins3_regularized

    official_rows = []
    gnn_index = {_nid(row["patient_id"]): row for row in gnn_test}
    for row in hybrid_rows:
        key = _nid(row["patient_id"])
        gnn = gnn_index[key]
        ratio = official_bins3_regularized(gnn["inclusion_ratio_estimated"])
        new = dict(row)
        new["inclusion_ratio_estimated"] = ratio
        new["inclusion_ratio_relative_error"] = abs(ratio / max(float(row["inclusion_ratio_true"]), 1e-8) - 1.0)
        official_rows.append(new)

    fem_path = ROOT / "results" / "ion_ct_official_prior_screened_fem120_v1" / "records.jsonl"
    fem_rows = [json.loads(line) for line in fem_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fem_ids = {_nid(row["patient_id"]) for row in fem_rows}
    official_partial = [row for row in official_rows if _nid(row["patient_id"]) in fem_ids]

    scoreboard = {
        "generated": "2026-08-25",
        "gates": GATES,
        "synthetic50": {
            "calibrator": {
                "label": "响应校准器 PCA–Ridge",
                "status": "complete",
                "n": 50,
                "E": {"median": cal["test"]["E_background_median_relative_error"], "ci95": None},
                "ratio": {"median": cal["test"]["inclusion_ratio_median_relative_error"], "ci95": None},
                "center": {"median": cal["test"]["center_error_normalized_median"], "ci95": None},
                "radius": {"median": cal["test"]["radius_relative_error_median"], "ci95": None},
                "notes": "论文校准器；E 3.82% 是合成上最强先验",
            },
            "screened_map": _from_records(
                syn_map_rows,
                label="Screened MAP（冻结）",
                status="complete",
                notes="接受率 8%；代价筛选回滚",
            ),
            "locked_fem": _from_records(
                locked50["records"],
                label="锁ρ + Hessian 门的 screened FEM",
                status="complete",
                notes="与 screened MAP 中位相同；Hessian 回滚 0%",
            ),
            "pinn": _from_records(
                pinn50["records"],
                label="PINN 反演臂",
                status="complete",
                notes="无筛选；E 差于 MAP，ratio 区间重叠",
            ),
            "fixed_fem": {
                "label": "固定初值 FEM",
                "status": "complete",
                "n": 50,
                "E": {
                    "median": syn_map["methods"]["fixed_init"]["E_background_median_relative_error"],
                    "ci95": syn_map["methods"]["fixed_init"]["E_background_median_bootstrap_95_ci"],
                },
                "ratio": {
                    "median": syn_map["methods"]["fixed_init"]["inclusion_ratio_median_relative_error"],
                    "ci95": syn_map["methods"]["fixed_init"]["inclusion_ratio_median_bootstrap_95_ci"],
                },
                "wall_s": syn_map["methods"]["fixed_init"]["wall_time_seconds_median"],
                "evals": syn_map["methods"]["fixed_init"]["function_evaluations_median"],
            },
            "multistart_fem": {
                "label": "四起点 FEM（无筛选）",
                "status": "complete",
                "n": 50,
                "E": {"median": multi["E"], "ci95": multi["E_ci"]},
                "ratio": {"median": multi["ratio"], "ci95": multi["ratio_ci"]},
                "evals": multi["total_evals_median"],
                "notes": "接受率 92%；更多起点没有赢过 screened MAP",
            },
        },
        "ct120": {
            "official_hybrid": _from_records(
                official_rows,
                label="官方 hybrid（2027 E/中心 + 2028 bins3 + PCA 半径）",
                status="complete",
                notes="论文锁定数字 8.55 / 17.77 / 9.60 / 11.46；ratio 6/6",
            ),
            "uncalibrated_hybrid": {
                "label": "未校准 hybrid（2027 ratio）",
                "status": "complete",
                "n": 120,
                "E": {"median": official["frozen_uncalibrated_hybrid"]["E_background_median"]},
                "ratio": {"median": official["frozen_uncalibrated_hybrid"]["ratio_median"]},
                "center": {"median": official["frozen_uncalibrated_hybrid"]["center_median"]},
                "radius": {"median": official["frozen_uncalibrated_hybrid"]["radius_median"]},
                "geometry_ratio_pass_25": {"pass": official["frozen_uncalibrated_hybrid"]["ratio_geometries_passing"]},
            },
            "pinn_officialstart": _from_records(
                pinn120_off["records"],
                label="PINN（官方 2028+bins3 起点）",
                status="complete",
                notes="走丢官方 ratio；3/6 过门",
            ),
            "pinn_hybridstart": _from_records(
                pinn120_hyb["records"],
                label="PINN（hybrid 初始化器+bins3 起点）",
                status="complete",
                notes="起点不是官方 17.77%；2/6 过门",
            ),
            "ridge_same_protocol": {
                "label": "Ridge（同协议）",
                "status": "complete",
                "n": 120,
                "ratio": {"median": official["same_protocol_baselines"]["ridge"]["test_after_freeze"]["median"]},
                "geometry_ratio_pass_25": {
                    "pass": f"{official['same_protocol_baselines']['ridge']['test_after_freeze']['geometries_passing']}/6"
                },
            },
            "pca_ridge_same_protocol": {
                "label": "PCA–Ridge（同协议）",
                "status": "complete",
                "n": 120,
                "ratio": {"median": official["same_protocol_baselines"]["pca_ridge"]["test_after_freeze"]["median"]},
                "geometry_ratio_pass_25": {
                    "pass": f"{official['same_protocol_baselines']['pca_ridge']['test_after_freeze']['geometries_passing']}/6"
                },
            },
            "official_prior_fem_partial": _from_records(
                fem_rows,
                label="官方先验 screened FEM（进行中）",
                status=f"partial_{len(fem_rows)}/120",
                notes="ratio 锁死；进程曾中断，已续跑。部分集不能当最终数。",
            ),
            "official_hybrid_on_same_partial": _from_records(
                official_partial,
                label="官方 hybrid（同一已完成子集）",
                status=f"partial_{len(official_partial)}/120",
                notes="与 FEM 部分集对齐，便于看 E 是否被 FEM 走坏",
            ),
        },
        "calibrator_stacking_val_select": {
            "rule": "validation_geometry_pass_then_median",
            "selected": stack["selected_by_val_geometries"],
            "leaderboard": [
                {
                    "policy": row["policy"],
                    "val_median": row["validation"]["median"],
                    "val_geom_pass": f"{row['validation']['geometries_passing']}/{row['validation']['geometry_count']}",
                    "test_median": row["test_after_freeze"]["median"],
                    "test_geom_pass": f"{row['test_after_freeze']['geometries_passing']}/{row['test_after_freeze']['geometry_count']}",
                }
                for row in stack["leaderboard"]
            ],
        },
    }
    OUT.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    print(json.dumps(scoreboard, indent=2)[:4000])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
