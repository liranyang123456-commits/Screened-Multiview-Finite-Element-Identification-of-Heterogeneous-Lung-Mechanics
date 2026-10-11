"""Label-aware and label-free headroom for PINN vs its own start."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _rel(estimate: float, truth: float) -> float:
    truth = float(truth)
    if abs(truth) < 1e-12:
        return abs(float(estimate) - truth)
    return abs(float(estimate) / truth - 1.0)


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else math.nan


def _geom_pass(records: list[dict], field: str, gate: float = 0.25) -> str:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in records:
        geometry = row.get("geometry_id") or str(row["patient_id"]).split("_scenario_")[0]
        buckets[geometry].append(row[field])
    medians = {key: float(np.median(values)) for key, values in buckets.items()}
    return f"{sum(value < gate for value in medians.values())}/{len(medians)}"


def analyze(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["records"]
    pinn_e = [row["E_background_relative_error"] for row in rows]
    start_e = [_rel(row["initial_E_background"], row["E_background_true"]) for row in rows]
    pinn_r = [row["inclusion_ratio_relative_error"] for row in rows]
    start_r = [_rel(row["initial_ratio"], row["inclusion_ratio_true"]) for row in rows]
    oracle_e = [min(a, b) for a, b in zip(pinn_e, start_e)]
    oracle_r = [min(a, b) for a, b in zip(pinn_r, start_r)]

    accept_ratio = []
    reject_ratio = []
    for row, start_err, pinn_err in zip(rows, start_r, pinn_r):
        history = row.get("loss_history") or []
        first = history[0] if history else {}
        last = row.get("final_loss") or (history[-1] if history else {})
        data0 = float(first.get("data", math.nan))
        data1 = float(last.get("data", math.nan))
        phys0 = float(first.get("physics", math.nan))
        phys1 = float(last.get("physics", math.nan))
        loss0 = float(first.get("loss", math.nan))
        loss1 = float(last.get("loss", math.nan))
        drift = abs(math.log(max(row["inclusion_ratio_estimated"], 1e-8)) - math.log(max(row["initial_ratio"], 1e-8)))
        data_drop = (data0 - data1) / max(abs(data0), 1e-12)
        phys_drop = (phys0 - phys1) / max(abs(phys0), 1e-12)
        loss_drop = (loss0 - loss1) / max(abs(loss0), 1e-12)
        accept_ratio.append(
            {
                "improved": pinn_err < start_err - 1e-12,
                "drift": drift,
                "data_drop": data_drop,
                "phys_drop": phys_drop,
                "loss_drop": loss_drop,
            }
        )
        # Label-free accept if ratio barely moved (trust start) else keep PINN
        # Two rules:
        # A: accept PINN only if |Δlog ρ| > 0.08 and data_drop > 0.15
        # B: accept PINN only if loss_drop > 0.99 (physics collapse from huge init)
        # C: always rollback ratio (keep start ratio, PINN E)
        # D: accept PINN ratio only if drift < 0.05 (stay near official)

    def screened(rule: str) -> tuple[list[float], list[float], int]:
        e_vals, r_vals = [], []
        accepted = 0
        for row, meta, e_p, e_s, r_p, r_s in zip(rows, accept_ratio, pinn_e, start_e, pinn_r, start_r):
            if rule == "ratio_rollback":
                take = False
            elif rule == "small_drift":
                take = meta["drift"] >= 0.05
            elif rule == "data_and_drift":
                take = meta["drift"] >= 0.08 and meta["data_drop"] >= 0.15
            elif rule == "loss_drop":
                take = meta["loss_drop"] >= 0.5
            else:
                take = True
            accepted += int(take)
            e_vals.append(e_p if take else e_s)
            r_vals.append(r_p if take else r_s)
        return e_vals, r_vals, accepted

    improved = [meta["improved"] for meta in accept_ratio]
    drifts_good = [meta["drift"] for meta, flag in zip(accept_ratio, improved) if flag]
    drifts_bad = [meta["drift"] for meta, flag in zip(accept_ratio, improved) if not flag]
    data_good = [meta["data_drop"] for meta, flag in zip(accept_ratio, improved) if flag]
    data_bad = [meta["data_drop"] for meta, flag in zip(accept_ratio, improved) if not flag]

    screened_rows = {}
    for rule in ("ratio_rollback", "small_drift", "data_and_drift", "loss_drop"):
        e_vals, r_vals, accepted = screened(rule)
        item = {
            "E_median": _median(e_vals),
            "ratio_median": _median(r_vals),
            "accepted": accepted,
            "accept_rate": accepted / len(rows),
        }
        if rows[0].get("geometry_id") or "_scenario_" in str(rows[0]["patient_id"]):
            fake = [
                {
                    "patient_id": row["patient_id"],
                    "geometry_id": row.get("geometry_id"),
                    "inclusion_ratio_relative_error": r_vals[index],
                }
                for index, row in enumerate(rows)
            ]
            item["geometry_pass_0.25"] = _geom_pass(fake, "inclusion_ratio_relative_error")
        screened_rows[rule] = item

    oracle_fake = [
        {
            "patient_id": row["patient_id"],
            "geometry_id": row.get("geometry_id"),
            "inclusion_ratio_relative_error": oracle_r[index],
        }
        for index, row in enumerate(rows)
    ]
    return {
        "label": label,
        "n": len(rows),
        "pinn": {"E": _median(pinn_e), "ratio": _median(pinn_r)},
        "start": {"E": _median(start_e), "ratio": _median(start_r)},
        "oracle_keep_better": {
            "E": _median(oracle_e),
            "ratio": _median(oracle_r),
            "ratio_geometry_pass_0.25": _geom_pass(oracle_fake, "inclusion_ratio_relative_error")
            if rows[0].get("geometry_id") or "_scenario_" in str(rows[0]["patient_id"])
            else None,
            "E_wins_kept": int(sum(a < b for a, b in zip(pinn_e, start_e))),
            "ratio_wins_kept": int(sum(a < b for a, b in zip(pinn_r, start_r))),
        },
        "label_free_screen": screened_rows,
        "signal": {
            "ratio_improved_n": int(sum(improved)),
            "drift_when_better": _median(drifts_good),
            "drift_when_worse": _median(drifts_bad),
            "data_drop_when_better": _median(data_good),
            "data_drop_when_worse": _median(data_bad),
        },
    }


def main() -> None:
    payload = {
        "synthetic50": analyze(
            ROOT / "results" / "lung_mesh_ai_v2" / "pinn_inverse50_v1" / "summary.json",
            "synthetic50",
        ),
        "ct120_officialstart": analyze(
            ROOT / "results" / "ion_ct_pinn_inverse120_officialstart_v1" / "summary.json",
            "ct120_officialstart",
        ),
    }
    out = ROOT / "results" / "ion_ct_pinn_inverse120_officialstart_v1" / "headroom.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
