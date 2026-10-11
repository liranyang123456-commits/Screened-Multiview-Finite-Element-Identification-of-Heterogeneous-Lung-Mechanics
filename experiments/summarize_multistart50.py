#!/usr/bin/env python3
"""Pick the lowest-cost start per patient and write paper-ready summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "lung_mesh_ai_v2" / "fem_multistart50_v1"
SEED = 2026
REPLICATES = 2000
STARTS = {
    0: (2500.0, 1.25),
    1: (5000.0, 1.8),
    2: (8000.0, 2.8),
    3: (12000.0, 4.0),
}


def load_records(index: int) -> list[dict]:
    folder = BASE / f"start_{index}"
    files = sorted(folder.glob("metrics_*.json"))
    if not files:
        return []
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    return payload.get("records", payload if isinstance(payload, list) else [])


def cost(row: dict) -> float:
    for key in ("cost", "optimized_cost", "final_cost", "objective"):
        if row.get(key) is not None:
            return float(row[key])
    return 1e9


def bootstrap_median(values: list[float], *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(REPLICATES)
    array = np.asarray(values, dtype=float)
    for index in range(REPLICATES):
        draws[index] = float(np.median(rng.choice(array, size=len(array), replace=True)))
    return np.quantile(draws, (0.025, 0.975)).tolist()


def main() -> None:
    starts = {index: {row["patient_id"]: row for row in load_records(index)} for index in range(4)}
    patients = sorted(set().union(*[set(mapping) for mapping in starts.values() if mapping]))
    chosen = []
    wall_totals = []
    eval_totals = []
    for patient in patients:
        candidates = []
        wall = 0.0
        evals = 0
        for index in range(4):
            if patient not in starts[index]:
                continue
            row = starts[index][patient]
            candidates.append((index, row))
            wall += float(row.get("wall_time_seconds", 0.0))
            evals += int(row.get("function_evaluations", 0))
        if not candidates:
            continue
        index, best = min(candidates, key=lambda item: cost(item[1]))
        chosen.append(
            {
                **best,
                "selected_start_index": index,
                "selected_start_E": STARTS[index][0],
                "selected_start_ratio": STARTS[index][1],
                "selected_cost": cost(best),
                "total_wall_time_seconds": wall,
                "total_function_evaluations": evals,
            }
        )
        wall_totals.append(wall)
        eval_totals.append(evals)

    e_vals = [float(row["E_background_relative_error"]) for row in chosen]
    r_vals = [float(row["inclusion_ratio_relative_error"]) for row in chosen]
    start_wins = {f"{E:g}/{ratio:g}": 0 for E, ratio in STARTS.values()}
    for row in chosen:
        start_wins[f"{row['selected_start_E']:g}/{row['selected_start_ratio']:g}"] += 1

    report = {
        "n": len(chosen),
        "per_start_n": {str(index): len(starts[index]) for index in range(4)},
        "protocol": "4 deterministic starts, 24 evaluations each, 96 total; lowest cost per patient",
        "starts": STARTS,
        "E": float(np.median(e_vals)),
        "E_ci": bootstrap_median(e_vals, seed=SEED),
        "ratio": float(np.median(r_vals)),
        "ratio_ci": bootstrap_median(r_vals, seed=SEED + 1),
        "evals_per_start": 24,
        "total_budget": 96,
        "total_evals_median": float(np.median(eval_totals)),
        "wall_time_sum_median": float(np.median(wall_totals)),
        "accepted": float(np.mean([bool(row.get("refinement_accepted")) for row in chosen])),
        "optimizer_success": float(np.mean([bool(row.get("optimizer_success")) for row in chosen])),
        "start_wins": start_wins,
        "reference_screened_map": {
            "E": 0.0411,
            "E_ci": [0.0334, 0.0506],
            "ratio": 0.1869,
            "ratio_ci": [0.1475, 0.2363],
            "evals": 7,
            "wall_s": 24.58,
            "accepted": 0.08,
        },
        "reference_fixed": {
            "E": 0.3414,
            "E_ci": [0.2415, 0.4415],
            "ratio": 0.4487,
            "ratio_ci": [0.3792, 0.5815],
            "evals": 79,
            "wall_s": 285.80,
            "accepted": 0.94,
        },
        "paper_rounding": {
            "E": "29.64%",
            "ratio": "33.78%",
            "evals": 96,
            "accepted": "92%",
        },
    }
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (BASE / "chosen_records.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
