"""Aggregate parallel patient chunks with patient-level bootstrap intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sim_lung_v2"


def bootstrap_median(values: np.ndarray, seed: int = 2026) -> list[float]:
    generator = np.random.default_rng(seed)
    draws = np.asarray(
        [
            np.median(generator.choice(values, size=len(values), replace=True))
            for _ in range(5000)
        ]
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(records: list[dict]) -> dict:
    E_errors = np.asarray([row["E_background_relative_error"] for row in records])
    ratio_errors = np.asarray(
        [row["inclusion_ratio_relative_error"] for row in records]
    )
    force_errors = np.asarray(
        [row.get("force_scale_relative_error", np.nan) for row in records],
        dtype=float,
    )
    finite_force = force_errors[np.isfinite(force_errors)]
    return {
        "test_patient_count": len(records),
        "E_background_median_relative_error": float(np.median(E_errors)),
        "E_background_median_bootstrap_95_ci": bootstrap_median(E_errors, 2026),
        "inclusion_ratio_median_relative_error": float(np.median(ratio_errors)),
        "inclusion_ratio_median_bootstrap_95_ci": bootstrap_median(
            ratio_errors, 2027
        ),
        "force_scale_median_relative_error": (
            float(np.median(finite_force)) if len(finite_force) else None
        ),
        "optimizer_success_rate": float(
            np.mean(
                [
                    bool(row.get("converged", row.get("optimizer_success", False)))
                    for row in records
                ]
            )
        ),
        "function_evaluations_median": float(
            np.median([row.get("function_evaluations", np.nan) for row in records])
        ),
        "wall_time_seconds_median": float(
            np.median([row.get("wall_time_seconds", np.nan) for row in records])
        ),
    }


def paired_statistics(reference: list[dict], candidate: list[dict]) -> dict:
    reference_by_id = {row["patient_id"]: row for row in reference}
    candidate_by_id = {row["patient_id"]: row for row in candidate}
    patient_ids = sorted(set(reference_by_id) & set(candidate_by_id))
    if not patient_ids:
        raise ValueError("Paired comparison has no shared patient IDs")
    output: dict[str, object] = {"paired_patient_count": len(patient_ids)}
    for metric in (
        "E_background_relative_error",
        "inclusion_ratio_relative_error",
        "force_scale_relative_error",
        "wall_time_seconds",
        "function_evaluations",
    ):
        pairs = [
            (reference_by_id[key].get(metric), candidate_by_id[key].get(metric))
            for key in patient_ids
        ]
        differences = np.asarray(
            [
                float(candidate_value) - float(reference_value)
                for reference_value, candidate_value in pairs
                if reference_value is not None
                and candidate_value is not None
                and np.isfinite(reference_value)
                and np.isfinite(candidate_value)
            ]
        )
        if len(differences):
            output[metric] = {
                "candidate_minus_reference_median": float(np.median(differences)),
                "paired_bootstrap_95_ci": bootstrap_median(
                    differences, 3100 + len(output)
                ),
                "candidate_win_rate": float(np.mean(differences < 0)),
            }
    reference_converged = np.asarray(
        [
            bool(
                reference_by_id[key].get(
                    "converged", reference_by_id[key].get("optimizer_success", False)
                )
            )
            for key in patient_ids
        ]
    )
    candidate_converged = np.asarray(
        [
            bool(
                candidate_by_id[key].get(
                    "converged", candidate_by_id[key].get("optimizer_success", False)
                )
            )
            for key in patient_ids
        ]
    )
    output["convergence"] = {
        "candidate_minus_reference_rate": float(
            candidate_converged.mean() - reference_converged.mean()
        ),
        "discordant_candidate_only": int((candidate_converged & ~reference_converged).sum()),
        "discordant_reference_only": int((reference_converged & ~candidate_converged).sum()),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESULTS / "metrics_large_noisy.json")
    parser.add_argument(
        "--reference-method",
        help="Method label used as the paired-statistics reference",
    )
    args = parser.parse_args()
    records_by_method: dict[str, list[dict]] = {}
    for path in args.inputs:
        result = json.loads(path.read_text(encoding="utf-8"))
        method = result.get("benchmark_method") or result.get("method") or "default"
        records_by_method.setdefault(method, []).extend(
            [{**row, "benchmark_method": method} for row in result["records"]]
        )
    for method, rows in records_by_method.items():
        unique = {row["patient_id"]: row for row in rows}
        records_by_method[method] = [unique[key] for key in sorted(unique)]
    methods = sorted(records_by_method)
    records = [row for method in methods for row in records_by_method[method]]
    result = {
        "protocol": "large held-out-patient noisy 3-D surface motion",
        "methods": {
            method: summarize(records_by_method[method]) for method in methods
        },
        "records": records,
    }
    if len(methods) == 1:
        result.update(result["methods"][methods[0]])
    if len(methods) > 1:
        reference_method = args.reference_method or methods[0]
        if reference_method not in records_by_method:
            raise ValueError(f"Unknown reference method: {reference_method}")
        result["paired_statistics"] = {
            method: paired_statistics(
                records_by_method[reference_method], records_by_method[method]
            )
            for method in methods
            if method != reference_method
        }
        result["reference_method"] = reference_method
        reference_nfev = result["methods"][reference_method][
            "function_evaluations_median"
        ]
        reductions = {}
        for method in methods:
            if method == reference_method:
                continue
            candidate_nfev = result["methods"][method][
                "function_evaluations_median"
            ]
            reductions[method] = (
                float((reference_nfev - candidate_nfev) / reference_nfev)
                if reference_nfev > 0
                else None
            )
        result["evaluation_reduction_by_method"] = reductions
        if len(reductions) == 1:
            result["evaluation_reduction"] = next(iter(reductions.values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
