#!/usr/bin/env python3
"""Regenerate (or verify) the group-level summary.json files of the reviewer
revision arms from their per-scene ``records.jsonl`` files.

Groups (all on the 18 locked CT scenes unless stated otherwise):

* ``constitutive_mismatch``  fiber / viscous observation laws
* ``structural_mismatch``    matched / two_inclusions / diffuse / sliding_bc
* ``ekf_sequential``         EKF baseline, noise 0 and 0.05 px
* ``decoupled_screen``       residual screen + MAP objective (R3 C2)
* ``population_prior``       population-mean prior centre (R4 C3)
* ``synthetic50_threshold_sweep``  50 synthetic test cases, early thresholds

Default mode is ``--check``: every summary is recomputed in memory and compared
value-by-value with the archived file; the exit status is non-zero on any
difference. ``--write`` overwrites the archived summaries. Records are never
modified. Medians follow the arithmetic convention (mean of the two central
values for even n); percentages are rounded to two decimals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "results" / "ct_reviewer_drivers_v6"
LM = ROOT / "results" / "lung_mesh_ai_v2"
LOCKED_18 = V6 / "paired_scene_ids.json"
BASELINE_18 = V6 / "ct_reviewer_drivers_v6" / "fem_paired" / "w2_s0p05_p2"
RESIDUAL_ONLY_18 = V6 / "ct_reviewer_drivers_v6" / "paired_controls" / "control_residual_only"
FIXED_5000_18 = V6 / "ct_reviewer_drivers_v6" / "paired_controls" / "control_learned_init_fixed_prior"
SIX_ARM_50 = LM / "synthetic50_full_paired_controls_20261008"
PROTOCOL_KEYS = (
    "cohort", "prior_width_scale", "screen_threshold", "accept_threshold", "probe_count",
    "initialization_mode", "prior_center_mode", "material_prior_weight", "screening_metric",
    "objective_metric", "max_nfev", "contrast_ratio_lock", "eq_correlation_rollback",
)
MEDIAN_CONVENTION = "Arithmetic mean of the two middle values for even sample sizes."


# ----------------------------------------------------------------------------- helpers
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_per_patient(rows: list[dict]) -> list[dict]:
    """Keep the last checkpoint row per patient (timing-only retries)."""
    return list({str(row["patient_id"]): row for row in rows}.values())


def med(values) -> float | None:
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    if n == 0:
        return None
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2 if n % 2 == 0 else ordered[n // 2]


def pct(value) -> float | None:
    return None if value is None else round(100.0 * value, 2)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def locked_ids() -> list[str]:
    ids = json.loads(LOCKED_18.read_text(encoding="utf-8"))
    if len(ids) != 18 or len(set(ids)) != 18:
        raise SystemExit(f"{LOCKED_18} must contain 18 unique scene IDs")
    return ids


def locked_50() -> list[str]:
    manifest = json.loads((SIX_ARM_50 / "learned_learned" / "run_manifest.json").read_text(encoding="utf-8"))
    ids = manifest["patient_ids"]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise SystemExit("six-arm learned_learned manifest must list 50 unique case IDs")
    return ids


def baseline_rows() -> dict[str, dict]:
    return {row["patient_id"]: row for row in latest_per_patient(load(BASELINE_18 / "records.jsonl"))}


# ----------------------------------------------------------------------------- constitutive
def constitutive_summary() -> dict:
    base = V6 / "constitutive_mismatch"
    summary = {
        "protocol": "Isotropic Neo-Hookean fitter; observations reprojected from finite-strain fiber or viscous forward solves at synthetic truth.",
        "baseline_background_percent": pct(med(r["E_background_relative_error"] for r in baseline_rows().values())),
        "baseline_accepted": f"{sum(bool(r['refinement_accepted']) for r in baseline_rows().values())}/{len(baseline_rows())}",
        "median_convention": MEDIAN_CONVENTION,
        "arms": {},
    }
    for law in ("fiber", "viscous"):
        records = base / law / "records.jsonl"
        rows = load(records)
        if len(rows) != 18 or len({row["patient_id"] for row in rows}) != 18:
            raise SystemExit(f"incomplete or duplicate constitutive records for {law}")
        summary["arms"][law] = {
            "n": len(rows),
            "unique": len({row["patient_id"] for row in rows}),
            "accepted": sum(bool(row["refinement_accepted"]) for row in rows),
            "background": pct(med(row["E_background_relative_error"] for row in rows)),
            "ratio": pct(med(row["inclusion_ratio_relative_error"] for row in rows)),
            "force": pct(med(row["force_scale_relative_error"] for row in rows)),
            "peak_displacement_gap": pct(med(row["peak_displacement_relative_gap"] for row in rows)),
            "records_sha256": sha(records),
            "source_records": rel(records),
        }
    return summary


# ----------------------------------------------------------------------------- structural
def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return round(wins / (len(pos) * len(neg)), 3)


def structural_summary() -> dict:
    base = V6 / "structural_mismatch"
    arms = {}
    rows_by_mode = {}
    for mode_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        rows = load(mode_dir / "records.jsonl")
        rows_by_mode[mode_dir.name] = rows
        if not rows:
            continue
        ids = [row["patient_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"duplicate ids in {mode_dir.name}")
        arms[mode_dir.name] = {
            "n": len(rows),
            "accepted": sum(1 for row in rows if row["refinement_accepted"]),
            "median_background_error_percent": pct(med(row["E_background_relative_error"] for row in rows)),
            "median_ratio_error_percent": pct(med(row["inclusion_ratio_relative_error"] for row in rows)),
            "median_force_error_percent": pct(med(row["force_scale_relative_error"] for row in rows)),
            "median_peak_displacement_gap_percent": pct(med(row["peak_displacement_relative_gap"] for row in rows)),
            "median_final_cost": round(med(row["cost"] for row in rows), 4),
            "median_initial_cost": round(med(row["initial_cost"] for row in rows), 4),
            "records_sha256": sha(mode_dir / "records.jsonl"),
        }
    matched_costs = [row["cost"] for row in rows_by_mode.get("matched", [])]
    detector = {}
    if matched_costs:
        ordered = sorted(matched_costs)
        threshold = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
        for mode, rows in rows_by_mode.items():
            if mode == "matched" or not rows:
                continue
            costs = [row["cost"] for row in rows]
            detector[mode] = {
                "auc_cost_vs_matched": _auc(costs, matched_costs),
                "flagged_at_matched_p90": sum(1 for cost in costs if cost > threshold),
                "n": len(costs),
            }
        detector["matched_p90_threshold"] = round(threshold, 4)
        detector["matched_self_flag_rate"] = sum(1 for cost in matched_costs if cost > threshold)
    return {
        "protocol": (
            "Same 18 locked CT test scenes. Observations regenerated with the production "
            "isotropic solver under a different truth structure; fitter is the production "
            "single-sphere isotropic screened MAP with ratio locked. 'matched' re-runs the "
            "unchanged truth through the identical code path as the control."
        ),
        "modes": {
            "matched": "single sphere, full posterior fixation (control)",
            "two_inclusions": "second stiff sphere of equal radius/contrast at the x-mirrored center",
            "diffuse": "Gaussian stiffness bump, sigma = inclusion radius, same peak contrast",
            "sliding_bc": "posterior face fixed only in z plus three in-plane pins; fitter keeps full fixation",
        },
        "arms": arms,
        "residual_mismatch_detector": detector,
    }


# ----------------------------------------------------------------------------- EKF
def ekf_summary(name: str) -> dict | None:
    records = V6 / name / "records.jsonl"
    rows = load(records)
    if not rows:
        return None
    batch = baseline_rows()
    paired = [(row, batch[row["patient_id"]]) for row in rows if row["patient_id"] in batch]
    deltas = [row["E_background_relative_error"] - ref["E_background_relative_error"] for row, ref in paired]
    better = sum(1 for delta in deltas if delta < -1e-9)
    worse = sum(1 for delta in deltas if delta > 1e-9)
    return {
        "protocol": (
            "EKF over log(E_bg), log(q) with the same learned initializer, same prior scales, "
            "ratio and predicted region locked as in the official-prior screened FEM. One "
            "measurement update per nonzero frame per load (five frames x four loads), "
            "multiview UV residuals, central-difference Jacobians, Huber-style innovation "
            "clipping. No screening or rollback."
        ),
        "n": len(rows),
        "observation_noise_px": rows[0].get("observation_noise_px", 0.0),
        "median_background_error_percent": pct(med(row["E_background_relative_error"] for row in rows)),
        "median_ratio_error_percent": pct(med(row["inclusion_ratio_relative_error"] for row in rows)),
        "median_force_error_percent": pct(med(row["force_scale_relative_error"] for row in rows)),
        "median_fem_solves": med(row["fem_solves"] for row in rows),
        "paired_vs_batch_w2_s0p05_p2": {
            "n_paired": len(paired),
            "better": better,
            "worse": worse,
            "tied": len(paired) - better - worse,
            "median_delta_pp": pct(med(deltas)) if deltas else None,
            "batch_median_background_error_percent": pct(med(ref["E_background_relative_error"] for _, ref in paired)),
            "batch_median_evaluations": med(ref["function_evaluations"] for _, ref in paired),
        },
        "records_sha256": sha(records),
    }


# ----------------------------------------------------------------------------- FEM arm groups
def arm_block(label: str, run_dir: Path, expected_ids: set[str], extra: dict | None = None) -> dict:
    records = run_dir / "records.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    rows = latest_per_patient(load(records))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = {row["patient_id"] for row in rows}
    block = {
        "label": label,
        "n": len(rows),
        "unique_ids": len(ids),
        "ids_equal_locked_set": ids == expected_ids,
        "complete": ids == expected_ids and len(rows) == len(expected_ids),
        "accepted": sum(bool(row.get("refinement_accepted")) for row in rows),
        "continuation_stopped_screening_rejected_flag": sum(bool(row.get("screening_rejected")) for row in rows),
        "stopped_and_accepted": sum(bool(row.get("screening_rejected")) and bool(row.get("refinement_accepted")) for row in rows),
        "eq_correlation_rejected": sum(bool(row.get("eq_correlation_rejected")) for row in rows),
        "ratio_locked": sum(bool(row.get("lock_ratio")) for row in rows),
        "median_background_error_percent": pct(med(row["E_background_relative_error"] for row in rows)),
        "median_ratio_error_percent": pct(med(row["inclusion_ratio_relative_error"] for row in rows)),
        "median_force_error_percent": pct(med(row["force_scale_relative_error"] for row in rows)),
        "mean_background_error_percent": round(100.0 * statistics.fmean(row["E_background_relative_error"] for row in rows), 2),
        "mean_ratio_error_percent": round(100.0 * statistics.fmean(row["inclusion_ratio_relative_error"] for row in rows), 2),
        "median_function_evaluations": med(row["function_evaluations"] for row in rows),
        "median_wall_time_seconds": round(med(row["wall_time_seconds"] for row in rows), 1),
        "total_wall_time_seconds": round(sum(row["wall_time_seconds"] for row in rows), 1),
        "protocol": {key: manifest[key] for key in PROTOCOL_KEYS},
        "records_sha256": sha(records),
        "run_manifest_sha256": sha(manifest_path),
        "source_records": rel(records),
    }
    if extra:
        block.update(extra)
    return block


def paired_block(run_a: Path, run_b: Path) -> dict:
    a = {row["patient_id"]: row["E_background_relative_error"] for row in latest_per_patient(load(run_a / "records.jsonl"))}
    b = {row["patient_id"]: row["E_background_relative_error"] for row in latest_per_patient(load(run_b / "records.jsonl"))}
    ids = sorted(set(a) & set(b))
    deltas = [a[i] - b[i] for i in ids]
    return {
        "n_paired": len(ids),
        "median_of_paired_differences_pp": pct(med(deltas)),
        "difference_of_medians_pp": pct(med(a[i] for i in ids) - med(b[i] for i in ids)),
        "scenes_improved": sum(1 for d in deltas if d < -1e-8),
        "scenes_worse": sum(1 for d in deltas if d > 1e-8),
        "scenes_unchanged": sum(1 for d in deltas if abs(d) <= 1e-8),
    }


def decoupled_summary() -> dict:
    ids = set(locked_ids())
    arm_dir = V6 / "decoupled_screen" / "residual_screen_map_objective"
    return {
        "protocol": (
            "Paired 18 locked CT test scenes (6 geometries x 3 scenarios). Decoupled screening ablation for "
            "Reviewer 3 C2: the first-round continuation screen is scored on the measurement/physics residual "
            "only (no material or force prior terms), while the optimisation objective and the final 5% "
            "acceptance remain the complete MAP objective with the learned prior centre, width multiplier 2, "
            "weight 0.05, 2 probes, 48 evaluations. Implemented via the new --objective-metric flag of "
            "experiments/run_screened_fem_locked.py (default keeps the historical screen=objective coupling)."
        ),
        "reviewer_item": "R3C2",
        "baseline_reference": arm_block("baseline w2_s0p05_p2 learned/learned (map screen, map objective)", BASELINE_18, ids),
        "existing_residual_only_control": arm_block("residual screen / residual objective (existing control)", RESIDUAL_ONLY_18, ids),
        "arms": {"residual_screen_map_objective": arm_block("residual screen / MAP objective", arm_dir, ids)},
        "paired_vs_baseline": {"residual_screen_map_objective": paired_block(arm_dir, BASELINE_18)},
        "paired_vs_residual_only": {"residual_screen_map_objective": paired_block(arm_dir, RESIDUAL_ONLY_18)},
    }


def population_summary() -> dict:
    ids = set(locked_ids())
    base = V6 / "population_prior"
    arm_dir = base / "learned_init_population_center"
    population = json.loads((base / "population_prior_E.json").read_text(encoding="utf-8"))
    return {
        "protocol": (
            "Paired 18 locked CT test scenes. Population-prior-centred FEM control for Reviewer 4 C3: "
            "initialisation stays learned, but the Gaussian material-prior centre on log E_background is the "
            "training-split mean of true E_background (4425.17 Pa; median 3709.79 Pa) instead of the per-scene "
            "learned estimate; all other settings equal the baseline arm."
        ),
        "reviewer_item": "R4C3",
        "population_prior": population,
        "baseline_reference": arm_block("baseline w2_s0p05_p2 learned/learned (map screen, map objective)", BASELINE_18, ids),
        "existing_fixed_5000_control": arm_block("learned init / fixed 5000 Pa prior centre (existing control)", FIXED_5000_18, ids),
        "arms": {"learned_init_population_center": arm_block("learned init / population-mean prior centre", arm_dir, ids)},
        "paired_vs_baseline": {"learned_init_population_center": paired_block(arm_dir, BASELINE_18)},
        "paired_vs_fixed_5000": {"learned_init_population_center": paired_block(arm_dir, FIXED_5000_18)},
    }


def threshold_sweep_summary() -> dict:
    ids = locked_50()
    base = LM / "synthetic50_threshold_sweep"
    reference = SIX_ARM_50 / "learned_learned"
    arms = {}
    paired = {}
    for name, threshold in (("s0p00", 0.0), ("s0p01", 0.01), ("s0p025", 0.025), ("s0p10", 0.1), ("s0p20", 0.2)):
        run_dir = base / name
        arms[name] = arm_block(f"threshold {threshold:g}", run_dir, set(ids), {"screen_threshold": threshold})
        paired[name] = paired_block(run_dir, reference)
    return {
        "protocol": (
            "50 synthetic test cases (same IDs as the six-arm matched-control matrix). Early continuation "
            "threshold sweep for Reviewer 1 C6 / Reviewer 4 C9 using the learned/learned six-arm flags (prior "
            "width 2, weight 0.05, 2 probes, 48 evaluations, contrast ratio lock 1.2, E-q correlation rollback "
            "0.85). Final acceptance threshold stays 0.05 in all arms; threshold 0 disables the first-round "
            "continuation screen entirely. The screening_rejected flag means continuation-stopped after round "
            "one, which is not disjoint from final acceptance."
        ),
        "reviewer_item": "R1C6/R4C9",
        "locked_patient_ids_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
        "existing_threshold_0p05_reference": arm_block("existing six-arm learned/learned (threshold 0.05)", reference, set(ids), {"screen_threshold": 0.05}),
        "arms": arms,
        "paired_vs_threshold_0p05": paired,
    }


def six_arm_summary() -> dict:
    """50-case six-arm matched-control matrix (summary_50patient_paired.json)."""
    import numpy as np

    arm_names = ("learned_learned", "fixed_fixed", "fixed_learned", "learned_fixed", "no_material_prior", "residual_only")
    metrics = ("E_background_relative_error", "inclusion_ratio_relative_error", "force_scale_relative_error")
    ids = locked_50()
    replicates, seed = 20000, 20261008
    arm_rows: dict[str, dict[str, dict]] = {}
    arms: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for arm in arm_names:
        records = SIX_ARM_50 / arm / "records.jsonl"
        rows = latest_per_patient(load(records))
        if {row["patient_id"] for row in rows} != set(ids):
            raise SystemExit(f"{arm}: case IDs do not equal the locked 50-case set")
        arm_rows[arm] = {row["patient_id"]: row for row in rows}
        hashes[arm] = sha(records)
        arms[arm] = {
            "n": len(rows),
            "accepted": sum(bool(row.get("refinement_accepted")) for row in rows),
            "screen_rejected": sum(bool(row.get("screening_rejected")) for row in rows),
            "median_function_evaluations": float(np.median([row["function_evaluations"] for row in rows])),
            "median_wall_time_seconds": float(np.median([row["wall_time_seconds"] for row in rows])),
            "median": {metric: float(np.median([row[metric] for row in rows])) for metric in metrics},
            "mean": {metric: float(np.mean([row[metric] for row in rows])) for metric in metrics},
        }
    reference = arm_rows["learned_learned"]
    comparisons: dict[str, dict] = {}
    rng = np.random.default_rng(seed)  # one stream shared over arms x metrics, in this order
    for arm in arm_names[1:]:
        current = arm_rows[arm]
        by_metric = {}
        for metric in metrics:
            deltas = np.asarray([current[i][metric] - reference[i][metric] for i in ids], dtype=float)
            sampled = np.median(rng.choice(deltas, size=(replicates, len(deltas)), replace=True), axis=1)
            by_metric[metric] = {
                "median_delta_pp": 100.0 * float(np.median(deltas)),
                "mean_delta_pp": 100.0 * float(np.mean(deltas)),
                "paired_delta_q05_q50_q95_pp": [100.0 * float(v) for v in np.quantile(deltas, [0.05, 0.5, 0.95])],
                "median_delta_bootstrap_95ci_pp": [100.0 * float(v) for v in np.quantile(sampled, [0.025, 0.975])],
                "lower_error_count": int(np.sum(deltas < 0)),
                "higher_error_count": int(np.sum(deltas > 0)),
                "tied_count": int(np.sum(deltas == 0)),
            }
        comparisons[f"{arm}_minus_learned_learned"] = {"n": len(ids), "by_metric": by_metric}
    return {
        "cohort": "synthetic50",
        "patient_count": len(ids),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "arms": arms,
        "paired_comparisons": comparisons,
        "input_sha256": hashes,
    }


GROUPS = {
    "six_arm_synthetic50": (SIX_ARM_50 / "summary_50patient_paired.json", six_arm_summary),
    "constitutive_mismatch": (V6 / "constitutive_mismatch" / "summary.json", constitutive_summary),
    "structural_mismatch": (V6 / "structural_mismatch" / "summary.json", structural_summary),
    "ekf_sequential": (V6 / "ekf_sequential" / "summary.json", lambda: ekf_summary("ekf_sequential")),
    "ekf_sequential_noise0p05": (V6 / "ekf_sequential_noise0p05" / "summary.json", lambda: ekf_summary("ekf_sequential_noise0p05")),
    "decoupled_screen": (V6 / "decoupled_screen" / "summary.json", decoupled_summary),
    "population_prior": (V6 / "population_prior" / "summary.json", population_summary),
    "synthetic50_threshold_sweep": (LM / "synthetic50_threshold_sweep" / "summary.json", threshold_sweep_summary),
}


def diff_paths(expected, actual, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                out.append(f"{prefix}/{key}: {'missing in archived' if key not in expected else 'missing in regenerated'}")
            else:
                out.extend(diff_paths(expected[key], actual[key], f"{prefix}/{key}"))
        return out
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        out = []
        for index, (e, a) in enumerate(zip(expected, actual)):
            out.extend(diff_paths(e, a, f"{prefix}[{index}]"))
        return out
    return [] if expected == actual else [f"{prefix}: archived={expected!r} regenerated={actual!r}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="*", default=list(GROUPS), choices=list(GROUPS), help="subset of groups")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify archived summaries (default)")
    mode.add_argument("--write", action="store_true", help="overwrite archived summaries from records")
    args = parser.parse_args()
    failures = 0
    for name in args.groups:
        target, builder = GROUPS[name]
        regenerated = builder()
        if regenerated is None:
            print(f"[skip]  {name}: no records")
            continue
        if args.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(regenerated, indent=2), encoding="utf-8")
            print(f"[write] {name}: {target.relative_to(ROOT)}")
            continue
        archived = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        differences = diff_paths(archived, regenerated)
        if differences:
            failures += 1
            print(f"[DIFF]  {name}: {len(differences)} difference(s)")
            for line in differences[:20]:
                print("        ", line)
        else:
            print(f"[ok]    {name}: archived summary equals regeneration from records ({target.relative_to(ROOT)})")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
