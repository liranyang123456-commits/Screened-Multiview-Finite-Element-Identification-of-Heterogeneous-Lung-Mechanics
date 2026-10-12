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

Second-revision groups (verified field-by-field against the shipped records,
run manifests and per-arm summaries; see ``VERIFY_GROUPS``):

* ``ekf_sequential_ct120_all`` / ``ekf_sequential_synthetic50`` (+ the
  superseded first synthetic run)  cohort-level sequential EKF
* ``model_mismatch_n48``     camera-yaw arms at the common 48-evaluation budget
* ``assumption_sensitivity_fixed``  corrected ellipsoid-aspect arms
* ``synthetic50_decoupled_residual``  residual screen + residual acceptance, 50 cases
* ``mismatch_initializer_reinference``  learned initializer on mismatched observations
* ``accepted_update_harm``, ``initializer_reference_18``,
  ``calibration_fourbin_test_sensitivity``, ``ct120_audit_mechanics``,
  ``aspect_arm_sanity``  record-only audits

Default mode is ``--check``: every summary is recomputed in memory and compared
value-by-value with the archived file; the exit status is non-zero on any
difference. ``--write`` overwrites the archived summaries (regeneration groups
only). Records are never modified. Medians follow the arithmetic convention
(mean of the two central values for even n); percentages are rounded to two
decimals. For the verify groups, fields that need the generated dataset
tensors or model checkpoints (occupancy node sets, prose conclusions,
wall-clock stamps) are listed as "not recomputable" and are not compared;
every other field, including every SHA-256 digest, is recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
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


# ============================================================================ second-revision groups
# These group summaries were produced by the drivers / audit runs of the second
# revision.  The builders below recompute every record-derived field from the
# shipped records.jsonl / run_manifest.json / per-arm summary.json files (and the
# dataset manifests) so that the archived file can be verified without FEM.
CT_MANIFEST = ROOT / "dataset" / "ion_ct_synthetic_mechanics540" / "manifest.json"
SYN_MANIFEST = ROOT / "dataset" / "sim_lung_ai_v2_multiview250" / "manifest.json"
CT120_AUDIT = ROOT / "results" / "ion_ct_official_prior_screened_fem120_v1"
CT120_BENCHMARK = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
LOCKED50_V1 = LM / "screened_fem_locked50_v1"
FROZEN_PRIMARY_METRICS = LM / "fem_multiview_screened50_strict" / (
    "metrics_image_tracks_force_map_sigma_0.05_px_0.05_nodes_allframes_temporalreg_smooth2_"
    "multiview_conf0.20_regionweight8.0_frozen50_screen05_strict_gnn_init_predicted_region.json"
)
CALIBRATE_V2 = ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2"
W1_ARM_18 = ROOT / "results" / "ct_reviewer_drivers_v1" / "recovered_raw" / "fem_paired" / "w1_s0p05_p2"
MM_COMPLETED = V6 / "model_mismatch_completed"
MM_DRV = V6 / "ct_reviewer_drivers_v6" / "model_mismatch"
DECOUPLED_50 = LM / "synthetic50_decoupled_residual"
REINFER = V6 / "mismatch_initializer_reinference"
ARM_MEDIAN_CONVENTION = "arithmetic: sorted, mean of the two middle values for even n"
ERROR_KEYS = ("E_background_relative_error", "inclusion_ratio_relative_error", "force_scale_relative_error")


def _rel_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows_by_id(path: Path) -> dict[str, dict]:
    return {str(row["patient_id"]): row for row in latest_per_patient(load(path))}


def _init_error(row: dict) -> float:
    return abs(float(row["initial_E_background"]) - float(row["E_background_true"])) / float(row["E_background_true"])


def _counts(values) -> dict:
    return {str(key): count for key, count in sorted(Counter(values).items(), key=lambda item: (str(type(item[0])), item[0]))}


def upper_median(values) -> float:
    """sorted(values)[n // 2]: the convention used by aspect_arm_sanity.py and by the
    inner-LOO selection statistic of run_ct_reviewer_drivers.py (NOT the arithmetic
    median used for every reported endpoint)."""
    ordered = sorted(float(value) for value in values)
    return ordered[len(ordered) // 2]


# ----------------------------------------------------------------------------- cohort-level EKF
def _ekf_paired(ekf: dict[str, dict], batch_rows: list[dict], tie_tol: float = 1e-9) -> dict:
    """Mirror of run_ekf_sequential_ct18._paired (kept torch-free here)."""
    batch = {str(r["patient_id"]): r for r in batch_rows}
    common = sorted(set(ekf) & set(batch))
    result: dict = {"n_paired": len(common), "n_batch": len(batch), "endpoints": {}}
    for key in ERROR_KEYS:
        deltas = []
        better = worse = tied = 0
        for pid in common:
            if key not in ekf[pid] or key not in batch[pid]:
                continue
            d = float(ekf[pid][key]) - float(batch[pid][key])
            deltas.append(100.0 * d)
            if abs(d) <= tie_tol:
                tied += 1
            elif d < 0:
                better += 1
            else:
                worse += 1
        result["endpoints"][key] = {
            "better": better,
            "worse": worse,
            "tied": tied,
            "median_paired_delta_pp": round(med(deltas), 4) if deltas else None,
            "ekf_median_percent": round(100.0 * med(float(ekf[p][key]) for p in common if key in ekf[p]), 4),
            "batch_median_percent": round(100.0 * med(float(batch[p][key]) for p in common if key in batch[p]), 4),
        }
    evals = [float(batch[p]["function_evaluations"]) for p in common if "function_evaluations" in batch[p]]
    result["batch_median_function_evaluations"] = med(evals) if evals else None
    checks = {"lock_ratio_equal": 0, "initial_E_equal": 0, "initial_ratio_equal": 0, "locked_ratio_value_equal": 0, "n_locked_in_both": 0, "n_checked": 0}
    for pid in common:
        e, b = ekf[pid], batch[pid]
        if "lock_ratio" not in e or "lock_ratio" not in b:
            continue
        checks["n_checked"] += 1
        checks["lock_ratio_equal"] += int(bool(e["lock_ratio"]) == bool(b["lock_ratio"]))
        if bool(e["lock_ratio"]) and bool(b["lock_ratio"]) and "inclusion_ratio_estimated" in e and "inclusion_ratio_estimated" in b:
            checks["n_locked_in_both"] += 1
            checks["locked_ratio_value_equal"] += int(abs(float(e["inclusion_ratio_estimated"]) - float(b["inclusion_ratio_estimated"])) <= 1e-9)
        if "initial_E_background" in e and "initial_E_background" in b:
            checks["initial_E_equal"] += int(abs(float(e["initial_E_background"]) - float(b["initial_E_background"])) <= 1e-9 * max(1.0, abs(float(b["initial_E_background"]))))
        if "initial_ratio" in e and "initial_ratio" in b:
            checks["initial_ratio_equal"] += int(abs(float(e["initial_ratio"]) - float(b["initial_ratio"])) <= 1e-9)
    result["matched_protocol_checks"] = checks
    return result


def ekf_cohort_summary(out_dir: Path, cohort: str) -> dict | None:
    """Regenerate the cohort-level EKF summary written by
    ``run_ekf_sequential_ct18.py --summarize`` from the shard / merged records."""
    merged_path = out_dir / "records.jsonl"
    if not merged_path.exists():
        return None
    shard_paths = sorted(out_dir.glob("records_shard*.jsonl"))
    manifest = _read_json(CT_MANIFEST if cohort == "ct120" else SYN_MANIFEST)
    order = {str(r["patient_id"]): i for i, r in enumerate(r for r in manifest["patients"] if r["split"] == "test")}
    # the merged file must equal the de-duplicated shard union in test-split order
    if shard_paths:
        union: dict[str, dict] = {}
        for path in shard_paths:
            for row in load(path):
                union.setdefault(str(row["patient_id"]), row)
        expected_merge = [union[p] for p in sorted(union, key=lambda p: order.get(p, 10**9))]
        if expected_merge != load(merged_path):
            raise SystemExit(f"{merged_path} is not the de-duplicated union of its shard files")
    records = load(merged_path)
    ekf = {str(r["patient_id"]): r for r in records}
    expected = list(order)
    summary: dict = {
        "protocol": (
            "EKF over log(E_bg), log(q)" + (", log(ratio) when the ratio is not contrast-locked" if cohort == "synthetic50" else "")
            + " with the same learned initializer, same prior scales (prior_width_scale 2.0, force prior 0.05), "
            + ("ratio locked (official 2028+bins3) and predicted region locked as in the official-prior screened FEM. "
               if cohort == "ct120" else
               "ratio contrast-lock decision (threshold 1.20), initial ratio and prior centre identical to the six-arm learned/learned matrix; "
               "unlocked ratio gets prior SD = material_prior_sigma[1] = max(0.05, 2*log_ratio_std). ")
            + "One measurement update per nonzero frame per load (five frames x four loads), multiview UV residuals, "
            + "central-difference Jacobians, Huber-style innovation clipping. No screening or rollback."
        ),
        "cohort": cohort,
        "n": len(records),
        "unique_ids": len(ekf),
        "expected_n": len(expected),
        "missing_ids": sorted(set(expected) - set(ekf)),
        "unexpected_ids": sorted(set(ekf) - set(expected)),
        # duplicate_rows_dropped_in_merge depends on whether records.jsonl already
        # existed when --summarize ran; the merge identity is asserted above instead.
        "shard_files": [p.name for p in shard_paths],
        "observation_noise_px": sorted({float(r.get("observation_noise_px", 0.0)) for r in records}),
        "ratio_locked_n": sum(bool(r.get("lock_ratio", True)) for r in records),
        "median_background_error_percent": round(100.0 * med(r["E_background_relative_error"] for r in records), 4),
        "median_ratio_error_percent": round(100.0 * med(r["inclusion_ratio_relative_error"] for r in records), 4),
        "median_force_error_percent": round(100.0 * med(r["force_scale_relative_error"] for r in records), 4),
        "median_fem_solves": med(float(r["fem_solves"]) for r in records),
        "paired_vs_batch": {},
        "records_sha256": sha(merged_path),
        "shard_records_sha256": {p.name: sha(p) for p in shard_paths},
    }
    references = (
        {"official_prior_screened_fem120_v1": CT120_AUDIT / "records.jsonl", "eighteen_scene_w2_s0p05_p2": BASELINE_18 / "records.jsonl"}
        if cohort == "ct120"
        else {"six_arm_learned_learned": SIX_ARM_50 / "learned_learned" / "records.jsonl", "frozen_primary_screened_map_locked50_v1": LOCKED50_V1 / "records.jsonl"}
    )
    for name, path in references.items():
        if not path.exists():
            summary["paired_vs_batch"][name] = {"missing": str(path)}
            continue
        comparison = _ekf_paired(ekf, load(path))
        comparison["records_path"] = _rel_posix(path)
        comparison["records_sha256"] = sha(path)
        summary["paired_vs_batch"][name] = comparison
    init_pairs = [(_init_error(r), float(r["E_background_relative_error"])) for r in records if "initial_E_background" in r and "E_background_true" in r]
    ratio_pairs = [
        (abs(float(r["initial_ratio"]) - float(r["inclusion_ratio_true"])) / float(r["inclusion_ratio_true"]), float(r["inclusion_ratio_relative_error"]))
        for r in records if not bool(r.get("lock_ratio", True)) and "initial_ratio" in r and "inclusion_ratio_true" in r
    ]

    def _vs_init(pairs: list[tuple[float, float]]) -> dict | None:
        if not pairs:
            return None
        deltas = [100.0 * (f - i) for i, f in pairs]
        return {
            "n": len(pairs),
            "ekf_better_than_initializer": sum(f < i for i, f in pairs),
            "ekf_worse_than_initializer": sum(f > i for i, f in pairs),
            "tied": sum(f == i for i, f in pairs),
            "median_initializer_error_percent": round(100.0 * med(i for i, _ in pairs), 4),
            "median_ekf_error_percent": round(100.0 * med(f for _, f in pairs), 4),
            "median_paired_delta_pp": round(med(deltas), 4),
        }

    summary["paired_vs_initializer"] = {
        "E_background_relative_error": _vs_init(init_pairs),
        "inclusion_ratio_relative_error_unlocked_only": _vs_init(ratio_pairs),
    }
    return summary


# ----------------------------------------------------------------------------- 18-scene FEM arm groups (second revision)
def _arm_block_v2(label: str, run_dir: Path, locked: set[str], per_scene: bool = True) -> dict:
    records = run_dir / "records.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary.json"
    rows = sorted(latest_per_patient(load(records)), key=lambda r: str(r["patient_id"]))
    manifest = _read_json(manifest_path)
    ids = {str(r["patient_id"]) for r in rows}
    block = {
        "label": label,
        "results_dir": _rel_posix(run_dir),
        "n": len(rows),
        "unique_ids_equal_locked_18": ids == locked,
        "manifest_patient_ids_equal_locked_18": {str(p) for p in (manifest.get("patient_ids") or [])} == locked,
        "max_nfev": manifest.get("max_nfev"),
        "pose_rotation_error_deg": manifest.get("pose_rotation_error_deg"),
        "region_aspect_xy": manifest.get("region_aspect_xy"),
        "accepted": sum(bool(r.get("refinement_accepted")) for r in rows),
        "screening_rejected": sum(bool(r.get("screening_rejected")) for r in rows),
        "eq_correlation_rejected": sum(bool(r.get("eq_correlation_rejected")) for r in rows),
        "median_convention": ARM_MEDIAN_CONVENTION,
        "E_background_median_relative_error": med(r["E_background_relative_error"] for r in rows),
        "inclusion_ratio_median_relative_error": med(r["inclusion_ratio_relative_error"] for r in rows),
        "force_scale_median_relative_error": med(r["force_scale_relative_error"] for r in rows),
        "function_evaluations_median": med(r["function_evaluations"] for r in rows),
        "wall_time_seconds_median": med(r["wall_time_seconds"] for r in rows),
        "termination_reasons": _counts(r.get("termination_reason") for r in rows),
        "records_sha256": sha(records),
        "run_manifest_sha256": sha(manifest_path),
    }
    if summary_path.exists():
        block["summary_json_sha256"] = sha(summary_path)
    if per_scene:
        block["per_scene"] = [
            {
                "patient_id": r["patient_id"],
                "E_background_estimated": r["E_background_estimated"],
                "E_background_relative_error": r["E_background_relative_error"],
                "inclusion_ratio_relative_error": r["inclusion_ratio_relative_error"],
                "force_scale_relative_error": r["force_scale_relative_error"],
                "refinement_accepted": bool(r.get("refinement_accepted")),
                "screening_rejected": bool(r.get("screening_rejected")),
                "function_evaluations": r["function_evaluations"],
                "termination_reason": r.get("termination_reason"),
            }
            for r in rows
        ]
    return block


def _paired_v2(new_dir: Path, ref_dir: Path, new_label: str, ref_label: str, convention: str) -> dict:
    new, ref = _rows_by_id(new_dir / "records.jsonl"), _rows_by_id(ref_dir / "records.jsonl")
    common = sorted(set(new) & set(ref))
    out: dict = {"new": new_label, "reference": ref_label, "n_common": len(common), "convention": convention}
    for key in (*ERROR_KEYS, "function_evaluations"):
        deltas = [float(new[p][key]) - float(ref[p][key]) for p in common]
        out[f"median_delta_{key}"] = med(deltas)
        out[f"mean_delta_{key}"] = statistics.fmean(deltas)
        out[f"n_improved_{key}"] = sum(d < 0 for d in deltas)
        out[f"n_worsened_{key}"] = sum(d > 0 for d in deltas)
        out[f"n_unchanged_{key}"] = sum(d == 0 for d in deltas)
    out["accepted_new"] = sum(bool(new[p].get("refinement_accepted")) for p in common)
    out["accepted_reference"] = sum(bool(ref[p].get("refinement_accepted")) for p in common)
    out["acceptance_changed_scenes"] = [p for p in common if bool(new[p].get("refinement_accepted")) != bool(ref[p].get("refinement_accepted"))]
    out["n_E_background_estimated_bitwise_identical"] = sum(new[p]["E_background_estimated"] == ref[p]["E_background_estimated"] for p in common)
    out["per_scene"] = [
        {
            "patient_id": p,
            "delta_E_background_relative_error": float(new[p]["E_background_relative_error"]) - float(ref[p]["E_background_relative_error"]),
            "delta_inclusion_ratio_relative_error": float(new[p]["inclusion_ratio_relative_error"]) - float(ref[p]["inclusion_ratio_relative_error"]),
            "delta_force_scale_relative_error": float(new[p]["force_scale_relative_error"]) - float(ref[p]["force_scale_relative_error"]),
            "delta_function_evaluations": float(new[p]["function_evaluations"]) - float(ref[p]["function_evaluations"]),
            "accepted_new": bool(new[p].get("refinement_accepted")),
            "accepted_old": bool(ref[p].get("refinement_accepted")),
            "E_background_estimated_identical": new[p]["E_background_estimated"] == ref[p]["E_background_estimated"],
        }
        for p in common
    ]
    return out


def model_mismatch_n48_summary() -> dict:
    locked = set(locked_ids())
    arms_48 = {
        "pose_yaw_m10": V6 / "model_mismatch_n48" / "pose_yaw_m10",
        "pose_yaw_p10": V6 / "model_mismatch_n48" / "pose_yaw_p10",
        "pose_yaw_m5": MM_DRV / "mismatch_pose_m5_baseline",
        "pose_yaw_p5": MM_DRV / "mismatch_pose_p5_baseline",
    }
    arms_16 = {
        "pose_yaw_m10": MM_COMPLETED / "pose_yaw_m10_v2",
        "pose_yaw_p10": MM_COMPLETED / "pose_yaw_p10_v1",
        "pose_yaw_m5": MM_COMPLETED / "pose_yaw_m5_v1",
        "pose_yaw_p5": MM_COMPLETED / "pose_yaw_p5_v1",
    }
    return {
        "task": "yaw mismatch at common 48-evaluation budget on the 18 locked CT scenes",
        "locked_scene_ids_source": _rel_posix(LOCKED_18),
        "locked_scene_ids_sha256": sha(LOCKED_18),
        "arms_48": {name: _arm_block_v2(name, path, locked) for name, path in arms_48.items()},
        "arms_16_reference": {name: _arm_block_v2(f"{name}_n16", path, locked) for name, path in arms_16.items()},
        "paired_48_vs_16": {
            name: _paired_v2(arms_48[name], arms_16[name], f"{name}@48", f"{name}@16",
                             "delta = new(48 evals) - reference(16 evals), per scene; medians arithmetic")
            for name in arms_48
        },
        "baseline_reference": _arm_block_v2("baseline_w2_s0p05_p2", BASELINE_18, locked, per_scene=False),
        "paired_48_vs_baseline": {
            name: _paired_v2(arms_48[name], BASELINE_18, f"{name}@48", "baseline@48",
                             "delta = yaw arm(48 evals) - unperturbed baseline(48 evals), per scene; medians arithmetic")
            for name in arms_48
        },
    }


def assumption_fixed_summary() -> dict:
    locked = set(locked_ids())
    base = _rows_by_id(BASELINE_18 / "records.jsonl")
    arms, vs_baseline, vs_archived = {}, {}, {}
    for name in ("aspect_1p5", "aspect_2p0"):
        run_dir = V6 / "assumption_sensitivity_fixed" / name
        arms[name] = _arm_block_v2(name, run_dir, locked)
        rows = _rows_by_id(run_dir / "records.jsonl")
        common = sorted(set(rows) & set(base))
        identical = [p for p in common if rows[p]["E_background_estimated"] == base[p]["E_background_estimated"]]
        shift = [
            {
                "patient_id": p,
                "initial_cost_relative_change": (float(rows[p]["initial_cost"]) - float(base[p]["initial_cost"])) / float(base[p]["initial_cost"]),
                "final_cost_relative_change": (float(rows[p]["cost"]) - float(base[p]["cost"])) / float(base[p]["cost"]),
                "function_evaluations_arm": rows[p]["function_evaluations"],
                "function_evaluations_baseline": base[p]["function_evaluations"],
                "force_scale_estimated_arm": rows[p]["force_scale_estimated"],
                "force_scale_estimated_baseline": base[p]["force_scale_estimated"],
            }
            for p in common
        ]
        vs_baseline[name] = {
            "n_common": len(common),
            "n_E_background_estimated_bitwise_identical_to_baseline": len(identical),
            "identical_scenes": identical,
            "differing_scenes": [p for p in common if p not in set(identical)],
            "median_delta_E_background_relative_error": med(float(rows[p]["E_background_relative_error"]) - float(base[p]["E_background_relative_error"]) for p in common),
            "n_cost_identical_to_baseline": sum(rows[p]["cost"] == base[p]["cost"] for p in common),
            "n_initial_cost_identical_to_baseline": sum(rows[p]["initial_cost"] == base[p]["initial_cost"] for p in common),
            "n_function_evaluations_identical_to_baseline": sum(rows[p]["function_evaluations"] == base[p]["function_evaluations"] for p in common),
            "n_force_scale_estimated_identical_to_baseline": sum(rows[p]["force_scale_estimated"] == base[p]["force_scale_estimated"] for p in common),
            "n_accepted_identical_to_baseline": sum(bool(rows[p].get("refinement_accepted")) == bool(base[p].get("refinement_accepted")) for p in common),
            "per_scene_objective_shift_vs_baseline": shift,
            "max_abs_initial_cost_relative_change": max(abs(s["initial_cost_relative_change"]) for s in shift),
            "max_abs_final_cost_relative_change": max(abs(s["final_cost_relative_change"]) for s in shift),
            # occupancy_* fields need the dataset tensors (node coordinates) and are not recomputed here
        }
        archived_dir = V6 / "assumption_sensitivity" / name
        archived = _rows_by_id(archived_dir / "records.jsonl")
        common_arch = sorted(set(rows) & set(archived))
        vs_archived[name] = {
            "archived_dir": _rel_posix(archived_dir),
            "archived_records_sha256": sha(archived_dir / "records.jsonl"),
            "n_common": len(common_arch),
            "n_E_background_estimated_bitwise_identical_to_archived": sum(rows[p]["E_background_estimated"] == archived[p]["E_background_estimated"] for p in common_arch),
            "archived_E_background_median_relative_error": med(r["E_background_relative_error"] for r in archived.values()),
            "archived_n_E_background_estimated_bitwise_identical_to_baseline": sum(archived[p]["E_background_estimated"] == base[p]["E_background_estimated"] for p in sorted(set(archived) & set(base))),
        }
    return {
        "task": "ellipsoid-aspect arms with region attention always using the predicted centre/radius",
        "locked_scene_ids_sha256": sha(LOCKED_18),
        "baseline": _arm_block_v2("baseline_w2_s0p05_p2", BASELINE_18, locked, per_scene=False),
        "arms": arms,
        "vs_baseline": vs_baseline,
        "vs_archived_defective_arm": vs_archived,
    }


# ----------------------------------------------------------------------------- decoupled residual screen / acceptance (50 cases)
def _decoupled_block(rows: list[dict], label: str, ids50: list[str]) -> dict:
    rows = sorted(rows, key=lambda r: str(r["patient_id"]))
    pids = [str(r["patient_id"]) for r in rows]
    accepted = [r for r in rows if bool(r.get("refinement_accepted"))]
    continued = [r for r in rows if not bool(r.get("screening_rejected"))]
    init_all = [(_init_error(r), float(r["E_background_relative_error"])) for r in rows]
    init_acc = [(_init_error(r), float(r["E_background_relative_error"])) for r in accepted]

    def _vs(pairs: list[tuple[float, float]]) -> dict:
        return {
            "n": len(pairs),
            "worse_than_initializer": sum(f > i for i, f in pairs),
            "better_than_initializer": sum(f < i for i, f in pairs),
            "tied": sum(f == i for i, f in pairs),
        }

    block: dict = {
        "label": label,
        "n": len(rows),
        "unique_ids": len(set(pids)),
        "unique_ids_equal_50": set(pids) == set(ids50),
        "accepted": len(accepted),
        "screening_rejected": sum(bool(r.get("screening_rejected")) for r in rows),
        "continued_not_screening_rejected": len(continued),
        "continued_but_rolled_back_at_final_acceptance": sum(not bool(r.get("refinement_accepted")) for r in continued),
        "eq_correlation_rejected": sum(bool(r.get("eq_correlation_rejected")) for r in rows),
        "lock_ratio_count": sum(bool(r.get("lock_ratio")) for r in rows),
        "termination_reason_counts": _counts(r.get("termination_reason") for r in rows),
        "function_evaluations_counts": {str(k): v for k, v in sorted(Counter(int(r["function_evaluations"]) for r in rows).items())},
        "median_function_evaluations": med(r["function_evaluations"] for r in rows),
        "median_wall_time_seconds": med(r["wall_time_seconds"] for r in rows),
        "median_relative_error": {
            "E_background": med(r["E_background_relative_error"] for r in rows),
            "inclusion_ratio": med(r["inclusion_ratio_relative_error"] for r in rows),
            "force_scale": med(r["force_scale_relative_error"] for r in rows),
        },
        "mean_relative_error": {
            "E_background": statistics.fmean(r["E_background_relative_error"] for r in rows),
            "inclusion_ratio": statistics.fmean(r["inclusion_ratio_relative_error"] for r in rows),
            "force_scale": statistics.fmean(r["force_scale_relative_error"] for r in rows),
        },
        "initializer_only_median_E_background_error": med(i for i, _ in init_all),
        "all_cases_vs_initializer": _vs(init_all),
        "accepted_set": {
            **_vs(init_acc),
            "median_initializer_E_background_error": med(i for i, _ in init_acc),
            "median_final_E_background_error": med(f for _, f in init_acc),
            "median_paired_delta_final_minus_initializer": med(f - i for i, f in init_acc),
            "ids": sorted(str(r["patient_id"]) for r in accepted),
        },
    }
    if any("acceptance_metric" in r for r in rows):
        block["acceptance_metric_counts"] = _counts(r.get("acceptance_metric") for r in rows)
        acc_red = sorted(float(r["acceptance_relative_cost_reduction"]) for r in continued)
        map_red = sorted(float(r["relative_cost_reduction"]) for r in continued)
        block["acceptance_relative_cost_reduction_continued_min_median_max"] = [acc_red[0], med(acc_red), acc_red[-1]]
        block["map_relative_cost_reduction_continued_min_median_max"] = [map_red[0], med(map_red), map_red[-1]]
        block["continued_cases_where_map_and_residual_acceptance_disagree"] = [
            str(r["patient_id"]) for r in continued
            if (float(r["relative_cost_reduction"]) >= 0.05) != (float(r["acceptance_relative_cost_reduction"]) >= 0.05)
        ]
    if any("objective_metric" in r for r in rows):
        block["objective_metric_counts"] = _counts(r.get("objective_metric") for r in rows)
    if any("screening_metric" in r for r in rows):
        block["screening_metric_counts"] = _counts(r.get("screening_metric") for r in rows)
    return block


DECOUPLED_LABELS = {
    "primary_like_w1_n96": "A1 primary-like: width 1.0, 96 evals, residual screen, MAP objective, residual acceptance",
    "sixarm_like_w2_n48": "A2 six-arm-like: width 2.0, 48 evals, residual screen, MAP objective, residual acceptance",
    "map_w1_n96": "A3 plain learned/learned at width 1.0, 96 evals, MAP screen + MAP objective + MAP acceptance",
}


def decoupled_residual_arm_summary(arm: str) -> dict | None:
    """Per-arm summary.json of results/lung_mesh_ai_v2/synthetic50_decoupled_residual/<arm>/."""
    arm_dir = DECOUPLED_50 / arm
    records = arm_dir / "records.jsonl"
    if not records.exists():
        return None
    ids50 = locked_50()
    rows = latest_per_patient(load(records))
    shards = []
    for shard_dir in sorted(p for p in arm_dir.glob("shard*") if p.is_dir()):
        manifest = _read_json(shard_dir / "run_manifest.json")
        shard_rows = latest_per_patient(load(shard_dir / "records.jsonl"))
        shards.append({
            "shard": shard_dir.name,
            "patient_start": manifest.get("patient_start"),
            "patient_limit": manifest.get("patient_limit"),
            "expected": manifest.get("patient_limit"),
            "done": len(shard_rows),
            "complete": len(shard_rows) == manifest.get("patient_limit"),
            "run_manifest_sha256": sha(shard_dir / "run_manifest.json"),
            "records_sha256": sha(shard_dir / "records.jsonl"),
            "manifest_acceptance_metric": manifest.get("acceptance_metric", "<absent>"),
            "manifest_objective_metric": manifest.get("objective_metric"),
            "manifest_screening_metric": manifest.get("screening_metric"),
            "manifest_prior_width_scale": manifest.get("prior_width_scale"),
            "manifest_max_nfev": manifest.get("max_nfev"),
            "manifest_probe_count": manifest.get("probe_count"),
            "manifest_material_prior_weight": manifest.get("material_prior_weight"),
        })
    block = _decoupled_block(rows, DECOUPLED_LABELS[arm], ids50)
    return {
        "arm": arm,
        "status": "complete" if block["unique_ids_equal_50"] else "incomplete",
        "records_path": _rel_posix(records),
        "records_sha256": sha(records),
        "shards": shards,
        **block,
    }


def decoupled_residual_comparison() -> dict:
    ids50 = locked_50()
    arms = {}
    for arm in DECOUPLED_LABELS:
        records = DECOUPLED_50 / arm / "records.jsonl"
        block = _decoupled_block(latest_per_patient(load(records)), DECOUPLED_LABELS[arm], ids50)
        arms[arm] = {"arm": arm, "status": "complete" if block["unique_ids_equal_50"] else "incomplete",
                     "records_path": _rel_posix(records), "records_sha256": sha(records), **block}
    references = {}
    if FROZEN_PRIMARY_METRICS.exists():
        metrics = _read_json(FROZEN_PRIMARY_METRICS)
        references["frozen_primary_screened_map_strict"] = {
            "path": _rel_posix(FROZEN_PRIMARY_METRICS),
            "sha256": sha(FROZEN_PRIMARY_METRICS),
            "benchmark_method": metrics.get("benchmark_method"),
            **_decoupled_block(
                metrics["records"],
                "frozen primary (evaluate_sim_lung_v2, width 1.0, 96 evals, MAP screen/objective/acceptance, no ratio lock, no eq rollback)",
                ids50,
            ),
        }
    six_ll = SIX_ARM_50 / "learned_learned" / "records.jsonl"
    references["six_arm_learned_learned"] = {
        "path": _rel_posix(six_ll),
        "sha256": sha(six_ll),
        **_decoupled_block(
            latest_per_patient(load(six_ll)),
            "six-arm learned_learned (run_screened_fem_locked, width 2.0, 48 evals, MAP screen/objective/acceptance)",
            ids50,
        ),
    }
    return {
        "arms": arms,
        "references": references,
        "all_arms_complete": all(a["status"] == "complete" for a in arms.values()),
    }


# ----------------------------------------------------------------------------- learned initializer re-inference on mismatched observations
def _quantiles(values: list[float]) -> dict:
    import numpy as np

    arr = np.asarray(sorted(float(v) for v in values), dtype=float)
    return {
        "n": int(arr.size),
        "median": med(arr.tolist()),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


REINFER_ERROR_KEYS = (
    "E_background_relative_error",
    "inclusion_ratio_raw_hybrid_relative_error",
    "inclusion_ratio_raw_2028_relative_error",
    "inclusion_ratio_official_relative_error",
    "center_error_normalized",
    "radius_relative_error",
)
REINFER_DELTA_KEYS = (
    "E_background_relative_change",
    "E_background_log_delta",
    "inclusion_ratio_raw_hybrid_log_delta",
    "inclusion_ratio_raw_2028_log_delta",
    "inclusion_ratio_official_relative_change",
    "inclusion_ratio_official_log_delta",
    "center_shift_normalized",
    "radius_relative_change",
    "heterogeneity_probability_hybrid_delta",
)
REINFER_ABS_DELTA_KEYS = (
    "E_background_log_delta",
    "inclusion_ratio_official_log_delta",
    "inclusion_ratio_raw_hybrid_log_delta",
    "radius_relative_change",
)


def reinference_summary() -> dict | None:
    records = REINFER / "records.jsonl"
    if not records.exists():
        return None
    rows = load(records)
    by_cond: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        by_cond.setdefault(str(row["condition"]), {}).setdefault(str(row["variant"]), []).append(row)
    matched = {str(r["patient_id"]): r for r in by_cond["matched"]["stored"]}

    def _errors(block_rows: list[dict]) -> dict:
        return {key: _quantiles([r[key] for r in block_rows]) for key in REINFER_ERROR_KEYS}

    def _common(block_rows: list[dict]) -> dict:
        return {
            "scenes_E_error_le_0.10": sum(float(r["E_background_relative_error"]) <= 0.10 for r in block_rows),
            "scenes_official_ratio_error_le_0.25": sum(float(r["inclusion_ratio_official_relative_error"]) <= 0.25 for r in block_rows),
            "pca_ridge_ood_detected_count": sum(bool(r.get("pca_ridge_ood_detected")) for r in block_rows),
        }

    conditions_summary: dict = {
        "matched": {"n": len(matched), "errors": _errors(list(matched.values())), **_common(list(matched.values()))},
    }
    for condition in ("fiber", "viscous", "two_inclusions", "diffuse", "sliding_bc"):
        conditions_summary[condition] = {}
        for variant, block_rows in by_cond.get(condition, {}).items():
            block_rows = sorted(block_rows, key=lambda r: str(r["patient_id"]))
            paired = [(r, matched[str(r["patient_id"])]) for r in block_rows if str(r["patient_id"]) in matched]
            conditions_summary[condition][variant] = {
                "n": len(block_rows),
                "peak_displacement_relative_gap": _quantiles([r["peak_displacement_relative_gap"] for r in block_rows]),
                "errors": _errors(block_rows),
                "paired_error_increase_vs_matched": {
                    key: _quantiles([float(r[key]) - float(m[key]) for r, m in paired]) for key in REINFER_ERROR_KEYS
                },
                "deltas_vs_matched_prediction": {key: _quantiles([r[key] for r in block_rows]) for key in REINFER_DELTA_KEYS},
                "abs_deltas_vs_matched_prediction": {key: _quantiles([abs(float(r[key])) for r in block_rows]) for key in REINFER_ABS_DELTA_KEYS},
                **_common(block_rows),
                "scenes_official_ratio_changed": sum(float(r["inclusion_ratio_official"]) != float(matched[str(r["patient_id"])]["inclusion_ratio_official"]) for r in block_rows),
            }
    return {
        "schema_version": 1,
        "scenes": sorted(matched),
        "conditions": ["fiber", "viscous", "two_inclusions", "diffuse", "sliding_bc"],
        "conditions_summary": conditions_summary,
        "records_file": _rel_posix(records),
        "records_sha256": sha(records),
        "record_count": len(rows),
        "script": "experiments/reinfer_mismatch_initializer.py",
        "script_sha256": sha(ROOT / "experiments" / "reinfer_mismatch_initializer.py"),
        "input_hashes": {
            "paired_scene_ids": sha(LOCKED_18),
            "dataset_manifest": sha(CT_MANIFEST),
            "hybrid_benchmark": sha(CT120_BENCHMARK),
            **({"official_ratio_source": sha(CALIBRATE_V2 / "gnn_test.json")} if (CALIBRATE_V2 / "gnn_test.json").exists() else {}),
        },
    }


# ----------------------------------------------------------------------------- record-only audits
def accepted_update_harm_payload() -> dict:
    sys.path.insert(0, str(ROOT))
    from experiments.accepted_update_harm import build_payload  # noqa: E402

    return build_payload()


def _initializer_block(path: Path, baseline: dict[str, dict]) -> dict:
    rows = sorted(latest_per_patient(load(path)), key=lambda r: str(r["patient_id"]))
    own = all("initial_E_background" in r and "E_background_true" in r for r in rows)

    def init_err(r: dict) -> float:
        return _init_error(r) if own else _init_error(baseline[str(r["patient_id"])])

    tol = 1e-9

    def outcome(i: float, f: float) -> str:
        return "tied" if abs(f - i) <= tol else ("worse" if f > i else "better")

    per_scene = [
        {"patient_id": r["patient_id"], "initializer_error": init_err(r), "final_error": float(r["E_background_relative_error"]),
         "accepted": bool(r.get("refinement_accepted")), "outcome": outcome(init_err(r), float(r["E_background_relative_error"]))}
        for r in rows
    ]
    accepted = [s for s in per_scene if s["accepted"]]

    def tally(items: list[dict]) -> dict:
        return {"n": len(items), "worse_than_initializer": sum(s["outcome"] == "worse" for s in items),
                "better_than_initializer": sum(s["outcome"] == "better" for s in items), "tied": sum(s["outcome"] == "tied" for s in items)}

    return {
        "path": _rel_posix(path),
        "records_sha256": sha(path),
        "n": len(rows),
        "unique_ids": len({str(r["patient_id"]) for r in rows}),
        "ids_match_baseline_18": {str(r["patient_id"]) for r in rows} == set(baseline),
        "accepted": len(accepted),
        "screening_rejected": sum(bool(r.get("screening_rejected")) for r in rows) if all("screening_rejected" in r for r in rows) else None,
        "initializer_fields_in_records": own,
        "initializer_reference_source": "own records (initial_E_background vs E_background_true)" if own
        else "baseline w2_s0p05_p2 records (same hybrid initializer and same truth E per scene)",
        "median_initializer_only_error": med(s["initializer_error"] for s in per_scene),
        "median_final_error": med(s["final_error"] for s in per_scene),
        "median_paired_delta_final_minus_initializer": med(s["final_error"] - s["initializer_error"] for s in per_scene),
        "all_scenes": tally(per_scene),
        "accepted_set": {**tally(accepted), "median_initializer_only_error": med(s["initializer_error"] for s in accepted),
                         "median_final_error": med(s["final_error"] for s in accepted)},
        "per_scene": per_scene,
    }


def initializer_reference_summary() -> dict:
    baseline = _rows_by_id(BASELINE_18 / "records.jsonl")
    groups = {
        "assumption_sensitivity": V6 / "assumption_sensitivity",
        "constitutive_mismatch": V6 / "constitutive_mismatch",
        "structural_mismatch": V6 / "structural_mismatch",
        "decoupled_screen": V6 / "decoupled_screen",
        "population_prior": V6 / "population_prior",
    }
    return {
        "tie_tolerance_abs": 1e-9,
        "baseline": _initializer_block(BASELINE_18 / "records.jsonl", baseline),
        "groups": {
            name: {arm.name: _initializer_block(arm / "records.jsonl", baseline) for arm in sorted(base.iterdir()) if (arm / "records.jsonl").exists()}
            for name, base in groups.items()
        },
    }


def calibration_fourbin_summary() -> dict | None:
    if not (CALIBRATE_V2 / "gnn_train.json").exists():
        return None
    sys.path.insert(0, str(ROOT))
    from experiments.run_ct_reviewer_drivers import fit_bins, score_policy  # noqa: E402

    # Locked rule of lung_inverse_rendering.pinn_inverse.official_bins3_regularized
    # (duplicated here so that the checker stays torch-free).
    OFFICIAL_BINS3_EDGES = (0.0, 2.3522442307010984, 3.1483814709163886, 10.0)
    OFFICIAL_BINS3_MULTIPLIERS = (0.625, 1.175, 1.05)

    splits = {name: _read_json(CALIBRATE_V2 / f"gnn_{name}.json") for name in ("train", "val", "test")}

    def rows_of(payload) -> list[dict]:
        return payload["records"] if isinstance(payload, dict) and "records" in payload else payload

    def geometries(rows: list[dict]) -> list[str]:
        return sorted({str(r.get("geometry_id") or str(r["patient_id"]).split("_scenario_", 1)[0]) for r in rows})

    train, val, test = (rows_of(splits[k]) for k in ("train", "val", "test"))
    four_edges, four_mult = fit_bins(train, 4, 1.0)
    three_edges, three_mult = fit_bins(train, 3, 1.0)
    official_edges, official_mult = list(OFFICIAL_BINS3_EDGES), list(OFFICIAL_BINS3_MULTIPLIERS)

    def inner_loo(bins: int) -> dict:
        folds = []
        for geom in geometries(train):
            held = [r for r in train if str(r.get("geometry_id") or str(r["patient_id"]).split("_scenario_", 1)[0]) == geom]
            rest = [r for r in train if r not in held]
            edges, mult = fit_bins(rest, bins, 1.0)
            folds.append({"held_train_geometry": geom, "median_error": score_policy(held, edges, mult)["median_error"]})
        # run_ct_reviewer_drivers.calibration_stability aggregates the fold medians with the
        # upper-middle element (sorted[n // 2]); reproduced verbatim here.
        fold_medians = sorted(f["median_error"] for f in folds)
        return {"bins": bins, "width_scale": 1.0, "inner_loo_median": fold_medians[len(fold_medians) // 2], "fold_count": len(folds), "folds": folds}

    def scored(rows: list[dict], edges, mult, lists: bool = False) -> dict:
        out = score_policy(rows, edges, mult)
        if lists:
            out["passing_geometries_list"] = [g for g, v in out["geometry_medians"].items() if v <= 0.25]
            out["failing_geometries_list"] = [g for g, v in out["geometry_medians"].items() if v > 0.25]
        return out

    val_four = scored(val, four_edges, four_mult)
    val_three = scored(val, three_edges, three_mult)
    val_official = scored(val, official_edges, official_mult)
    test_four = scored(test, four_edges, four_mult, lists=True)
    test_three = scored(test, three_edges, three_mult)
    test_official = scored(test, official_edges, official_mult, lists=True)
    test_raw = score_policy(test, [0.0, 10.0], [1.0])
    inner_three, inner_four = inner_loo(3), inner_loo(4)
    return {
        "post_lock_labelled_sensitivity_not_selection": True,
        "inputs": {
            f"gnn_{name}": {"path": _rel_posix(CALIBRATE_V2 / f"gnn_{name}.json"), "sha256": sha(CALIBRATE_V2 / f"gnn_{name}.json"),
                            "n": len(rows_of(splits[name])), "geometries": geometries(rows_of(splits[name]))}
            for name in ("train", "val", "test")
        },
        "rules": {
            "fourbin_w1_train_fit": {"bins": 4, "width_scale": 1.0, "edges": four_edges, "multipliers": four_mult},
            "threebin_w1_train_fit_refit": {"bins": 3, "width_scale": 1.0, "edges": three_edges, "multipliers": three_mult},
            "official_bins3_regularized_locked": {"edges": official_edges, "multipliers": official_mult,
                                                   "source": "lung_inverse_rendering.pinn_inverse.official_bins3_regularized",
                                                   "identical_to_threebin_w1_refit": official_edges == three_edges and official_mult == three_mult},
        },
        "inner_loo_train": {"threebin_w1": inner_three, "fourbin_w1": inner_four},
        "validation_scores_that_drove_lock": {
            "fourbin_w1": val_four, "threebin_w1_refit": val_three, "official_bins3": val_official,
            "fourbin_w1_failing_geometries": {g: {"median_error": v, "excess_over_0.25": v - 0.25} for g, v in val_four["geometry_medians"].items() if v > 0.25},
        },
        "post_lock_test_sensitivity": {"fourbin_w1": test_four, "threebin_w1_refit": test_three, "official_bins3": test_official, "uncalibrated_raw_gnn": test_raw},
        "summary": {
            "inner_loo_median_threebin_w1": inner_three["inner_loo_median"],
            "inner_loo_median_fourbin_w1": inner_four["inner_loo_median"],
            "val_fourbin_median": val_four["median_error"],
            "val_fourbin_pass": f"{val_four['geometries_passing_0.25']}/{val_four['geometry_count']}",
            "val_official3_median": val_official["median_error"],
            "val_official3_pass": f"{val_official['geometries_passing_0.25']}/{val_official['geometry_count']}",
            "test_fourbin_median": test_four["median_error"],
            "test_fourbin_pass": f"{test_four['geometries_passing_0.25']}/{test_four['geometry_count']}",
            "test_official3_median": test_official["median_error"],
            "test_official3_pass": f"{test_official['geometries_passing_0.25']}/{test_official['geometry_count']}",
        },
    }


def ct120_audit_mechanics_summary() -> dict:
    records = CT120_AUDIT / "records.jsonl"
    summary_path = CT120_AUDIT / "summary.json"
    rows = latest_per_patient(load(records))
    archived_summary = _read_json(summary_path)
    continued = [r for r in rows if not bool(r.get("screening_rejected"))]
    rejected = [r for r in rows if bool(r.get("screening_rejected"))]
    benchmark = _read_json(CT120_BENCHMARK)
    # benchmark records carry geometry_id / scenario_id (some also a "geom/scenario" patient_id)
    benchmark_ids = {
        f"{row['geometry_id']}_{row['scenario_id']}" if row.get("geometry_id") and row.get("scenario_id") else str(row.get("patient_id"))
        for row in benchmark.get("records", [])
    }
    locked = set(locked_ids())
    audit_rows = _rows_by_id(records)

    def identical(arm_dir: Path, key: str) -> int:
        arm = _rows_by_id(arm_dir / "records.jsonl")
        return sum(arm[p][key] == audit_rows[p][key] for p in sorted(locked & set(arm) & set(audit_rows)))

    def reductions(items: list[dict]) -> list[float]:
        values = sorted(float(r["relative_cost_reduction"]) for r in items)
        return [values[0], med(values), values[-1]] if values else []

    out: dict = {
        "run_dir": _rel_posix(CT120_AUDIT),
        "run_manifest_exists": (CT120_AUDIT / "run_manifest.json").exists(),
        "run_manifest_sha256": sha(CT120_AUDIT / "run_manifest.json") if (CT120_AUDIT / "run_manifest.json").exists() else None,
        "records_sha256": sha(records),
        "summary_sha256": sha(summary_path),
        "settings_from_summary_protocol": archived_summary.get("protocol"),
        "settings_evidence_in_records": {
            "max_nfev_48_consistent": max(int(r["function_evaluations"]) for r in rows) <= 48,
            "max_function_evaluations_observed": max(int(r["function_evaluations"]) for r in rows),
            "screening_rejected_function_evaluations": sorted({int(r["function_evaluations"]) for r in rejected}),
            "probe_count_2_consistent": all(int(r["function_evaluations"]) == 5 for r in rejected),
            "lock_ratio": _counts(str(bool(r.get("lock_ratio"))) for r in rows),
            "ratio_source": _counts(r.get("ratio_source") for r in rows),
            "method": _counts(r.get("method") for r in rows),
            "record_keys_present": sorted(rows[0].keys()),
        },
        "n_records": len(load(records)),
        "unique_patient_ids": len(audit_rows),
        "ids_match_ct_benchmark_120": set(audit_rows) == benchmark_ids if benchmark_ids else None,
        "termination_reason_counts": _counts(r.get("termination_reason") for r in rows),
        "E_force_correlation_status_counts": _counts(r.get("E_force_correlation_status") for r in rows),
        "function_evaluations_counts": {str(k): v for k, v in sorted(Counter(int(r["function_evaluations"]) for r in rows).items())},
        "acceptance_mechanics": {
            "screening_rejected": len(rejected),
            "continued_search_not_screening_rejected": len(continued),
            "refinement_accepted": sum(bool(r.get("refinement_accepted")) for r in rows),
            "continued_but_final_acceptance_rolled_back": sum(not bool(r.get("refinement_accepted")) for r in continued),
            "continued_but_rolled_back_ids": sorted(str(r["patient_id"]) for r in continued if not bool(r.get("refinement_accepted"))),
            "screening_rejected_but_accepted": sum(bool(r.get("refinement_accepted")) for r in rejected),
            "eq_correlation_rejected": sum(bool(r.get("eq_correlation_rejected")) for r in rows),
            "eq_correlation_rejected_ids": sorted(str(r["patient_id"]) for r in rows if bool(r.get("eq_correlation_rejected"))),
            "every_continued_search_was_accepted": all(bool(r.get("refinement_accepted")) for r in continued),
            "relative_cost_reduction_continued_min_median_max": reductions(continued),
            "relative_cost_reduction_screening_rejected_min_median_max": reductions(rejected),
        },
        "ct_dataset_test_count": sum(1 for r in _read_json(CT_MANIFEST)["patients"] if r["split"] == "test"),
        "prior_width_scale_evidence": {
            "width_1_arm": {
                "records": _rel_posix(W1_ARM_18 / "records.jsonl"),
                "records_sha256": sha(W1_ARM_18 / "records.jsonl"),
                "run_manifest_prior_width_scale": _read_json(W1_ARM_18 / "run_manifest.json").get("prior_width_scale"),
                "scenes_compared": len(locked & set(_rows_by_id(W1_ARM_18 / "records.jsonl")) & set(audit_rows)),
                "E_background_estimated_identical": identical(W1_ARM_18, "E_background_estimated"),
                "function_evaluations_identical": identical(W1_ARM_18, "function_evaluations"),
                "refinement_accepted_identical": identical(W1_ARM_18, "refinement_accepted"),
            },
            "width_2_arm": {
                "records": _rel_posix(BASELINE_18 / "records.jsonl"),
                "records_sha256": sha(BASELINE_18 / "records.jsonl"),
                "run_manifest_prior_width_scale": _read_json(BASELINE_18 / "run_manifest.json").get("prior_width_scale"),
                "scenes_compared": len(locked & set(_rows_by_id(BASELINE_18 / "records.jsonl")) & set(audit_rows)),
                "E_background_estimated_identical": identical(BASELINE_18, "E_background_estimated"),
            },
        },
    }
    out["prior_width_scale_evidence"]["value"] = 1.0 if out["prior_width_scale_evidence"]["width_1_arm"]["E_background_estimated_identical"] == 18 else None
    return out


def aspect_arm_sanity_summary() -> dict:
    """Record-derived part of aspect_arm_sanity.json (the occupancy recompute needs the dataset tensors)."""
    base = _rows_by_id(BASELINE_18 / "records.jsonl")
    fields = ("E_background_estimated", "inclusion_ratio_estimated", "force_scale_estimated", "function_evaluations",
              "refinement_accepted", "screening_rejected", "cost", "initial_cost", "relative_cost_reduction")
    arms = {}
    for name, aspect in (("aspect_1p5", 1.5), ("aspect_2p0", 2.0)):
        records = V6 / "assumption_sensitivity" / name / "records.jsonl"
        rows = _rows_by_id(records)
        common = sorted(set(rows) & set(base))
        by_field = {f: sum(rows[p][f] == base[p][f] for p in common) for f in fields}
        arms[name] = {
            "records_path": _rel_posix(records),
            "records_sha256": sha(records),
            "region_aspect_xy": aspect,
            "n": len(rows),
            "n_paired_with_baseline": len(common),
            "E_background_estimated_bitwise_identical": by_field["E_background_estimated"],
            "bitwise_identical_by_field": by_field,
            "all_compared_fields_identical": sum(all(rows[p][f] == base[p][f] for f in fields) for p in common),
            "differing_scenes": [
                {"patient_id": p, "E_arm": rows[p]["E_background_estimated"], "E_baseline": base[p]["E_background_estimated"],
                 "nfev_arm": rows[p]["function_evaluations"], "nfev_baseline": base[p]["function_evaluations"]}
                for p in common if not all(rows[p][f] == base[p][f] for f in fields)
            ],
            # aspect_arm_sanity.py reports these four descriptive values with the upper-middle
            # element (sorted[n // 2]), not the arithmetic median; reproduced verbatim.
            "median_E_error_percent_arm": round(100.0 * upper_median(r["E_background_relative_error"] for r in rows.values()), 4),
            "median_E_error_percent_baseline": round(100.0 * upper_median(r["E_background_relative_error"] for r in base.values()), 4),
            "wall_time_median_arm_s": upper_median(r["wall_time_seconds"] for r in rows.values()),
            "wall_time_median_baseline_s": upper_median(r["wall_time_seconds"] for r in base.values()),
        }
    return {
        "baseline_records": _rel_posix(BASELINE_18 / "records.jsonl"),
        "baseline_sha256": sha(BASELINE_18 / "records.jsonl"),
        "baseline_n": len(base),
        "arms": arms,
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

# Second-revision groups: verified field-by-field (every key the builder
# recomputes must equal the archived value; archived-only keys are listed as
# not recomputable).  ``--write`` is refused for these groups because the
# archived files also carry fields that need the dataset tensors or
# checkpoints (occupancy checks, prose conclusions, wall-clock stamps).
VERIFY_GROUPS = {
    "ekf_sequential_ct120_all": (V6 / "ekf_sequential_ct120_all" / "summary.json", lambda: ekf_cohort_summary(V6 / "ekf_sequential_ct120_all", "ct120")),
    "ekf_sequential_synthetic50": (LM / "ekf_sequential_synthetic50" / "summary.json", lambda: ekf_cohort_summary(LM / "ekf_sequential_synthetic50", "synthetic50")),
    "ekf_sequential_synthetic50_superseded_v0": (
        LM / "ekf_sequential_synthetic50" / "superseded_v0_unclipped_initial_ratio" / "summary.json",
        lambda: ekf_cohort_summary(LM / "ekf_sequential_synthetic50" / "superseded_v0_unclipped_initial_ratio", "synthetic50"),
    ),
    "model_mismatch_n48": (V6 / "model_mismatch_n48" / "summary.json", model_mismatch_n48_summary),
    "assumption_sensitivity_fixed": (V6 / "assumption_sensitivity_fixed" / "summary.json", assumption_fixed_summary),
    "synthetic50_decoupled_residual": (DECOUPLED_50 / "comparison.json", decoupled_residual_comparison),
    "synthetic50_decoupled_residual/primary_like_w1_n96": (DECOUPLED_50 / "primary_like_w1_n96" / "summary.json", lambda: decoupled_residual_arm_summary("primary_like_w1_n96")),
    "synthetic50_decoupled_residual/sixarm_like_w2_n48": (DECOUPLED_50 / "sixarm_like_w2_n48" / "summary.json", lambda: decoupled_residual_arm_summary("sixarm_like_w2_n48")),
    "synthetic50_decoupled_residual/map_w1_n96": (DECOUPLED_50 / "map_w1_n96" / "summary.json", lambda: decoupled_residual_arm_summary("map_w1_n96")),
    "mismatch_initializer_reinference": (REINFER / "summary.json", reinference_summary),
    "accepted_update_harm": (V6 / "accepted_update_harm.json", accepted_update_harm_payload),
    "initializer_reference_18": (V6 / "initializer_reference_18.json", initializer_reference_summary),
    "calibration_fourbin_test_sensitivity": (V6 / "calibration_fourbin_test_sensitivity.json", calibration_fourbin_summary),
    "ct120_audit_mechanics": (V6 / "ct120_audit_mechanics.json", ct120_audit_mechanics_summary),
    "aspect_arm_sanity": (V6 / "assumption_sensitivity" / "aspect_arm_sanity.json", aspect_arm_sanity_summary),
}
ALL_GROUPS = {**GROUPS, **VERIFY_GROUPS}

# Keys of the verify groups that are intentionally not recomputed (they need
# the generated dataset tensors / checkpoints, are prose, or are wall-clock
# stamps). Listed so the report is explicit about what is NOT verified.
NOT_RECOMPUTED = {
    "ekf_sequential_ct120_all": ("duplicate_rows_dropped_in_merge (merge identity asserted instead)",),
    "ekf_sequential_synthetic50": ("duplicate_rows_dropped_in_merge (merge identity asserted instead)",),
    "ekf_sequential_synthetic50_superseded_v0": ("duplicate_rows_dropped_in_merge (merge identity asserted instead)",),
    "model_mismatch_n48": ("notes (prose)",),
    "assumption_sensitivity_fixed": ("vs_baseline/*/occupancy_* (needs node coordinates)", "pre_run_occupancy_check (needs node coordinates)", "conclusion, notes (prose)"),
    "synthetic50_decoupled_residual": ("protocol_notes (prose)",),
    "synthetic50_decoupled_residual/primary_like_w1_n96": ("shards[*].command (launch command text)",),
    "synthetic50_decoupled_residual/sixarm_like_w2_n48": ("shards[*].command (launch command text)",),
    "synthetic50_decoupled_residual/map_w1_n96": ("shards[*].command (launch command text)",),
    "mismatch_initializer_reinference": ("purpose, variants, learned_pipeline, truth_convention (prose / checkpoint hashes)", "reproduction_check* (needs checkpoints)"),
    "initializer_reference_18": ("description (prose)",),
    "calibration_fourbin_test_sensitivity": ("statement (prose)", "inputs/fit_bins_score_policy_source (path string)"),
    "ct120_audit_mechanics": ("records_last_write (file mtime)", "settings_inferred_from_documented_reproduction_command (prose)", "prior_width_scale_evidence/status, method, conclusion (prose)", "settings_evidence_in_records/*_note, *_from_summary_protocol (prose copies)", "manifest_patient_ids_count, ids_match_manifest (no run manifest)"),
    "aspect_arm_sanity": ("description, fit_patient_side_effect_note (prose)", "occupancy_recompute (needs node coordinates)"),
}


def _normalise(value):
    """Path strings are compared after stripping the machine-local root and the
    directory separator convention, so a Windows development tree and the
    public mirror (where the root is scrubbed to '.') compare equal."""
    if isinstance(value, str) and ("\\" in value or "/" in value):
        text = value.replace(str(ROOT), "").replace(str(ROOT).replace("\\", "\\\\"), "")
        text = text.replace("\\", "/")
        while text.startswith("./") or text.startswith("/"):
            text = text[2:] if text.startswith("./") else text[1:]
        return text
    return value


def _equal(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, float) and isinstance(actual, float) and math.isnan(expected) and math.isnan(actual):
            return True
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-12)
    return _normalise(expected) == _normalise(actual)


def diff_paths(expected, actual, prefix: str = "", subset: bool = False) -> list[str]:
    """Differences between the archived (expected) and regenerated (actual)
    payloads. With ``subset=True`` keys that the regeneration does not produce
    are ignored (verify mode)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                # verify mode: an archived file written by an older script version may
                # lack fields the current regeneration adds (e.g. the superseded EKF run)
                if not subset:
                    out.append(f"{prefix}/{key}: missing in archived")
            elif key not in actual:
                if not subset:
                    out.append(f"{prefix}/{key}: missing in regenerated")
            else:
                out.extend(diff_paths(expected[key], actual[key], f"{prefix}/{key}", subset))
        return out
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        out = []
        for index, (e, a) in enumerate(zip(expected, actual)):
            out.extend(diff_paths(e, a, f"{prefix}[{index}]", subset))
        return out
    return [] if _equal(expected, actual) else [f"{prefix}: archived={expected!r} regenerated={actual!r}"]


def count_leaves(value) -> int:
    if isinstance(value, dict):
        return sum(count_leaves(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_leaves(v) for v in value) if value else 1
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="*", default=list(ALL_GROUPS), choices=list(ALL_GROUPS), help="subset of groups")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify archived summaries (default)")
    mode.add_argument("--write", action="store_true", help="overwrite archived summaries from records (regeneration groups only)")
    args = parser.parse_args()
    failures = 0
    for name in args.groups:
        verify = name in VERIFY_GROUPS
        target, builder = ALL_GROUPS[name]
        if not target.exists():
            print(f"[skip]  {name}: archived summary not present ({target.relative_to(ROOT)})")
            continue
        regenerated = builder()
        if regenerated is None:
            print(f"[skip]  {name}: no records")
            continue
        if args.write:
            if verify:
                print(f"[skip]  {name}: verify-only group, --write refused (archived file carries fields that need the dataset/checkpoints)")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(regenerated, indent=2), encoding="utf-8")
            print(f"[write] {name}: {target.relative_to(ROOT)}")
            continue
        archived = json.loads(target.read_text(encoding="utf-8-sig"))
        differences = diff_paths(archived, regenerated, subset=verify)
        if differences:
            failures += 1
            print(f"[DIFF]  {name}: {len(differences)} difference(s)")
            for line in differences[:20]:
                print("        ", line)
        elif verify:
            skipped = "; ".join(NOT_RECOMPUTED.get(name, ()))
            print(f"[ok]    {name}: {count_leaves(regenerated)} recomputed values equal the archived file ({target.relative_to(ROOT)})"
                  + (f"; not recomputed: {skipped}" if skipped else ""))
        else:
            print(f"[ok]    {name}: archived summary equals regeneration from records ({target.relative_to(ROOT)})")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
