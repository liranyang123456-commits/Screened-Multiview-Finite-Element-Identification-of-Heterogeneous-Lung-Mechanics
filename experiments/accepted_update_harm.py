"""Accepted-update harm statistics from EXISTING screened-FEM records (no FEM runs).

For every arm: among scenes whose refinement was accepted, count how many
accepted updates increased / decreased / left unchanged the background error
relative to the initializer (initializer error = |initial_E_background -
E_background_true| / E_background_true, which the records store), plus the
accepted-set medians of initializer and final error.  The same is done for the
inclusion ratio restricted to accepted scenes whose ratio was not locked.

Writes results/ct_reviewer_drivers_v6/accepted_update_harm.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ct_reviewer_drivers_v6" / "accepted_update_harm.json"
SIX_ARM = ROOT / "results" / "lung_mesh_ai_v2" / "synthetic50_full_paired_controls_20261008"
SWEEP = ROOT / "results" / "lung_mesh_ai_v2" / "synthetic50_threshold_sweep"
TIE_TOL = 1e-12

ARMS: dict[str, Path] = {
    "six_arm/learned_learned": SIX_ARM / "learned_learned" / "records.jsonl",
    "six_arm/learned_fixed": SIX_ARM / "learned_fixed" / "records.jsonl",
    "six_arm/fixed_learned": SIX_ARM / "fixed_learned" / "records.jsonl",
    "six_arm/fixed_fixed": SIX_ARM / "fixed_fixed" / "records.jsonl",
    "six_arm/no_material_prior": SIX_ARM / "no_material_prior" / "records.jsonl",
    "six_arm/residual_only": SIX_ARM / "residual_only" / "records.jsonl",
    "threshold_sweep/s0p00": SWEEP / "s0p00" / "records.jsonl",
    "threshold_sweep/s0p01": SWEEP / "s0p01" / "records.jsonl",
    "threshold_sweep/s0p025": SWEEP / "s0p025" / "records.jsonl",
    "threshold_sweep/s0p05(=six_arm learned_learned)": SIX_ARM / "learned_learned" / "records.jsonl",
    "threshold_sweep/s0p10": SWEEP / "s0p10" / "records.jsonl",
    "threshold_sweep/s0p20": SWEEP / "s0p20" / "records.jsonl",
    "ct120/official_prior_screened_fem120_v1": ROOT / "results" / "ion_ct_official_prior_screened_fem120_v1" / "records.jsonl",
    "ct18/w2_s0p05_p2": ROOT / "results" / "ct_reviewer_drivers_v6" / "ct_reviewer_drivers_v6" / "fem_paired" / "w2_s0p05_p2" / "records.jsonl",
}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return float(ordered[mid]) if n % 2 else float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 4)


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _direction_counts(pairs: list[tuple[float, float]]) -> dict:
    increased = decreased = tied = 0
    for initial, final in pairs:
        delta = final - initial
        if abs(delta) <= TIE_TOL:
            tied += 1
        elif delta > 0:
            increased += 1
        else:
            decreased += 1
    deltas_pp = [100.0 * (final - initial) for initial, final in pairs]
    return {
        "n": len(pairs),
        "increased_error": increased,
        "decreased_error": decreased,
        "tied": tied,
        "median_initializer_error_percent": _pct(_median([p[0] for p in pairs])),
        "median_final_error_percent": _pct(_median([p[1] for p in pairs])),
        "median_paired_delta_pp": round(_median(deltas_pp), 4) if deltas_pp else None,
        "mean_paired_delta_pp": round(sum(deltas_pp) / len(deltas_pp), 4) if deltas_pp else None,
    }


def analyse_arm(rows: list[dict]) -> dict:
    accepted = [r for r in rows if bool(r.get("refinement_accepted"))]
    rejected = [r for r in rows if not bool(r.get("refinement_accepted"))]
    missing = [r["patient_id"] for r in rows if "initial_E_background" not in r or "E_background_true" not in r]

    def e_init(r: dict) -> float:
        return abs(float(r["initial_E_background"]) - float(r["E_background_true"])) / float(r["E_background_true"])

    def r_init(r: dict) -> float:
        return abs(float(r["initial_ratio"]) - float(r["inclusion_ratio_true"])) / float(r["inclusion_ratio_true"])

    e_pairs_acc = [(e_init(r), float(r["E_background_relative_error"])) for r in accepted]
    e_pairs_rej = [(e_init(r), float(r["E_background_relative_error"])) for r in rejected]
    e_pairs_all = [(e_init(r), float(r["E_background_relative_error"])) for r in rows]
    unlocked_acc = [r for r in accepted if not bool(r.get("lock_ratio"))]
    r_pairs_acc = [(r_init(r), float(r["inclusion_ratio_relative_error"])) for r in unlocked_acc]
    r_pairs_all_unlocked = [(r_init(r), float(r["inclusion_ratio_relative_error"])) for r in rows if not bool(r.get("lock_ratio"))]
    rejected_identity_violations = [
        r["patient_id"] for r in rejected
        if abs(float(r["E_background_estimated"]) - float(r["initial_E_background"])) > 1e-9 * max(1.0, float(r["initial_E_background"]))
    ]
    return {
        "n": len(rows),
        "unique_ids": len({r["patient_id"] for r in rows}),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_ratio_locked": sum(bool(r.get("lock_ratio")) for r in rows),
        "n_accepted_ratio_unlocked": len(unlocked_acc),
        "records_missing_initializer_fields": missing,
        "rejected_scenes_whose_E_differs_from_initializer": rejected_identity_violations,
        "E_background": {
            "accepted": _direction_counts(e_pairs_acc),
            "rejected_(final==initializer_by_construction)": _direction_counts(e_pairs_rej),
            "all_scenes": _direction_counts(e_pairs_all),
        },
        "inclusion_ratio_unlocked_only": {
            "accepted": _direction_counts(r_pairs_acc),
            "all_unlocked_scenes": _direction_counts(r_pairs_all_unlocked),
        },
        "cohort_median_E_error_percent": _pct(_median([float(r["E_background_relative_error"]) for r in rows])),
        "cohort_median_initializer_E_error_percent": _pct(_median([e_init(r) for r in rows])),
    }


def build_payload() -> dict:
    """Recompute the complete accepted_update_harm.json payload from the archived records.

    Shared by ``main()`` (which writes the file) and by
    ``experiments/summarize_reviewer_arms.py --check`` (which compares it with the
    archived file without writing).
    """
    out: dict = {
        "description": __doc__.strip(),
        "tie_tolerance_relative": TIE_TOL,
        "definitions": {
            "initializer_error": "|initial_E_background - E_background_true| / E_background_true (fields stored in records)",
            "final_error": "E_background_relative_error (record field)",
            "increased_error": "final_error > initializer_error (accepted update harmed the estimate)",
            "ratio_initializer_error": "|initial_ratio - inclusion_ratio_true| / inclusion_ratio_true, only for scenes with lock_ratio == false",
        },
        "arms": {},
        "claim_checks": {},
    }
    for name, path in ARMS.items():
        if not path.exists():
            out["arms"][name] = {"missing": str(path)}
            continue
        rows = _read(path)
        payload = analyse_arm(rows)
        payload["records_path"] = str(path.relative_to(ROOT))
        payload["records_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        out["arms"][name] = payload

    ll = out["arms"]["six_arm/learned_learned"]["E_background"]["accepted"]
    s20 = out["arms"]["threshold_sweep/s0p20"]["E_background"]["accepted"]
    out["claim_checks"] = {
        "claim_six_arm_learned_learned_23_of_26_worsened": {
            "claimed": {"worsened": 23, "accepted": 26, "accepted_median_initializer_percent": 4.01, "accepted_median_final_percent": 15.99},
            "observed": {
                "worsened": ll["increased_error"],
                "accepted": ll["n"],
                "accepted_median_initializer_percent": ll["median_initializer_error_percent"],
                "accepted_median_final_percent": ll["median_final_error_percent"],
            },
            "matches": (
                ll["increased_error"] == 23 and ll["n"] == 26
                and ll["median_initializer_error_percent"] is not None
                and abs(ll["median_initializer_error_percent"] - 4.01) < 0.01
                and abs(ll["median_final_error_percent"] - 15.99) < 0.01
            ),
        },
        "claim_threshold_0p20_only_11_of_26_worsened": {
            "claimed": {"worsened": 11, "accepted": 26},
            "observed": {"worsened": s20["increased_error"], "accepted": s20["n"]},
            "matches": s20["increased_error"] == 11 and s20["n"] == 26,
        },
    }
    return out


def main() -> None:
    out = build_payload()
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    compact = {
        name: {
            "n": a.get("n"), "accepted": a.get("n_accepted"),
            "E_acc_inc/dec/tie": (
                a["E_background"]["accepted"]["increased_error"],
                a["E_background"]["accepted"]["decreased_error"],
                a["E_background"]["accepted"]["tied"],
            ) if "E_background" in a else None,
            "E_acc_med_init->final_%": (
                a["E_background"]["accepted"]["median_initializer_error_percent"],
                a["E_background"]["accepted"]["median_final_error_percent"],
            ) if "E_background" in a else None,
            "ratio_unlocked_acc_inc/dec/tie": (
                a["inclusion_ratio_unlocked_only"]["accepted"]["increased_error"],
                a["inclusion_ratio_unlocked_only"]["accepted"]["decreased_error"],
                a["inclusion_ratio_unlocked_only"]["accepted"]["tied"],
            ) if "inclusion_ratio_unlocked_only" in a else None,
            "ratio_unlocked_acc_med_init->final_%": (
                a["inclusion_ratio_unlocked_only"]["accepted"]["median_initializer_error_percent"],
                a["inclusion_ratio_unlocked_only"]["accepted"]["median_final_error_percent"],
            ) if "inclusion_ratio_unlocked_only" in a else None,
        }
        for name, a in out["arms"].items()
    }
    print(json.dumps(compact, indent=1))
    print(json.dumps(out["claim_checks"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
