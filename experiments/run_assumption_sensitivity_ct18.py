#!/usr/bin/env python3
"""Paired assumption-sensitivity arms on the locked 18 CT test scenes.

Every arm calls ``experiments/run_screened_fem_locked.py`` with the baseline
protocol (``--cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.05
--probe-count 2 --max-nfev 48`` on the 18 explicit locked scene IDs) plus one
fitter-side or observation/geometry-proxy override. Observations stay on the
generator (nu = 0.45, posterior 20 % fixation, spherical inclusion); the CT
ratio remains locked, so ratio error is not a sensitivity endpoint.

``ARMS`` is the complete set archived under
``results/ct_reviewer_drivers_v6/assumption_sensitivity/`` and reported in the
revision supplement. The three ``BASELINE_EQUIVALENT_ARMS`` reproduce the
generator setting exactly (identical to the baseline arm) and are therefore not
run by default.

``--summarize-only`` regenerates ``summary.json`` from the existing
``records.jsonl`` files in the archived schema without running any FEM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEM = ROOT / "experiments" / "run_screened_fem_locked.py"
SCENE_IDS = ROOT / "results" / "ct_reviewer_drivers_v6" / "paired_scene_ids.json"
OUT = ROOT / "results" / "ct_reviewer_drivers_v6" / "assumption_sensitivity"
BASELINE_RECORDS = (
    ROOT / "results" / "ct_reviewer_drivers_v6" / "ct_reviewer_drivers_v6"
    / "fem_paired" / "w2_s0p05_p2" / "records.jsonl"
)
BASE_FLAGS = [
    "--cohort", "ct120",
    "--prior-width-scale", "2.0",
    "--screen-threshold", "0.05",
    "--probe-count", "2",
]

ARMS = [
    # fitter-side constitutive / boundary / region assumptions
    ("nu_0p40", {"--model-nu": "0.40"}),
    ("nu_0p49", {"--model-nu": "0.49"}),
    ("fix_q0p70", {"--posterior-fixation-quantile": "0.70"}),
    ("fix_q0p90", {"--posterior-fixation-quantile": "0.90"}),
    ("aspect_1p5", {"--region-aspect-xy": "1.5"}),
    ("aspect_2p0", {"--region-aspect-xy": "2.0"}),
    # one joint force/yaw perturbation
    ("joint_force_p10_yaw_p5", {"--model-force-bias": "0.10", "--pose-rotation-error-deg": "5.0"}),
    # observation / geometry / contact proxies
    ("corr_dropout_0p20", {"--tracking-error-model": "dropout", "--tracking-dropout-rate": "0.20"}),
    ("mesh_jitter_0p02", {"--mesh-jitter-fraction": "0.02"}),
    ("contact_shift_0p15", {"--contact-shift-fraction": "0.15"}),
]
# Identical to the generator setting; equal to the baseline arm by construction.
BASELINE_EQUIVALENT_ARMS = [
    ("nu_0p45", {"--model-nu": "0.45"}),
    ("fix_q0p80", {"--posterior-fixation-quantile": "0.80"}),
    ("aspect_1p0", {"--region-aspect-xy": "1.0"}),
]
MEDIAN_CONVENTION = "Arithmetic mean of the two middle values for even sample sizes."
PROTOCOL = (
    "Paired 18 locked CT test scenes. FEM assumption overrides only; "
    "observations stay on the generator (nu=0.45, posterior 20% fixation, sphere). "
    "CT ratio remains locked, so ratio error is not the sensitivity endpoint."
)


def _load_records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["patient_id"] for row in rows if "patient_id" in row]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate patient ids in {path}")
    return rows


def _median(rows: list[dict], key: str) -> float:
    """Arithmetic median: mean of the two central values for even n."""
    return statistics.median(float(row[key]) for row in rows)


def summarize(records_path: Path) -> dict:
    """Fraction-valued per-arm statistics (kept for tests / quick inspection)."""
    rows = _load_records(records_path)
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "accepted": sum(1 for row in rows if row.get("refinement_accepted")),
        "median_background_error": _median(rows, "E_background_relative_error"),
        "median_ratio_error": _median(rows, "inclusion_ratio_relative_error"),
        "median_force_error": _median(rows, "force_scale_relative_error"),
    }


def summarize_arm(records_path: Path) -> dict:
    """Archived summary.json schema: percentages (2 dp), records hash, source path."""
    rows = _load_records(records_path)
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "accepted": sum(1 for row in rows if row.get("refinement_accepted")),
        "median_background_error_percent": round(100.0 * _median(rows, "E_background_relative_error"), 2),
        "median_ratio_error_percent": round(100.0 * _median(rows, "inclusion_ratio_relative_error"), 2),
        "median_force_error_percent": round(100.0 * _median(rows, "force_scale_relative_error"), 2),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "source_records": str(records_path.relative_to(ROOT)) if records_path.is_relative_to(ROOT) else str(records_path),
    }


def baseline_reference() -> dict:
    rows = [json.loads(line) for line in BASELINE_RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {str(row["patient_id"]): row for row in rows}.values()
    rows = list(latest)
    return {
        "arm": "w2_s0p05_p2 learned/learned",
        "median_background_error_percent": round(100.0 * _median(rows, "E_background_relative_error"), 2),
        "median_ratio_error_percent": round(100.0 * _median(rows, "inclusion_ratio_relative_error"), 2),
        "accepted": sum(1 for row in rows if row.get("refinement_accepted")),
        "n": len(rows),
    }


def write_summary(out_dir: Path, expected_ids: set[str]) -> Path:
    summary = {
        "protocol": PROTOCOL,
        "baseline_reference": baseline_reference(),
        "arms": {},
        "median_convention": MEDIAN_CONVENTION,
    }
    for arm_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
        records = arm_dir / "records.jsonl"
        if not records.exists():
            continue
        arm = summarize_arm(records)
        observed = {row["patient_id"] for row in _load_records(records)}
        if observed != expected_ids:
            raise SystemExit(
                f"{arm_dir.name}: {len(observed)} scene IDs do not equal the locked 18-scene set"
            )
        summary["arms"][arm_dir.name] = arm
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_json


def build_command(result_dir: Path, flags: dict[str, str], patient_ids: list[str], max_nfev: int) -> list[str]:
    cmd = [sys.executable, str(FEM), *BASE_FLAGS, "--max-nfev", str(max_nfev), "--results", str(result_dir)]
    for key, value in flags.items():
        cmd.extend([key, value])
    cmd.extend(["--patient-ids", *patient_ids])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="*", default=[name for name, _ in ARMS], help="arm names to run (default: all archived arms)")
    parser.add_argument("--include-baseline-equivalent", action="store_true", help="also allow nu_0p45 / fix_q0p80 / aspect_1p0 (identical to the baseline arm)")
    parser.add_argument("--max-nfev", type=int, default=48)
    parser.add_argument("--out", type=Path, default=OUT, help="output root (default: results/ct_reviewer_drivers_v6/assumption_sensitivity)")
    parser.add_argument("--dry-run", action="store_true", help="print the exact FEM commands without running them")
    parser.add_argument("--summarize-only", action="store_true", help="regenerate summary.json from existing records.jsonl files; no FEM")
    args = parser.parse_args()

    patient_ids = json.loads(SCENE_IDS.read_text(encoding="utf-8"))
    if len(patient_ids) != 18 or len(set(patient_ids)) != 18:
        parser.error(f"{SCENE_IDS} must list the 18 locked scene IDs")
    out_dir: Path = args.out
    if args.summarize_only:
        out_json = write_summary(out_dir, set(patient_ids))
        print("wrote", out_json)
        return

    catalogue = dict(ARMS)
    if args.include_baseline_equivalent:
        catalogue.update(dict(BASELINE_EQUIVALENT_ARMS))
    unknown = sorted(set(args.arms) - set(catalogue))
    if unknown:
        parser.error(f"unknown arm(s): {unknown}; known: {sorted(catalogue)}")
    selected = {name: catalogue[name] for name in args.arms}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, flags in selected.items():
        result_dir = out_dir / name
        cmd = build_command(result_dir, flags, patient_ids, args.max_nfev)
        print("RUN", name, flush=True)
        if args.dry_run:
            print(" ", subprocess.list2cmdline(cmd))
            continue
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / "run.log").open("a", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        records = result_dir / "records.jsonl"
        arm_summary = summarize_arm(records) if records.exists() else {"n": 0}
        print("DONE", name, "exit", proc.returncode, arm_summary, flush=True)
    if not args.dry_run:
        print("wrote", write_summary(out_dir, set(patient_ids)))


if __name__ == "__main__":
    main()
