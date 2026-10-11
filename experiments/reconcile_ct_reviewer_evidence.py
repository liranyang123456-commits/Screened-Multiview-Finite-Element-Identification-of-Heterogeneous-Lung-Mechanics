#!/usr/bin/env python3
"""Recover the exact pilot subset and summaries from auditable local result files."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "ct_reviewer_drivers_v6"
RUN_ROOT = BASE / "ct_reviewer_drivers_v6"
OUT = BASE / "reconciled_paired_experiments.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def deduplicate_records(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep the latest checkpoint per patient and audit retry outcome identity."""
    by_id: dict[str, list[dict]] = {}
    for row in rows:
        by_id.setdefault(str(row["patient_id"]), []).append(row)
    canonical = [items[-1] for items in by_id.values()]
    metric_keys = (
        "E_background_relative_error",
        "inclusion_ratio_relative_error",
        "force_scale_relative_error",
        "refinement_accepted",
        "screening_rejected",
    )
    changed = []
    for patient_id, items in by_id.items():
        signatures = {tuple(item.get(key) for key in metric_keys) for item in items}
        if len(signatures) > 1:
            changed.append(patient_id)
    return canonical, {
        "raw_record_count": len(rows),
        "unique_patient_count": len(by_id),
        "duplicate_retry_count": len(rows) - len(by_id),
        "retry_outcome_mismatch_patient_ids": changed,
    }


def summarize(path: Path) -> dict:
    raw_rows = load_records(path)
    rows, retry_audit = deduplicate_records(raw_rows)
    output = {
        "records_sha256": sha256(path),
        **retry_audit,
        "n": len(rows),
        "unique_patient_ids": len({row["patient_id"] for row in rows}),
        "patient_ids": [row["patient_id"] for row in rows],
        "accepted": sum(bool(row.get("refinement_accepted")) for row in rows),
        "screen_rejected": sum(bool(row.get("screening_rejected")) for row in rows),
        "median_function_evaluations": statistics.median(float(row["function_evaluations"]) for row in rows),
        "median_wall_time_seconds": statistics.median(float(row["wall_time_seconds"]) for row in rows),
    }
    for key, label in (
        ("E_background_relative_error", "background_error"),
        ("inclusion_ratio_relative_error", "ratio_error"),
        ("force_scale_relative_error", "force_error"),
    ):
        output[f"median_{label}"] = statistics.median(float(row[key]) for row in rows)
    output["geometries"] = sorted({str(row.get("geometry_id")) for row in rows})
    return output


def main() -> None:
    expected_ids = set(json.loads((BASE / "paired_scene_ids.json").read_text(encoding="utf-8")))
    controls = {}
    for path in sorted((RUN_ROOT / "paired_controls").glob("*/records.jsonl")):
        manifest_path = path.parent / "driver_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        controls[path.parent.name] = {
            **summarize(path),
            "factors": manifest.get("factors", {}),
            "manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
            "source_records": str(path.relative_to(ROOT)),
        }

    # Reconcile the complete mismatch suite by its distinct perturbation factor,
    # not by historical directory names. Several old/extended folders are
    # duplicate reruns of the same setting and must not inflate the arm count.
    mismatch_sources = {
        "force_m20": BASE / "model_mismatch_completed" / "force_m20",
        "force_m10": RUN_ROOT / "model_mismatch" / "mismatch_force_m10_baseline",
        "force_p5": BASE / "model_mismatch_completed" / "force_p5",
        "force_p10": BASE / "model_mismatch_completed" / "force_p10",
        "force_p20": BASE / "model_mismatch_completed" / "force_p20",
        "pose_yaw_m10": BASE / "model_mismatch_completed" / "pose_yaw_m10_v2",
        "pose_yaw_m5": BASE / "model_mismatch_completed" / "pose_yaw_m5_v1",
        "pose_yaw_p5": BASE / "model_mismatch_completed" / "pose_yaw_p5_v1",
        "pose_yaw_p10": BASE / "model_mismatch_completed" / "pose_yaw_p10_v1",
    }
    mismatches = {}
    for arm, run_dir in mismatch_sources.items():
        records_path = run_dir / "records.jsonl"
        if not records_path.exists():
            raise FileNotFoundError(f"missing canonical mismatch records for {arm}: {records_path}")
        raw_rows = load_records(records_path)
        rows, retry_audit = deduplicate_records(raw_rows)
        observed_ids = {str(row["patient_id"]) for row in rows}
        if observed_ids != expected_ids:
            raise ValueError(f"incomplete paired mismatch arm {arm}: {len(observed_ids)} scene IDs")
        manifest_candidates = [run_dir / "run_manifest.json", run_dir / "driver_manifest.json"]
        manifest_path = next((path for path in manifest_candidates if path.exists()), None)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else {}
        mismatches[arm] = {
            **summarize(records_path),
            "factors": manifest.get("factors", {
                key: manifest.get(key) for key in ("model_force_bias", "pose_rotation_error_deg")
                if key in manifest
            }),
            "protocol": manifest,
            "manifest_sha256": sha256(manifest_path) if manifest_path else None,
            "source_records": str(records_path.relative_to(ROOT)),
            "deduplicated_retry_audit": retry_audit,
        }

    sensitivities = {}
    recovered = BASE.parent / "ct_reviewer_drivers_v1" / "recovered_raw" / "fem_paired"
    for path in sorted(recovered.glob("*/records.jsonl")):
        sensitivities[path.parent.name] = summarize(path)

    weight_sweep = {}
    weight_root = BASE / "material_prior_weight_sweep"
    for manifest_path in sorted(weight_root.glob("w*/run_manifest.json")):
        run_dir = manifest_path.parent
        records_path = run_dir / "records.jsonl"
        if not records_path.exists():
            continue
        rows, retry_audit = deduplicate_records(load_records(records_path))
        observed_ids = {str(row["patient_id"]) for row in rows}
        if observed_ids != expected_ids:
            continue
        protocol = json.loads(manifest_path.read_text(encoding="utf-8"))
        weight_sweep[run_dir.name] = {
            **summarize(records_path),
            "protocol": protocol,
            "manifest_sha256": sha256(manifest_path),
            "source_records": str(records_path.relative_to(ROOT)),
            "deduplicated_retry_audit": retry_audit,
        }

    resampling = {}
    for path in sorted(BASE.glob("repeated_outer_audit_seed*_1000.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        runs = result.get("runs", [])
        repeat_ids = {int(row["repeat"]) for row in runs}
        if len(runs) != int(result.get("repeats", -1)) or len(repeat_ids) != len(runs):
            raise ValueError(f"incomplete or duplicate outer resampling repeats in {path}")
        resampling[path.stem] = {
            **result,
            "result_sha256": sha256(path),
            "verified_repeat_count": len(runs),
            "unique_repeat_count": len(repeat_ids),
            "source": str(path.relative_to(ROOT)),
        }

    payload = {
        "protocol_note": "All FEM rows are computed on the same 18 explicit locked CT test scenes from six geometries; calibration resampling uses training geometries only.",
        "paired_controls": controls,
        "model_mismatch": mismatches,
        "sensitivity_arms": sensitivities,
        "material_prior_weight_sweep": weight_sweep,
        "repeated_outer_geometry_resampling": resampling,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "control_arms": list(controls),
        "mismatch_arms": {key: (value.get("n_records", value.get("n")), value.get("median_relative_error_percent")) for key, value in mismatches.items()},
        "sensitivity_arms": list(sensitivities),
    }, indent=2))


if __name__ == "__main__":
    main()
