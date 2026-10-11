from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results/ion_ct_official_prior_screened_fem120_v1/records.jsonl"
OUTPUT = ROOT / "results/ion_ct_official_prior_screened_fem120_v1/geometry_influence_audit.json"

METRICS = {
    "background_error": "E_background_relative_error",
    "ratio_error": "inclusion_ratio_relative_error",
    "center_error": "center_error_normalized",
    "radius_error": "radius_relative_error",
    "force_error": "force_scale_relative_error",
}


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name, key in METRICS.items():
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[name] = median(values) if values else None
        out[f"{name}_n"] = len(values)
    out["accepted_fraction"] = sum(bool(row["refinement_accepted"]) for row in rows) / len(rows)
    out["screen_rejected_fraction"] = sum(bool(row["screening_rejected"]) for row in rows) / len(rows)
    out["n_scenes"] = len(rows)
    return out


def main() -> None:
    raw = RECORDS.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    by_geometry: dict[str, list[dict]] = {}
    for row in rows:
        by_geometry.setdefault(str(row["geometry_id"]), []).append(row)

    if len(rows) != 120 or len(by_geometry) != 6 or any(len(group) != 20 for group in by_geometry.values()):
        raise ValueError(f"Expected 120 rows, 6 geometries, 20 scenes each; got {len(rows)} / { {k: len(v) for k, v in by_geometry.items()} }")
    if len({row["patient_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate scene IDs in official-prior CT records")
    if any(row.get("ratio_source") != "official_2028_bins3" or not row.get("lock_ratio") for row in rows):
        raise ValueError("Unexpected calibration source or unlocked ratio in official-prior records")

    geometry = {gid: summarize(group) for gid, group in sorted(by_geometry.items())}
    pooled = summarize(rows)
    # Leave-one-geometry-out is descriptive influence analysis only: the locked
    # predictions are held fixed; no model, calibration, or threshold is refit.
    loo = {}
    for excluded in sorted(by_geometry):
        retained = [row for row in rows if row["geometry_id"] != excluded]
        cluster_metrics = {
            name: median([float(geometry[gid][name]) for gid in geometry if gid != excluded])
            for name in [*METRICS, "accepted_fraction", "screen_rejected_fraction"]
        }
        loo[excluded] = {
            "excluded_geometry": excluded,
            "remaining_geometry_count": 5,
            "remaining_scene_count": len(retained),
            "scene_pooled": summarize(retained),
            "equal_geometry_weighted_median": cluster_metrics,
        }

    metrics = [*METRICS, "accepted_fraction", "screen_rejected_fraction"]
    influence_ranges = {}
    for metric in metrics:
        values = [entry["scene_pooled"][metric] for entry in loo.values()]
        values = [value for value in values if value is not None]
        full_value = pooled.get(metric)
        influence_ranges[metric] = {
            "leave_one_out_min": min(values) if values else None,
            "leave_one_out_max": max(values) if values else None,
            "full_cohort_value": full_value,
            "max_absolute_shift": max(abs(value - full_value) for value in values) if values and full_value is not None else None,
        }

    payload = {
        "protocol": "Locked-outcome-only geometry influence analysis of the completed official-prior screened-FEM CT test. No fitting, calibration, threshold selection, or prediction changes are performed.",
        "records_path": str(RECORDS.relative_to(ROOT)),
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "n_scenes": len(rows),
        "n_geometries": len(by_geometry),
        "scenes_per_geometry": 20,
        "calibration_source": "official_2028_bins3",
        "full_cohort_scene_pooled": pooled,
        "geometry_level_medians": geometry,
        "leave_one_geometry_out": loo,
        "leave_one_out_influence_ranges": influence_ranges,
        "limitations": "This is a descriptive leave-one-geometry-out influence analysis of already-locked predictions, not an independent validation, a recalibration, or a replacement for cluster-level uncertainty intervals. Six geometries are too few for broad population inference.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "records_sha256": payload["records_sha256"],
        "full_cohort": pooled,
        "per_geometry": geometry,
        "loo_influence_ranges": influence_ranges,
    }, indent=2))


if __name__ == "__main__":
    main()
