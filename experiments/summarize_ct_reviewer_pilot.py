#!/usr/bin/env python3
"""Validate and summarize the paired CT reviewer FEM pilot records."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ct_reviewer_drivers_v1"
BASELINE = "w2_s0p05_p2"
ARMS = [
    "w1_s0p05_p2", "w2_s0_p2", "w2_s0p01_p2", "w2_s0p025_p2",
    "w2_s0p05_p2", "w2_s0p05_p4", "w2_s0p05_p6", "w2_s0p1_p2",
    "w2_s0p2_p2", "w3_s0p05_p2", "mismatch_force_p10_baseline",
]
METRICS = [
    "E_background_relative_error",
    "inclusion_ratio_relative_error",
    "force_scale_relative_error",
]


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def path_for(arm: str) -> Path:
    if arm == "mismatch_force_p10_baseline":
        return OUT / "model_mismatch" / arm / "records.jsonl"
    return OUT / "fem_paired" / arm / "records.jsonl"


def load_arm(arm: str, expected: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = path_for(arm)
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[str(row["patient_id"])].append(row)
        by_id[str(row["patient_id"])] = row
    duplicate_groups = {key: rows for key, rows in groups.items() if len(rows) > 1}
    duplicate_diffs = {}
    for key, rows in duplicate_groups.items():
        changed = sorted({field for row in rows for field in row} if len(rows) > 1 else set())
        changed = [field for field in changed if any(row.get(field) != rows[0].get(field) for row in rows[1:])]
        duplicate_diffs[key] = {
            "rows": len(rows),
            "changed_fields": changed,
            "records": rows,
        }
    nonidentical_retry_groups = [
        key for key, rows in duplicate_groups.items()
        if any(row != rows[0] for row in rows[1:])
    ]
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    summary_records = {str(row["patient_id"]): row for row in (summary or {}).get("records", [])}
    outcome_fields = METRICS + [
        "refinement_accepted", "screening_rejected", "eq_correlation_rejected",
        "relative_cost_reduction", "initial_cost", "cost", "termination_reason",
        "function_evaluations", "candidate_probe_count", "candidate_directions",
    ]
    summary_mismatches = {}
    for key, summary_row in summary_records.items():
        record = by_id.get(key)
        changed = [field for field in outcome_fields
                   if record is None or record.get(field) != summary_row.get(field)]
        if changed:
            summary_mismatches[key] = changed
    duplicate_outcome_diffs = {}
    for key, rows in duplicate_groups.items():
        changed = [field for field in outcome_fields
                   if any(row.get(field) != rows[0].get(field) for row in rows[1:])]
        duplicate_outcome_diffs[key] = changed
    audit = {
        "raw_rows": len(raw), "unique_scenes": len(by_id),
        "duplicate_groups": len(duplicate_groups),
        "extra_retry_rows": sum(len(rows) - 1 for rows in duplicate_groups.values()),
        "duplicate_changed_fields": {key: value["changed_fields"] for key, value in duplicate_diffs.items()},
        "duplicate_outcome_diffs": duplicate_outcome_diffs,
        "nonidentical_retry_groups": nonidentical_retry_groups,
        "summary_record_count": len(summary_records),
        "summary_outcome_mismatches": summary_mismatches,
        "summary_matches_last_record_for_all_ids": len(summary_records) == len(by_id) and not summary_mismatches,
        "missing_expected": sorted(expected - set(by_id)),
        "unexpected_ids": sorted(set(by_id) - expected),
        "summary_present": summary is not None,
    }
    return by_id, {"audit": audit, "summary": summary}


def main() -> None:
    expected = set(json.loads((OUT / "pilot_patient_ids.json").read_text(encoding="utf-8")))
    records: dict[str, dict[str, dict[str, Any]]] = {}
    details: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        records[arm], details[arm] = load_arm(arm, expected)

    arm_stats: dict[str, Any] = {}
    for arm, rows in records.items():
        summary = details[arm]["summary"]
        fem = summary.get("screened_fem_locked", {}) if summary else {}
        ratio_by_geometry: dict[str, list[float]] = defaultdict(list)
        for row in rows.values():
            ratio_by_geometry[str(row["geometry_id"])].append(float(row["inclusion_ratio_relative_error"]))
        geom_medians = {geom: median(values) for geom, values in sorted(ratio_by_geometry.items())}
        arm_stats[arm] = {
            "audit": details[arm]["audit"],
            "metrics_median": {
                metric: median([float(row[metric]) for row in rows.values()]) for metric in METRICS
            },
            "accepted": sum(bool(row.get("refinement_accepted")) for row in rows.values()),
            "screen_rejected": sum(bool(row.get("screening_rejected")) for row in rows.values()),
            "ratio_geometry_medians": geom_medians,
            "ratio_geometry_pass_25pct": f"{sum(value <= 0.25 for value in geom_medians.values())}/{len(geom_medians)}",
            "summary_n": fem.get("n"),
        }

    baseline = records[BASELINE]
    paired: dict[str, Any] = {}
    for arm, rows in records.items():
        if arm == BASELINE:
            continue
        ids = sorted(set(baseline) & set(rows))
        paired[arm] = {
            "paired_n": len(ids),
            "paired_ids_match_expected": set(ids) == expected,
            "metrics": {
                metric: {
                    "median_delta_vs_baseline": median([
                        float(rows[key][metric]) - float(baseline[key][metric]) for key in ids
                    ]),
                    "fraction_improved": (
                        sum(float(rows[key][metric]) < float(baseline[key][metric]) for key in ids) / len(ids)
                        if ids else None
                    ),
                }
                for metric in METRICS
            },
        }

    report = {
        "expected_scenes": len(expected), "arms": arm_stats,
        "paired_vs_baseline": paired,
        "all_arms_complete_and_id_matched": all(
            details[arm]["audit"]["unique_scenes"] == len(expected)
            and not details[arm]["audit"]["missing_expected"]
            and not details[arm]["audit"]["unexpected_ids"]
            and not any(details[arm]["audit"]["duplicate_outcome_diffs"].values())
            and details[arm]["audit"]["summary_present"]
            and details[arm]["audit"]["summary_matches_last_record_for_all_ids"]
            and details[arm]["audit"]["summary_record_count"] == len(expected)
            for arm in ARMS
        ),
    }
    target = OUT / "paired_pilot_audit_checked.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWROTE {target}")


if __name__ == "__main__":
    main()
