"""Create a compact evidence summary for the expanded ION CT experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEARNING = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
DEFAULT_PHYSICS = ROOT / "results" / "ion_ct_expanded30_physics16" / "benchmark.json"
DEFAULT_OUTPUT = ROOT / "results" / "ion_ct_expanded_summary.json"
GATES = {
    "E_background_relative_error": 0.15,
    "inclusion_ratio_relative_error": 0.25,
    "center_error_normalized": 0.10,
    "radius_relative_error": 0.12,
}


def method_summary(payload: dict[str, Any], method: str) -> dict[str, Any]:
    rows = [row for row in payload["records"] if row["method"] == method]
    nested = payload["methods"][method]["nested_descriptive"]["overall_scenarios"]
    return {
        "entry_count": len(rows),
        "geometry_count": len({row["geometry_id"] for row in rows}),
        "medians": {
            metric: nested[metric].get("median")
            for metric in GATES
        },
        "gate_pass_rates": {
            metric: (
                sum(
                    float(row[metric]) <= threshold
                    for row in rows
                    if row.get(metric) is not None
                )
                / sum(row.get(metric) is not None for row in rows)
                if any(row.get(metric) is not None for row in rows)
                else None
            )
            for metric, threshold in GATES.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning", type=Path, default=DEFAULT_LEARNING)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    learning = json.loads(args.learning.read_text(encoding="utf-8"))
    physics = json.loads(args.physics.read_text(encoding="utf-8"))
    learning_methods = (
        "training_population",
        "ridge",
        "pls",
        "extra_trees",
        "pca_ridge",
        "mesh_gnn",
        "ours_hybrid_initializer",
    )
    physics_methods = (
        "fem_fixed_init",
        "fem_deterministic_multistart",
        "fem_learned_screened_map",
        "ours_ood_screened_fem",
        "fem_oracle_region_force",
    )
    fallback = [
        row
        for row in physics["records"]
        if row["method"] == "ours_ood_screened_fem"
    ]
    result = {
        "schema_version": 1,
        "geometry_protocol": {
            "total": 27,
            "train": 18,
            "validation": 3,
            "test": 6,
            "scenarios_per_geometry": 20,
            "total_scenarios": 540,
            "total_load_experiments": 2160,
        },
        "criteria": GATES,
        "learning_test_120": {
            method: method_summary(learning, method)
            for method in learning_methods
        },
        "controlled_physics_test_30": {
            method: method_summary(physics, method)
            for method in physics_methods
        },
        "ood_fallback": {
            "entry_count": len(fallback),
            "trigger_rate": (
                sum(bool(row.get("fallback_triggered")) for row in fallback)
                / len(fallback)
            ),
            "fixed_fem_selected_rate": (
                sum(bool(row.get("fallback_used")) for row in fallback)
                / len(fallback)
            ),
            "selection": "lower physical objective after train-support trigger",
        },
        "evidence_boundary": (
            "real de-identified CT geometry with synthetic materials, loads, "
            "motions, and ground truth; not patient material validation"
        ),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
