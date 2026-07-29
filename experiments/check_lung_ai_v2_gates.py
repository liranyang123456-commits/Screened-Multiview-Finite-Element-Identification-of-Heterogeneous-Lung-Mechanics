"""Evaluate frozen publication gates without tuning on test records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def nested(payload: dict | None, *keys: str):
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def first(*values):
    return next((value for value in values if value is not None), None)


def candidate_method_summary(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return None
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        return payload
    reference = payload.get("reference_method")
    candidates = [
        summary for name, summary in methods.items() if name != reference
    ]
    return candidates[0] if len(candidates) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-metrics", type=Path)
    parser.add_argument("--fem-metrics", type=Path)
    parser.add_argument("--multiseed", type=Path)
    parser.add_argument("--ood-metrics", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    image = load(args.image_metrics)
    fem = load(args.fem_metrics)
    fem_candidate = candidate_method_summary(fem)
    seeds = load(args.multiseed)
    ood = load(args.ood_metrics)
    values = {
        "image_E_error": first(
            nested(image, "test", "E_background_median_relative_error"),
            nested(image, "metrics", "E_background_median_relative_error"),
        ),
        "image_ratio_error": first(
            nested(image, "test", "inclusion_ratio_median_relative_error"),
            nested(image, "metrics", "inclusion_ratio_median_relative_error"),
        ),
        "center_error": first(
            nested(image, "test", "center_error_normalized_median"),
            nested(image, "metrics", "center_error_normalized_median"),
        ),
        "radius_error": first(
            nested(image, "test", "radius_relative_error_median"),
            nested(image, "metrics", "radius_relative_error_median"),
        ),
        "E_coverage_90": first(
            nested(image, "test", "log_E_uncertainty", "coverage", "90"),
            nested(image, "metrics", "log_E_uncertainty", "coverage", "90"),
        ),
        "ratio_coverage_90": first(
            nested(image, "test", "log_ratio_uncertainty", "coverage", "90"),
            nested(image, "metrics", "log_ratio_uncertainty", "coverage", "90"),
        ),
        "fem_E_error": nested(
            fem_candidate, "E_background_median_relative_error"
        ),
        "fem_ratio_error": nested(
            fem_candidate, "inclusion_ratio_median_relative_error"
        ),
        "fem_success": nested(fem_candidate, "optimizer_success_rate"),
        "fem_evaluation_reduction": nested(fem, "evaluation_reduction"),
        "seed_count": nested(seeds, "seed_count"),
        "ood_success_rate": nested(ood, "geometry_forward_success_rate"),
    }
    rules = {
        "image_E_error": ("max", 0.15),
        "image_ratio_error": ("max", 0.25),
        "center_error": ("max", 0.10),
        "radius_error": ("max", 0.12),
        "E_coverage_90": ("range", (0.85, 0.95)),
        "ratio_coverage_90": ("range", (0.85, 0.95)),
        "fem_E_error": ("max", 0.10),
        "fem_ratio_error": ("max", 0.25),
        "fem_success": ("min", 0.90),
        "fem_evaluation_reduction": ("min", 0.30),
        "seed_count": ("min", 5),
        "ood_success_rate": ("min", 0.90),
    }
    gates = {}
    for name, (kind, threshold) in rules.items():
        value = values[name]
        if value is None:
            gates[name] = {"passed": False, "reason": "missing", "value": None}
        elif kind == "max":
            gates[name] = {
                "passed": value <= threshold,
                "value": value,
                "threshold": threshold,
            }
        elif kind == "min":
            gates[name] = {
                "passed": value >= threshold,
                "value": value,
                "threshold": threshold,
            }
        else:
            lower, upper = threshold
            gates[name] = {
                "passed": lower <= value <= upper,
                "value": value,
                "threshold": [lower, upper],
            }
    result = {
        "publication_ready": all(row["passed"] for row in gates.values()),
        "gates": gates,
        "evidence_boundary": (
            "Synthetic material ground truth; real ION CT contributes geometry-domain "
            "stability only."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
