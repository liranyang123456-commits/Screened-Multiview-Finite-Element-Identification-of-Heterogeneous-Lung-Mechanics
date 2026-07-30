
"""Print the principal external CT benchmark summaries."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
METRICS = (
    "E_background_relative_error",
    "inclusion_ratio_relative_error",
    "center_error_normalized",
    "radius_relative_error",
)


def main() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    print("method\tE_background\tratio\tcenter\tradius")
    for method, summary in payload["methods"].items():
        metrics = summary["nested_descriptive"]["overall_scenarios"]
        values = [metrics[name].get("median") for name in METRICS]
        rendered = "\t".join(
            "NA" if value is None else f"{100 * float(value):.2f}%"
            for value in values
        )
        print(f"{method}\t{rendered}")


if __name__ == "__main__":
    main()
