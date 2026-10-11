from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_assumption_sensitivity_ct18 import summarize, summarize_arm


class AssumptionSummaryStatisticsTest(unittest.TestCase):
    def summary(self, values: list[float]) -> dict:
        rows = [
            {
                "E_background_relative_error": value,
                "inclusion_ratio_relative_error": value,
                "force_scale_relative_error": value,
                "refinement_accepted": index % 2 == 0,
            }
            for index, value in enumerate(values)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            return summarize(path)

    def test_even_sample_uses_both_middle_values(self) -> None:
        result = self.summary([0.10, 0.30])
        self.assertAlmostEqual(result["median_background_error"], 0.20)
        self.assertAlmostEqual(result["median_ratio_error"], 0.20)
        self.assertAlmostEqual(result["median_force_error"], 0.20)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accepted"], 1)

    def test_eighteen_scene_median_is_not_the_upper_middle_value(self) -> None:
        result = self.summary([index / 100 for index in range(18)])
        self.assertAlmostEqual(result["median_background_error"], 0.085)
        self.assertNotAlmostEqual(result["median_background_error"], 0.09)

    def test_odd_sample_uses_the_middle_value(self) -> None:
        result = self.summary([0.50, 0.10, 0.30])
        self.assertAlmostEqual(result["median_background_error"], 0.30)

    def test_archived_schema_reports_rounded_percent_and_hash(self) -> None:
        rows = [
            {
                "patient_id": f"scene_{index}",
                "E_background_relative_error": value,
                "inclusion_ratio_relative_error": value,
                "force_scale_relative_error": value,
                "refinement_accepted": True,
            }
            for index, value in enumerate([0.10, 0.30])
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = summarize_arm(path)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["median_background_error_percent"], 20.0)
        self.assertEqual(len(result["records_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
