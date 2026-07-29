from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multiseed_rejects_different_patient_cohorts(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "patient_id": "a",
                        "E_background_relative_error": 0.1,
                        "inclusion_ratio_relative_error": 0.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "patient_id": "b",
                        "E_background_relative_error": 0.1,
                        "inclusion_ratio_relative_error": 0.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluation" / "aggregate_multiseed.py"),
            "--inputs",
            str(first),
            str(second),
            "--seeds",
            "1",
            "2",
            "--out",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "identical patient cohort" in result.stderr
