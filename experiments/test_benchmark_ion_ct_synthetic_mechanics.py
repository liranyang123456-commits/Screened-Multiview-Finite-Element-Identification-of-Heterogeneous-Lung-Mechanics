from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmark_ion_ct_synthetic_mechanics import (
    METRICS,
    compose_hybrid_initializer,
    compose_ood_screened_fem,
    evidence_tier,
    geometry_aggregate,
    geometry_cluster_bootstrap,
    load_physics_records,
    normalized_external_manifest,
    paired_comparisons,
    physics_commands,
)


def _records(method: str, offset: float = 0.0) -> list[dict]:
    rows = []
    for geometry_index in range(3):
        for scenario_index in range(2):
            value = 0.1 * (geometry_index + 1) + 0.01 * scenario_index + offset
            rows.append(
                {
                    "method": method,
                    "evidence_tier": evidence_tier(method),
                    "geometry_id": f"geometry_{geometry_index}",
                    "scenario_id": f"scenario_{scenario_index}",
                    **{metric: value for metric in METRICS},
                }
            )
    return rows


def test_external_manifest_requires_three_external_geometry_clusters(
    tmp_path: Path,
) -> None:
    entries = [
        {
            "geometry_id": f"g{geometry}",
            "scenario_id": f"s{scenario}",
            "split": "external_test",
            "experiments": [],
        }
        for geometry in range(3)
        for scenario in range(2)
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": "sim_lung_v2_external_v1", "entries": entries}),
        encoding="utf-8",
    )
    _, normalized = normalized_external_manifest(tmp_path)
    assert len(normalized) == 6
    assert {row["split"] for row in normalized} == {"test"}
    assert normalized[0]["patient_id"] == "g0/s0"
    _, limited = normalized_external_manifest(
        tmp_path,
        scenarios_per_geometry=1,
    )
    assert len(limited) == 3
    assert {row["scenario_id"] for row in limited} == {"s0"}


def test_external_manifest_filters_training_geometries_and_accepts_six_tests(
    tmp_path: Path,
) -> None:
    entries = [
        {
            "geometry_id": f"g{geometry}",
            "scenario_id": "s0",
            "split": (
                "train"
                if geometry < 18
                else "val"
                if geometry < 21
                else "test"
            ),
            "experiments": [],
        }
        for geometry in range(27)
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": "sim_lung_v2_external_v2", "patients": entries}),
        encoding="utf-8",
    )
    _, normalized = normalized_external_manifest(tmp_path)
    assert len(normalized) == 6
    assert {row["geometry_id"] for row in normalized} == {
        f"g{index}" for index in range(21, 27)
    }


def test_cluster_statistics_and_pairing_preserve_geometry() -> None:
    reference = _records("pca_ridge")
    candidate = _records("ridge", 0.02)
    interval = geometry_cluster_bootstrap(
        reference, METRICS[0], replicates=100
    )
    assert interval["cluster_count"] == 3
    assert len(geometry_aggregate(reference)) == 3
    paired = paired_comparisons(
        reference + candidate, replicates=100
    )
    metric = paired["ridge"]["metrics"][METRICS[0]]
    assert metric["paired_entry_count"] == 6
    assert metric["geometry_cluster_count"] == 3
    assert metric["median_difference"] == pytest.approx(0.02)


def test_ood_screening_selects_fixed_fem_fallback() -> None:
    records = [
        {
            "entry_id": "g0/s0",
            "method": "pca_ridge",
            "ood_detected": True,
            "ood_score": 0.4,
        },
        {
            "entry_id": "g0/s0",
            "method": "fem_fixed_init",
            "E_background_relative_error": 0.1,
        },
        {
            "entry_id": "g0/s0",
            "method": "fem_learned_screened_map",
            "E_background_relative_error": 0.3,
            "screening_rejected": False,
        },
    ]
    selected = compose_ood_screened_fem(records)
    assert selected[0]["method"] == "ours_ood_screened_fem"
    assert selected[0]["fallback_used"] is True
    assert selected[0]["E_background_relative_error"] == 0.1


def test_hybrid_initializer_uses_mesh_material_and_pca_radius() -> None:
    records = [
        {
            "entry_id": "g0/s0",
            "method": "mesh_gnn",
            "E_background_relative_error": 0.1,
            "radius_fraction_estimated": 0.3,
            "radius_relative_error": 0.5,
        },
        {
            "entry_id": "g0/s0",
            "method": "pca_ridge",
            "radius_fraction_estimated": 0.2,
            "radius_relative_error": 0.05,
        },
    ]
    hybrid = compose_hybrid_initializer(records)[0]
    assert hybrid["method"] == "ours_hybrid_initializer"
    assert hybrid["E_background_relative_error"] == 0.1
    assert hybrid["radius_fraction_estimated"] == 0.2
    assert hybrid["radius_relative_error"] == 0.05


def test_physics_config_separates_tiers_and_controls_multistart_budget(
    tmp_path: Path,
) -> None:
    predictions = {
        "fixed": tmp_path / "fixed.json",
        "learned": tmp_path / "learned.json",
        "oracle": tmp_path / "oracle.json",
        **{
            f"multistart_{index}": tmp_path / f"start_{index}.json"
            for index in range(4)
        },
    }
    jobs = physics_commands(
        tmp_path / "manifest.json",
        tmp_path,
        predictions,
        multistart_total_budget=96,
    )
    starts = [
        job
        for job in jobs
        if job["method"] == "fem_deterministic_multistart"
    ]
    assert len(starts) == 4
    assert sum(job["max_nfev_per_entry"] for job in starts) == 96
    assert all(job["evidence_tier"] == "secondary" for job in starts)
    oracle = next(job for job in jobs if job["method"] == "fem_oracle_region_force")
    assert oracle["evidence_tier"] == "oracle"
    assert evidence_tier("pca_ridge") == "common_input"
    assert evidence_tier("training_population") == "secondary"


def test_multistart_results_select_lowest_cost_and_sum_budget(
    tmp_path: Path,
) -> None:
    paths = []
    for index, cost in enumerate((3.0, 1.0, 2.0, 4.0)):
        path = tmp_path / f"result_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark_method": "fem_deterministic_multistart",
                    "records": [
                        {
                            "patient_id": "g0/s0",
                            "cost": cost,
                            "function_evaluations": 24,
                            "E_background_relative_error": cost / 10.0,
                            "inclusion_ratio_relative_error": cost / 20.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    records = load_physics_records(paths)
    assert len(records) == 1
    assert records[0]["cost"] == 1.0
    assert records[0]["function_evaluations"] == 96
    assert records[0]["deterministic_starts_completed"] == 4
