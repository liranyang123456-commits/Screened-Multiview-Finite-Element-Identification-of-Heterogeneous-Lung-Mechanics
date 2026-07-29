from __future__ import annotations

import math

import pytest
import torch

import lung_inverse_rendering.evaluate_sim_lung_v2 as evaluation


def simple_problem(*, measured: bool = True) -> tuple[dict, list[dict]]:
    nodes = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    force = torch.zeros(3, 6, dtype=torch.float64)
    force[2] = torch.tensor([10.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    observation = torch.zeros(3, 2, 3, dtype=torch.float64)
    observation[2] = (force[2] / 5000.0).view(2, 3)
    experiment = {
        "patient_id": "patient_000",
        "forces": force * (10.0 if measured else 1.0),
        "surface_motion_observed": observation,
        "inclusion_center": torch.zeros(3, dtype=torch.float64),
        "inclusion_radius": 0.1,
    }
    if measured:
        experiment["measured_forces"] = force
    scene = {
        "nodes": nodes,
        "elems": torch.tensor([[0, 1]], dtype=torch.long),
        "surface_node_ids": torch.tensor([0, 1]),
        "surface_tris": torch.empty((0, 3), dtype=torch.long),
        "fixed": torch.empty(0, dtype=torch.long),
        "nu_true": torch.tensor(0.3, dtype=torch.float64),
    }
    return scene, [experiment]


@pytest.fixture
def proportional_fem(monkeypatch: pytest.MonkeyPatch) -> None:
    def solve(
        nodes: torch.Tensor,
        elems: torch.Tensor,
        E_nodes: torch.Tensor,
        nu: torch.Tensor,
        force: torch.Tensor,
        fixed: torch.Tensor,
    ) -> torch.Tensor:
        return force.to(torch.float64) / E_nodes.mean()

    monkeypatch.setattr(evaluation, "solve_nh_heterogeneous", solve)


def test_informative_force_prior_pulls_scale_to_one(proportional_fem: None) -> None:
    scene, experiments = simple_problem()
    result = evaluation.fit_patient(
        scene,
        experiments,
        observation_key="surface_motion_observed",
        max_nfev=100,
        optimize_force_scale=True,
        force_prior_sigma=0.02,
        initial_E_background=7000.0,
        initial_force_scale=1.4,
        soft_occupancy=torch.zeros(2),
    )
    assert abs(math.log(result["force_scale"])) < 0.08
    assert result["force_scale_identifiability"] == "informative_prior"
    assert result["diagnostic_function_evaluations"] == 9


def test_force_fit_without_prior_is_rejected(proportional_fem: None) -> None:
    scene, experiments = simple_problem()
    with pytest.raises(ValueError, match="not identifiable"):
        evaluation.fit_patient(
            scene,
            experiments,
            observation_key="surface_motion_observed",
            max_nfev=2,
            optimize_force_scale=True,
            soft_occupancy=torch.zeros(2),
        )


def test_external_calibration_prefers_measured_forces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, experiments = simple_problem(measured=True)
    seen_forces: list[torch.Tensor] = []

    def solve(
        nodes: torch.Tensor,
        elems: torch.Tensor,
        E_nodes: torch.Tensor,
        nu: torch.Tensor,
        force: torch.Tensor,
        fixed: torch.Tensor,
    ) -> torch.Tensor:
        seen_forces.append(force.clone())
        return force.to(torch.float64) / E_nodes.mean()

    monkeypatch.setattr(evaluation, "solve_nh_heterogeneous", solve)
    result = evaluation.fit_patient(
        scene,
        experiments,
        observation_key="surface_motion_observed",
        max_nfev=1,
        force_calibration_factor=1.25,
        soft_occupancy=torch.zeros(2),
    )
    assert torch.allclose(seen_forces[0], experiments[0]["measured_forces"][2] * 1.25)
    assert result["force_scale"] == 1.0
    assert result["force_scale_identifiability"] == "externally_calibrated"


def test_legacy_two_parameter_forces_field_still_works(
    proportional_fem: None,
) -> None:
    scene, experiments = simple_problem(measured=False)
    result = evaluation.fit_patient(
        scene,
        experiments,
        observation_key="surface_motion_observed",
        max_nfev=8,
        soft_occupancy=torch.zeros(2),
    )
    assert result["function_evaluations"] <= 8
    assert result["force_scale"] == 1.0
    assert result["diagnostic_function_evaluations"] == 0


def test_oracle_force_mode_uses_simulator_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, experiments = simple_problem(measured=True)
    seen_forces: list[torch.Tensor] = []

    def solve(
        nodes: torch.Tensor,
        elems: torch.Tensor,
        E_nodes: torch.Tensor,
        nu: torch.Tensor,
        force: torch.Tensor,
        fixed: torch.Tensor,
    ) -> torch.Tensor:
        seen_forces.append(force.clone())
        return force.to(torch.float64) / E_nodes.mean()

    monkeypatch.setattr(evaluation, "solve_nh_heterogeneous", solve)
    evaluation.fit_patient(
        scene,
        experiments,
        observation_key="surface_motion_observed",
        max_nfev=1,
        use_true_forces=True,
        soft_occupancy=torch.zeros(2),
    )
    assert torch.allclose(seen_forces[0], experiments[0]["forces"][2])
