"""Structural mismatch audits on the locked 18 CT scenes.

Modes regenerate observations with a different truth structure using the
production isotropic solver, then fit the production single-sphere isotropic
model. Post-fit cost is saved so a residual-based mismatch flag can be tested.

  two_inclusions : truth has a second stiff sphere (same ratio) at a mirrored center
  diffuse        : truth is a smooth Gaussian stiffness bump (no hard boundary)
  sliding_bc     : truth posterior face slides in-plane (only normal DOF fixed,
                   three anchor nodes pinned); fitter keeps the full fixation
  matched        : control re-run through the same code path (no structural change)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_screened_fem_locked import (  # noqa: E402
    CT_DATASET,
    CT_HYBRID,
    _index_records,
    _normalize_id,
    _official_2028_ratio_index,
    _official_ratio_key,
    _predicted_region,
)
from lung_inverse_rendering.evaluate_sim_lung_v2 import (  # noqa: E402
    fit_patient,
    force_scale_target,
    load_patient,
    relative_error,
)
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous  # noqa: E402
from rendering.gaussian_pbr import project  # noqa: E402

IDS = ROOT / "results" / "ct_reviewer_drivers_v6" / "paired_scene_ids.json"
OUT = ROOT / "results" / "ct_reviewer_drivers_v6" / "structural_mismatch"


def _truth_field(scene: dict, experiment: dict, mode: str) -> torch.Tensor:
    nodes = scene["nodes"]
    center = experiment["inclusion_center"].to(torch.float64)
    radius = float(experiment["inclusion_radius"])
    E_bg = float(experiment["E_background"])
    E_inc = float(experiment["E_inclusion"])
    base = make_heterogeneous_E_field(nodes, center, radius, E_bg, E_inc)
    if mode in {"matched", "sliding_bc"}:
        return base
    if mode == "two_inclusions":
        lo = nodes.min(dim=0).values
        hi = nodes.max(dim=0).values
        mirrored = center.clone()
        mirrored[0] = lo[0] + hi[0] - center[0]
        second = make_heterogeneous_E_field(nodes, mirrored, radius, E_bg, E_inc)
        return torch.maximum(base, second)
    if mode == "diffuse":
        d = torch.linalg.vector_norm(nodes - center, dim=1)
        # Gaussian bump with same peak contrast and sigma = radius
        weight = torch.exp(-0.5 * (d / radius).square())
        return E_bg + (E_inc - E_bg) * weight
    raise ValueError(mode)


def _sliding_fixed(scene: dict) -> torch.Tensor:
    nodes = scene["nodes"].to(torch.float64)
    back_cut = torch.quantile(nodes[:, 2], 0.80)
    back = torch.where(nodes[:, 2] >= back_cut)[0]
    z_only = 3 * back + 2
    # pin three well-separated back nodes in-plane to remove rigid modes
    order = torch.argsort(nodes[back, 0])
    anchors = back[order[[0, len(order) // 2, len(order) - 1]]]
    pins = torch.cat([3 * anchors, 3 * anchors + 1])
    return torch.unique(torch.cat([z_only, pins]))


def _reproject(scene: dict, experiments: list[dict], mode: str) -> dict:
    intrinsics = experiments[0]["render_intrinsics"]
    focal = float(intrinsics["focal_px"])
    height = int(intrinsics["height_px"])
    width = int(intrinsics["width_px"])
    surface = scene["surface_node_ids"]
    nodes = scene["nodes"]
    fixed = _sliding_fixed(scene) if mode == "sliding_bc" else scene["fixed"]
    gaps = []
    copied = []
    for experiment in experiments:
        fields = dict(experiment)
        observed = fields["image_uv_deformed_multiview_seq"].to(torch.float64).clone()
        E_truth = _truth_field(scene, experiment, mode)
        E_ref = make_heterogeneous_E_field(
            nodes, experiment["inclusion_center"], float(experiment["inclusion_radius"]),
            float(experiment["E_background"]), float(experiment["E_inclusion"]),
        )
        for frame in range(observed.shape[0]):
            force = experiment["forces"][frame].to(torch.float64)
            if float(force.abs().max()) == 0.0:
                continue
            u = solve_nh_heterogeneous(nodes, scene["elems"], E_truth, scene["nu_true"], force, fixed)
            if frame == 2:
                u_ref = solve_nh_heterogeneous(nodes, scene["elems"], E_ref, scene["nu_true"], force, scene["fixed"])
                gaps.append(float((u - u_ref).norm()) / (float(u_ref.norm()) + 1e-12))
            points = (nodes + u.view(-1, 3))[surface]
            for view in range(observed.shape[1]):
                uv, _, _ = project(points, experiment["poses_multiview"][frame, view], focal=focal, H=height, W=width)
                observed[frame, view] = uv.to(observed.dtype)
        fields["image_uv_deformed_multiview_seq"] = observed
        copied.append(fields)
    gaps = sorted(gaps)
    return {"experiments": copied, "peak_gap": gaps[len(gaps) // 2] if gaps else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("matched", "two_inclusions", "diffuse", "sliding_bc"), required=True)
    args = parser.parse_args()
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    manifest = json.loads((CT_DATASET / "manifest.json").read_text(encoding="utf-8"))
    rows = {r["patient_id"]: r for r in manifest["patients"]}
    predictions = _index_records(CT_HYBRID, method="ours_hybrid_initializer")
    official_ratios = _official_2028_ratio_index()
    dest = OUT / args.mode
    dest.mkdir(parents=True, exist_ok=True)
    rec_path = dest / "records.jsonl"
    done = {}
    if rec_path.exists():
        done = {json.loads(l)["patient_id"]: 1 for l in rec_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    with rec_path.open("a", encoding="utf-8") as handle:
        for pid in ids:
            if pid in done:
                print("skip", pid, flush=True)
                continue
            scene, experiments = load_patient(CT_DATASET, rows[pid])
            rep = _reproject(scene, experiments, args.mode)
            prior = predictions[_normalize_id(pid)]
            official = official_ratios.get(_official_ratio_key(rows[pid])) or prior
            center, radius = _predicted_region(scene, prior)
            sig_E = max(0.03, 2.0 * float(prior.get("log_E_std", 0.06)))
            sig_r = max(0.05, 2.0 * float(official.get("log_ratio_std", 0.23)))
            est = fit_patient(
                scene, rep["experiments"], observation_key="image_tracks", max_nfev=48,
                inclusion_center=center, inclusion_radius=radius, track_noise_px=0.05,
                tracking_error_model="iid", use_all_track_frames=True, temporal_track_regression=True,
                track_smoothing_iterations=2, multiview_tracks=True, minimum_track_confidence=0.2,
                region_track_weight=8.0, initial_E_background=float(prior["E_background_estimated"]),
                initial_ratio=float(official["inclusion_ratio_estimated"]),
                search_half_width_log=(max(0.15, 2.0 * sig_E), 0.0), optimize_force_scale=True,
                force_prior_sigma=0.05, material_prior_sigma=(sig_E, sig_r), material_prior_weight=0.05,
                material_prior_center=(float(prior["E_background_estimated"]), float(official["inclusion_ratio_estimated"])),
                screening_metric="map", objective_metric="map", screening_minimum_cost_reduction=0.05,
                lock_ratio=True, eq_correlation_rollback=0.85, candidate_probe_count=2,
            )
            record = {
                "patient_id": pid,
                "mode": args.mode,
                "peak_displacement_relative_gap": rep["peak_gap"],
                "E_background_relative_error": relative_error(est["E_background"], float(experiments[0]["E_background"])),
                "inclusion_ratio_relative_error": relative_error(est["inclusion_ratio"], float(experiments[0]["inclusion_ratio"])),
                "force_scale_relative_error": relative_error(est["force_scale"], force_scale_target(experiments)),
                "refinement_accepted": bool(est.get("refinement_accepted")),
                "cost": float(est.get("cost", float("nan"))),
                "initial_cost": float(est.get("initial_cost", float("nan"))),
                "function_evaluations": int(est.get("function_evaluations", 0)),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(pid, round(record["E_background_relative_error"], 4), round(record["peak_displacement_relative_gap"], 4), round(record["cost"], 4), flush=True)


if __name__ == "__main__":
    main()
