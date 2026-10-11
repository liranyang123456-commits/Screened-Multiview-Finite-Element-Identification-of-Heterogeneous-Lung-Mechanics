"""Ellipsoid-aspect assumption-sensitivity arm sanity (no FEM runs).

1. Compare aspect_1p5 / aspect_2p0 records against the paired 18-scene baseline
   w2_s0p05_p2: count scenes whose E_background_estimated (and other fields)
   are bitwise identical.
2. Recompute, for every scene, the hard inclusion-node set produced by
   ``_ellipsoid_occupancy`` (aspect 1.5 / 2.0) versus the sphere occupancy
   used by the baseline (``norm(nodes - center) < radius``) with the SAME
   predicted centre/radius, on the 256-node patch; report whether the node set
   changes.  Also reports the side effect that passing ``soft_occupancy`` makes
   ``fit_patient`` fall back to the TRUE inclusion centre/radius for the
   region-attention weights (center/radius are None in that call).

Writes results/ct_reviewer_drivers_v6/assumption_sensitivity/aspect_arm_sanity.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_screened_fem_locked import (  # noqa: E402
    CT_DATASET,
    CT_HYBRID,
    _ellipsoid_occupancy,
    _index_records,
    _normalize_id,
    _predicted_region,
)
from lung_inverse_rendering.evaluate_sim_lung_v2 import load_patient, material_occupancy  # noqa: E402

SENS = ROOT / "results" / "ct_reviewer_drivers_v6" / "assumption_sensitivity"
BASELINE = ROOT / "results" / "ct_reviewer_drivers_v6" / "ct_reviewer_drivers_v6" / "fem_paired" / "w2_s0p05_p2" / "records.jsonl"
ARMS = {"aspect_1p5": (SENS / "aspect_1p5" / "records.jsonl", 1.5), "aspect_2p0": (SENS / "aspect_2p0" / "records.jsonl", 2.0)}
OUT = SENS / "aspect_arm_sanity.json"
COMPARE_FIELDS = (
    "E_background_estimated",
    "inclusion_ratio_estimated",
    "force_scale_estimated",
    "function_evaluations",
    "refinement_accepted",
    "screening_rejected",
    "cost",
    "initial_cost",
    "relative_cost_reduction",
)


def _read(path: Path) -> dict[str, dict]:
    return {str(json.loads(l)["patient_id"]): json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def main() -> None:
    baseline = _read(BASELINE)
    manifest = json.loads((CT_DATASET / "manifest.json").read_text(encoding="utf-8"))
    rows = {str(r["patient_id"]): r for r in manifest["patients"]}
    predictions = _index_records(CT_HYBRID, method="ours_hybrid_initializer")
    out: dict = {
        "description": __doc__.strip(),
        "baseline_records": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "baseline_n": len(baseline),
        "arms": {},
        "occupancy_recompute": {},
    }
    # ---- 1. record-level bitwise comparison
    for name, (path, aspect) in ARMS.items():
        arm = _read(path)
        common = sorted(set(arm) & set(baseline))
        identical_counts = {field: 0 for field in COMPARE_FIELDS}
        all_fields_identical = 0
        differing_scenes = []
        for pid in common:
            same_all = True
            for field in COMPARE_FIELDS:
                a, b = arm[pid].get(field), baseline[pid].get(field)
                same = (a == b) if not (isinstance(a, float) and isinstance(b, float)) else (a.hex() == b.hex())
                identical_counts[field] += int(same)
                same_all &= bool(same)
            if same_all:
                all_fields_identical += 1
            else:
                differing_scenes.append({
                    "patient_id": pid,
                    "E_arm": arm[pid].get("E_background_estimated"),
                    "E_baseline": baseline[pid].get("E_background_estimated"),
                    "nfev_arm": arm[pid].get("function_evaluations"),
                    "nfev_baseline": baseline[pid].get("function_evaluations"),
                })
        out["arms"][name] = {
            "records_path": str(path.relative_to(ROOT)),
            "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "region_aspect_xy": aspect,
            "n": len(arm),
            "n_paired_with_baseline": len(common),
            "E_background_estimated_bitwise_identical": identical_counts["E_background_estimated"],
            "bitwise_identical_by_field": identical_counts,
            "all_compared_fields_identical": all_fields_identical,
            "differing_scenes": differing_scenes,
            "median_E_error_percent_arm": round(100.0 * sorted(r["E_background_relative_error"] for r in arm.values())[len(arm) // 2], 4),
            "median_E_error_percent_baseline": round(100.0 * sorted(r["E_background_relative_error"] for r in baseline.values())[len(baseline) // 2], 4),
            "wall_time_median_arm_s": sorted(r["wall_time_seconds"] for r in arm.values())[len(arm) // 2],
            "wall_time_median_baseline_s": sorted(r["wall_time_seconds"] for r in baseline.values())[len(baseline) // 2],
        }
    # ---- 2. occupancy recompute on the 256-node patch
    per_scene = []
    for pid in sorted(baseline):
        scene, experiments = load_patient(CT_DATASET, rows[pid])
        prior = predictions[_normalize_id(pid)]
        center, radius = _predicted_region(scene, prior)
        nodes = scene["nodes"].to(torch.float64)
        sphere, source = material_occupancy(nodes, center=center, radius=radius)
        sphere_set = set(torch.where(sphere > 0.5)[0].tolist())
        delta = nodes - center
        dist = torch.linalg.vector_norm(delta, dim=1)
        entry = {
            "patient_id": pid,
            "n_nodes": int(nodes.shape[0]),
            "predicted_center": [float(v) for v in center],
            "predicted_radius": radius,
            "mesh_extent": [float(v) for v in (nodes.amax(dim=0) - nodes.amin(dim=0))],
            "sphere_source": source,
            "sphere_inclusion_nodes": len(sphere_set),
            "nearest_node_distance_over_radius": round(float(dist.min()) / radius, 4) if radius > 0 else None,
            "n_nodes_within_2r": int((dist < 2.0 * radius).sum()),
            "true_center": [float(v) for v in experiments[0]["inclusion_center"]],
            "true_radius": float(experiments[0]["inclusion_radius"]),
            "aspects": {},
        }
        # region-attention weights (region_track_weight = 8) as fit_patient builds them:
        # baseline uses the predicted centre/radius; the aspect arm (soft_occupancy path,
        # inclusion_center=None) falls back to the ground-truth centre/radius.
        surface = nodes[scene["surface_node_ids"]]
        true_center = experiments[0]["inclusion_center"].to(torch.float64)
        true_radius = max(float(experiments[0]["inclusion_radius"]), 1e-6)

        def _attention(c: torch.Tensor, r: float) -> torch.Tensor:
            d = torch.linalg.vector_norm(surface - c, dim=1)
            return 1.0 + 8.0 * torch.exp(-0.5 * (d / (2.0 * r)) ** 2)

        att_pred = _attention(center, max(radius, 1e-6))
        att_true = _attention(true_center, true_radius)
        entry["attention_weights_predicted_vs_true_region"] = {
            "max_abs_difference": float((att_pred - att_true).abs().max()),
            "identical": bool(torch.equal(att_pred, att_true)),
            "predicted_minus_true_center_norm_over_true_radius": float(torch.linalg.vector_norm(center - true_center)) / true_radius,
        }
        for name, (_, aspect) in ARMS.items():
            ell = _ellipsoid_occupancy(nodes, center, radius, aspect)
            ell_set = set(torch.where(ell > 0.5)[0].tolist())
            arm_rows = _read(ARMS[name][0])
            entry["aspects"][name] = {
                "ellipsoid_inclusion_nodes": len(ell_set),
                "added_vs_sphere": sorted(ell_set - sphere_set),
                "removed_vs_sphere": sorted(sphere_set - ell_set),
                "node_set_identical": ell_set == sphere_set,
                "E_field_bitwise_identical": bool(torch.equal(ell, sphere)),
                "E_estimate_bitwise_identical_to_baseline": (
                    arm_rows[pid]["E_background_estimated"].hex() == baseline[pid]["E_background_estimated"].hex()
                ),
                "cost_identical_to_baseline": arm_rows[pid]["cost"] == baseline[pid]["cost"],
                "cost_arm_over_baseline": arm_rows[pid]["cost"] / baseline[pid]["cost"],
                "refinement_accepted_baseline": bool(baseline[pid]["refinement_accepted"]),
            }
        per_scene.append(entry)
    out["occupancy_recompute"] = {
        "n_scenes": len(per_scene),
        "scenes_with_zero_sphere_inclusion_nodes": sum(e["sphere_inclusion_nodes"] == 0 for e in per_scene),
        "sphere_inclusion_node_counts": [e["sphere_inclusion_nodes"] for e in per_scene],
        "nearest_node_distance_over_radius": [e["nearest_node_distance_over_radius"] for e in per_scene],
        "attention_weights_differ_from_baseline_in_n_scenes": sum(
            not e["attention_weights_predicted_vs_true_region"]["identical"] for e in per_scene
        ),
        "summary_by_aspect": {
            name: {
                "scenes_node_set_identical_to_sphere": sum(e["aspects"][name]["node_set_identical"] for e in per_scene),
                "scenes_node_set_changed": sum(not e["aspects"][name]["node_set_identical"] for e in per_scene),
                "ellipsoid_inclusion_node_counts": [e["aspects"][name]["ellipsoid_inclusion_nodes"] for e in per_scene],
                "node_set_changed_but_E_estimate_identical": sum(
                    (not e["aspects"][name]["node_set_identical"]) and e["aspects"][name]["E_estimate_bitwise_identical_to_baseline"]
                    for e in per_scene
                ),
                "node_set_changed_and_E_estimate_differs": sum(
                    (not e["aspects"][name]["node_set_identical"]) and not e["aspects"][name]["E_estimate_bitwise_identical_to_baseline"]
                    for e in per_scene
                ),
                "node_set_identical_but_cost_differs": sum(
                    e["aspects"][name]["node_set_identical"] and not e["aspects"][name]["cost_identical_to_baseline"]
                    for e in per_scene
                ),
                "cost_identical_to_baseline_n": sum(e["aspects"][name]["cost_identical_to_baseline"] for e in per_scene),
                "accepted_scenes_with_E_identical": sum(
                    e["aspects"][name]["refinement_accepted_baseline"] and e["aspects"][name]["E_estimate_bitwise_identical_to_baseline"]
                    for e in per_scene
                ),
                "accepted_scenes_n": sum(e["aspects"][name]["refinement_accepted_baseline"] for e in per_scene),
            }
            for name in ARMS
        },
        "per_scene": per_scene,
    }
    # fit_patient side effect: with soft_occupancy, inclusion_center/radius are None -> fall back to TRUE centre/radius
    out["fit_patient_side_effect_note"] = (
        "run_screened_fem_locked passes inclusion_center=None/inclusion_radius=None when region_aspect_xy is set; "
        "fit_patient then falls back to first['inclusion_center'] / first['inclusion_radius'] (ground-truth region) for the "
        "region-attention weights (region_track_weight=8), while the E-field occupancy comes from soft_occupancy. "
        "So the aspect arm differs from the baseline in attention centre/radius even when the occupancy node set is unchanged."
    )
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    compact = {k: {kk: vv for kk, vv in v.items() if kk != "differing_scenes"} for k, v in out["arms"].items()}
    print(json.dumps(compact, indent=1))
    print(json.dumps({k: v for k, v in out["occupancy_recompute"].items() if k != "per_scene"}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
