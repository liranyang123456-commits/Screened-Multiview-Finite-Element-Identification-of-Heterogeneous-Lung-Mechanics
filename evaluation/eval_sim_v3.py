"""Frozen-protocol sim_v3 joint inversion and force-error sensitivity."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import material_metrics, psnr, ssim  # noqa: E402
from inverse.heterogeneous_recover import (  # noqa: E402
    heterogeneous_forward,
    recover_inclusion,
)
from inverse.joint_opt import joint_recover  # noqa: E402
from physics.fem import solve_nh  # noqa: E402
from rendering.gaussian_pbr import render, seed_surface_gaussians, set_albedo  # noqa: E402


DATASET = ROOT / "dataset" / "sim_v3"
RESULTS = ROOT / "results" / "sim_v3"
CONFIG_PATH = RESULTS / "frozen_config.json"
DEFAULT_CONFIG = {
    "resolution": 48,
    "frame_stride": 4,
    "uniform_iters": 15,
    "heterogeneous_iters": 5,
    "E_init_multiplier": 2.0,
    "inclusion_init_ratio": 2.0,
    "warmup_albedo": 2,
    "schedule": True,
    "seed": 2026,
}


def reconstruct_scene(spec: dict, E: float) -> dict:
    def d64(value):
        return value.to(torch.float64) if torch.is_tensor(value) else value

    nodes = d64(spec["nodes"])
    return {
        "nodes": nodes,
        "elems": spec["elems"],
        "Nn": spec["Nn"],
        "D": spec["D"],
        "fixed": spec["fixed"],
        "surface_tris": spec["surface_tris"],
        "E_true": torch.tensor(E, dtype=torch.float64),
        "nu_true": torch.tensor(0.45, dtype=torch.float64),
        "lx": float(nodes[:, 0].max()),
        "ly": float(nodes[:, 1].max()),
        "lz": float(nodes[:, 2].max() - nodes[:, 2].min()),
    }


def load_images(scene_dir: Path, gt: dict, config: dict):
    indices = list(range(0, gt["T"], config["frame_stride"]))
    observed = []
    for index in indices:
        image = np.asarray(
            Image.open(scene_dir / "images" / f"frame_{index:02d}.png").convert("RGB"),
            dtype=np.float64,
        )
        observed.append(torch.from_numpy(image).permute(2, 0, 1) / 255.0)
    observed_tensor = F.interpolate(
        torch.stack(observed),
        size=(config["resolution"], config["resolution"]),
        mode="bilinear",
        align_corners=False,
    )
    clean = gt["clean_images_uint8"][indices].to(torch.float64) / 255.0
    clean = F.interpolate(
        clean,
        size=(config["resolution"], config["resolution"]),
        mode="bilinear",
        align_corners=False,
    )
    return indices, observed_tensor, clean


def force_rmse_from_real_data() -> float:
    path = ROOT / "results" / "small_bowel_force" / "summary.json"
    if not path.exists():
        return 0.323
    rows = json.loads(path.read_text(encoding="utf-8"))
    values = [row["rmse_n"]["mean"] for row in rows]
    return float(min(values))


def perturb_forces(
    forces: list[torch.Tensor],
    scenario: str,
    force_rmse_n: float,
    seed: int,
) -> list[torch.Tensor]:
    if scenario == "known":
        return forces
    target_rmse = 0.10 if scenario == "noisy" else force_rmse_n
    totals = torch.tensor(
        [float(force.view(-1, 3)[:, 2].abs().sum()) for force in forces]
    )
    nonzero = totals[totals > 0]
    denominator = float(nonzero.mean()) if len(nonzero) else 1.0
    relative_sigma = target_rmse / max(denominator, 1e-6)
    generator = torch.Generator().manual_seed(seed)
    return [
        force
        * (
            1.0
            + float(torch.randn((), generator=generator))
            * relative_sigma
        )
        for force in forces
    ]


def render_uniform(
    scene: dict,
    gaussians: dict,
    forces: list[torch.Tensor],
    poses: torch.Tensor,
    E: float,
    albedo: list[float],
    roughness: float,
    resolution: int,
) -> torch.Tensor:
    set_albedo(gaussians, albedo)
    gaussians["roughness"] = torch.full_like(
        gaussians["roughness"], roughness
    )
    images = []
    with torch.no_grad():
        for force, pose in zip(forces, poses):
            displacement = solve_nh(
                scene["nodes"],
                scene["elems"],
                torch.tensor(E, dtype=torch.float64),
                scene["nu_true"],
                force,
                scene["fixed"],
                D=scene["D"],
            )
            images.append(
                render(
                    gaussians,
                    scene,
                    displacement,
                    pose,
                    H=resolution,
                    W=resolution,
                    light_intensity=2.0,
                ).clamp(0, 1)
            )
    return torch.stack(images)


def image_metrics(predicted: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": float(np.mean([psnr(p, g) for p, g in zip(predicted, clean)])),
        "ssim": float(np.mean([ssim(p, g) for p, g in zip(predicted, clean)])),
    }


def evaluate_case(
    scene_row: dict,
    scenario: str,
    config: dict,
    force_rmse_n: float,
) -> dict:
    scene_dir = DATASET / f"scene_{scene_row['id']:04d}"
    gt = torch.load(scene_dir / "gt.pt", weights_only=False)
    scene = reconstruct_scene(gt["scene_spec"], gt["E_bg"])
    indices, observed, clean = load_images(scene_dir, gt, config)
    forces = [gt["forces"][index].to(torch.float64) for index in indices]
    forces = perturb_forces(
        forces,
        scenario,
        force_rmse_n=force_rmse_n,
        seed=config["seed"] + scene_row["id"] * 10 + {"known": 0, "noisy": 1, "estimated": 2}[scenario],
    )
    poses = gt["poses"][indices].to(torch.float64)
    torch.manual_seed(config["seed"])
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    uniform = joint_recover(
        scene,
        gaussians,
        forces,
        poses,
        observed,
        E_init=gt["E_bg"] * config["E_init_multiplier"],
        albedo_init=(0.5, 0.5, 0.5),
        rough_init=0.7,
        iters=config["uniform_iters"],
        lr_E=gt["E_bg"] * 0.08,
        lr_opt=5e-2,
        H=config["resolution"],
        W=config["resolution"],
        light=2.0,
        verbose=False,
        warmup_albedo=config["warmup_albedo"],
        schedule=config["schedule"],
    )
    optical = material_metrics(
        {
            "E": uniform["E_recovered"],
            "albedo": uniform["albedo_recovered"],
            "roughness": uniform["rough_recovered"],
        },
        {"E": gt["E_bg"], "albedo": gt["albedo"], "roughness": gt["roughness"]},
    )

    if gt["stiffness_mode"] == "homogeneous":
        predicted = render_uniform(
            scene,
            gaussians,
            forces,
            poses,
            uniform["E_recovered"],
            uniform["albedo_recovered"],
            uniform["rough_recovered"],
            config["resolution"],
        )
        mechanics = {
            "E_rel": optical["E_rel"],
            "E_bg_rel": optical["E_rel"],
            "E_inc_rel": None,
            "ratio_rel": None,
            "center_error": None,
            "radius_rel": None,
        }
    else:
        set_albedo(gaussians, uniform["albedo_recovered"])
        gaussians["roughness"] = torch.full_like(
            gaussians["roughness"], uniform["rough_recovered"]
        )
        center = gt["inclusion_center"]
        recovered = recover_inclusion(
            scene,
            gaussians,
            forces,
            poses,
            observed,
            E_bg_init=uniform["E_recovered"],
            E_inc_init=uniform["E_recovered"] * config["inclusion_init_ratio"],
            inc_xy_init=(float(scene["nodes"][:, 0].mean()), float(scene["nodes"][:, 1].mean())),
            inc_r_init=0.20 * min(scene["lx"], scene["ly"]),
            iters=config["heterogeneous_iters"],
            lr=0.05,
            H=config["resolution"],
            W=config["resolution"],
            light=2.0,
        )
        predicted = heterogeneous_forward(
            scene,
            gaussians,
            forces,
            poses,
            recovered["E_bg"],
            recovered["E_inc"],
            recovered["inc_center"],
            recovered["inc_radius"],
            H=config["resolution"],
            W=config["resolution"],
            light=2.0,
        )
        gt_ratio = gt["E_inc"] / gt["E_bg"]
        recovered_ratio = recovered["E_inc"] / recovered["E_bg"]
        center_error = math.dist(
            recovered["inc_center"],
            [float(center[0]), float(center[1])],
        )
        mechanics = {
            "E_rel": abs(recovered["E_bg"] - gt["E_bg"]) / gt["E_bg"],
            "E_bg_rel": abs(recovered["E_bg"] - gt["E_bg"]) / gt["E_bg"],
            "E_inc_rel": abs(recovered["E_inc"] - gt["E_inc"]) / gt["E_inc"],
            "ratio_rel": abs(recovered_ratio - gt_ratio) / gt_ratio,
            "center_error": center_error,
            "radius_rel": abs(recovered["inc_radius"] - gt["inclusion_radius"])
            / gt["inclusion_radius"],
        }
    return {
        "scene_id": scene_row["id"],
        "scenario": scenario,
        "split": scene_row["split"],
        "geometry": gt["geometry"],
        "geometry_family": "c3vd" if gt["geometry"].startswith("c3vd_") else "procedural",
        "stiffness_mode": gt["stiffness_mode"],
        "noise_sigma": gt["noise_sigma"],
        "pose_jitter": gt["pose_jitter"],
        "optical_coupling": gt["optical_coupling"],
        "force_scale": gt["force_scale"],
        **mechanics,
        "albedo_rel": optical["albedo_rel"],
        "roughness_rel": optical["rough_rel"],
        **image_metrics(predicted, clean),
    }


def bootstrap_ci(values: list[float], seed: int = 2026) -> list[float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(10_000, len(array)), replace=True)
    return np.percentile(np.median(samples, axis=1), [2.5, 97.5]).tolist()


def paired_sign_flip(a: np.ndarray, b: np.ndarray, seed: int = 2026) -> float:
    differences = np.asarray(a) - np.asarray(b)
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(100_000, len(differences)))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def summarize(rows: list[dict], force_rmse_n: float) -> dict:
    metrics = [
        "E_rel",
        "E_bg_rel",
        "E_inc_rel",
        "ratio_rel",
        "center_error",
        "radius_rel",
        "albedo_rel",
        "roughness_rel",
        "psnr",
        "ssim",
    ]
    groups = {}
    for scenario in ("known", "noisy", "estimated"):
        subset = [row for row in rows if row["scenario"] == scenario]
        groups[scenario] = {}
        for metric in metrics:
            values = [
                float(row[metric])
                for row in subset
                if row[metric] is not None and np.isfinite(row[metric])
            ]
            if values:
                groups[scenario][metric] = {
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "bootstrap_95_ci_of_median": bootstrap_ci(values),
                }
    known = sorted(
        [row for row in rows if row["scenario"] == "known"],
        key=lambda row: row["scene_id"],
    )
    estimated = sorted(
        [row for row in rows if row["scenario"] == "estimated"],
        key=lambda row: row["scene_id"],
    )
    tests = {
        metric: paired_sign_flip(
            np.asarray([row[metric] for row in known], dtype=float),
            np.asarray([row[metric] for row in estimated], dtype=float),
        )
        for metric in ("E_rel", "albedo_rel", "roughness_rel", "psnr", "ssim")
    }
    strata = {}
    for field in ("geometry_family", "stiffness_mode", "noise_sigma", "pose_jitter", "optical_coupling"):
        strata[field] = {}
        for value in sorted({str(row[field]) for row in known}):
            subset = [row for row in known if str(row[field]) == value]
            strata[field][value] = {
                metric: float(np.median([row[metric] for row in subset if row[metric] is not None]))
                for metric in ("E_rel", "albedo_rel", "roughness_rel", "psnr", "ssim")
            }
    return {
        "force_model_rmse_n": force_rmse_n,
        "frozen_config": json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        "groups": groups,
        "paired_sign_flip_known_vs_estimated_p": tests,
        "known_force_strata_medians": strata,
        "rows": rows,
    }


def write_outputs(summary: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# sim_v3 frozen-protocol evaluation",
        "",
        f"Real-video force RMSE used for injection: {summary['force_model_rmse_n']:.3f} N.",
        "",
        "| Force input | E error | Albedo error | Roughness error | PSNR | SSIM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, group in summary["groups"].items():
        lines.append(
            f"| {scenario} | {group['E_rel']['median']*100:.1f}% | "
            f"{group['albedo_rel']['median']*100:.1f}% | "
            f"{group['roughness_rel']['median']*100:.1f}% | "
            f"{group['psnr']['median']:.2f} | {group['ssim']['median']:.3f} |"
        )
    (RESULTS / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "paper_tbme" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    scenarios = ("known", "noisy", "estimated")
    axes[0].boxplot(
        [
            [
                row["E_rel"] * 100
                for row in summary["rows"]
                if row["scenario"] == scenario
            ]
            for scenario in scenarios
        ],
        labels=scenarios,
        showfliers=False,
    )
    axes[0].set_ylabel("Young's modulus error (%)")
    axes[0].set_title("Force-input sensitivity")
    known = [row for row in summary["rows"] if row["scenario"] == "known"]
    families = ("procedural", "c3vd")
    axes[1].boxplot(
        [
            [row["E_rel"] * 100 for row in known if row["geometry_family"] == family]
            for family in families
        ],
        labels=families,
        showfliers=False,
    )
    axes[1].set_ylabel("Young's modulus error (%)")
    axes[1].set_title("Geometry transfer")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_sim_v3.png", dpi=300)
    plt.close(fig)


def freeze_default_config() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        **DEFAULT_CONFIG,
        "selection_basis": (
            "Fixed before full test evaluation from sim_v2 convergence behavior, "
            "sim_v3 QC, and validation-split runtime/identifiability constraints. "
            "The single test smoke run was used only for interface validation and "
            "was excluded from hyperparameter selection and final statistics."
        ),
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-config", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scenario", choices=["known", "noisy", "estimated", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.freeze_config or not CONFIG_PATH.exists():
        freeze_default_config()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    scenes = [row for row in manifest["scenes"] if row["split"] == args.split]
    if args.limit is not None:
        scenes = scenes[: args.limit]
    scenarios = ("known", "noisy", "estimated") if args.scenario == "all" else (args.scenario,)
    RESULTS.mkdir(parents=True, exist_ok=True)
    checkpoint = RESULTS / f"{args.split}_rows.json"
    rows = []
    if checkpoint.exists() and not args.overwrite:
        rows = json.loads(checkpoint.read_text(encoding="utf-8"))
    done = {(row["scene_id"], row["scenario"]) for row in rows}
    force_rmse = force_rmse_from_real_data()
    for scene in scenes:
        for scenario in scenarios:
            key = (scene["id"], scenario)
            if key in done:
                continue
            row = evaluate_case(scene, scenario, config, force_rmse)
            rows.append(row)
            checkpoint.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(
                f"scene {scene['id']:04d} {scenario}: "
                f"E={row['E_rel']*100:.1f}% PSNR={row['psnr']:.2f}",
                flush=True,
            )
    if args.split == "test" and args.scenario == "all" and args.limit is None:
        write_outputs(summarize(rows, force_rmse))


if __name__ == "__main__":
    main()
