"""Generate synthetic mechanics on approved de-identified ION CT geometries."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.ct_geometry import build_scene_from_ct_mesh  # noqa: E402
from lung_inverse_rendering.generate_sim_lung_v2 import (  # noqa: E402
    EXPERIMENTS,
    LOAD_ENVELOPE,
    MULTIVIEW_SCHEMA_VERSION,
    NUM_MULTIVIEW_CAMERAS,
    build_patient,
    generate_experiment,
    patient_spec,
)


DEFAULT_MESH_DIR = ROOT / "results" / "ion_geometry_ood" / "deidentified_meshes"
DEFAULT_OUT = ROOT / "dataset" / "ion_ct_synthetic_mechanics"
SCHEMA_VERSION = "sim_lung_v2_ion_ct_synthetic_mechanics_v1"


def discover_geometry_meshes(mesh_dir: Path, geometry_limit: int | None = None) -> list[Path]:
    """Return only opaque de-identified geometry exports in deterministic order."""
    if geometry_limit is not None and geometry_limit <= 0:
        raise ValueError("geometry_limit must be positive")
    meshes = sorted(mesh_dir.glob("geom_*.npz"))
    if geometry_limit is not None:
        meshes = meshes[:geometry_limit]
    if not meshes:
        raise FileNotFoundError(f"No de-identified geom_*.npz meshes found in {mesh_dir}")
    return meshes


def scenario_spec(
    scenario_index: int,
    scenario_count: int,
    *,
    geometry_id: str,
) -> dict[str, Any]:
    """Create a synthetic material/mechanics scenario with an opaque public ID."""
    if scenario_index < 0 or scenario_index >= scenario_count:
        raise ValueError("scenario_index is outside the declared scenario range")
    spec = patient_spec(
        scenario_index,
        max(scenario_count, 5),
        randomize_materials=True,
    )
    spec.update(
        {
            "scenario_id": f"scenario_{scenario_index:03d}",
            "scenario_template_index": scenario_index,
            "patient_id": f"{geometry_id}_scenario_{scenario_index:03d}",
            "split": "test",
            "geometry_id": geometry_id,
            "geometry_source": "deidentified_ct_mesh",
            "material_source": "synthetic",
            "mechanics_source": "synthetic",
        }
    )
    return spec


def generation_config(
    *,
    geometry_ids: list[str],
    scenarios_per_geometry: int,
    resolution: int,
    motion_noise_std: float,
    save_images: bool,
    force_prior_fraction: float,
) -> dict[str, Any]:
    return {
        "geometry_count": len(geometry_ids),
        "geometry_ids": geometry_ids,
        "scenarios_per_geometry": scenarios_per_geometry,
        "resolution": resolution,
        "motion_noise_std": motion_noise_std,
        "images_saved": save_images,
        "multiview": True,
        "num_views": NUM_MULTIVIEW_CAMERAS,
        "load_count": len(EXPERIMENTS),
        "frame_count": len(LOAD_ENVELOPE),
        "force_prior_fraction": force_prior_fraction,
        "base_protocol": MULTIVIEW_SCHEMA_VERSION,
    }


def manifest_payload(
    rows: dict[str, dict[str, Any]],
    *,
    total_scenarios: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    patients = [rows[key] for key in sorted(rows)]
    return {
        "version": SCHEMA_VERSION,
        "patient_count": total_scenarios,
        "generated_patient_count": len(patients),
        "experiment_count": sum(len(row["experiments"]) for row in patients),
        "patient_level_split": True,
        "generation_config": config,
        "geometry_split": "test",
        "material_source": "synthetic",
        "mechanics_source": "synthetic",
        "known_inputs": [
            "de-identified CT-conditioned geometry",
            "synthetic per-frame nodal force and contact location",
            "camera pose",
            "measured force with calibrated uncertainty",
            "boundary-condition DOFs",
        ],
        "observations": [
            "rendered image sequence" if config["images_saved"] else "images omitted",
            "noisy surface motion sequence",
            "image-plane surface-node tracks and visibility",
            "synchronized three-view depth and foreground confidence",
        ],
        "privacy": {
            "scenario_ids_contain_source_identifiers": False,
            "raw_paths_persisted": False,
            "geometry_is_patient_derived": True,
            "material_and_mechanics_are_synthetic": True,
        },
        "patients": patients,
    }


def write_manifest(
    manifest_path: Path,
    rows: dict[str, dict[str, Any]],
    *,
    total_scenarios: int,
    config: dict[str, Any],
) -> None:
    payload = manifest_payload(rows, total_scenarios=total_scenarios, config=config)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def generate_dataset(args: argparse.Namespace) -> None:
    if args.scenarios_per_geometry <= 0:
        raise ValueError("scenarios_per_geometry must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    mesh_paths = discover_geometry_meshes(args.mesh_dir, args.geometry_limit)
    geometry_ids = [path.stem for path in mesh_paths]
    total_scenarios = len(mesh_paths) * args.scenarios_per_geometry
    scenario_end = total_scenarios if args.scenario_end is None else args.scenario_end
    if not 0 <= args.scenario_start < scenario_end <= total_scenarios:
        raise ValueError(
            "Require 0 <= scenario-start < scenario-end <= total scenario count"
        )

    config = generation_config(
        geometry_ids=geometry_ids,
        scenarios_per_geometry=args.scenarios_per_geometry,
        resolution=args.resolution,
        motion_noise_std=args.motion_noise_std,
        save_images=not args.no_images,
        force_prior_fraction=args.force_prior_fraction,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    rows: dict[str, dict[str, Any]] = {}
    if manifest_path.exists() and not args.overwrite:
        if not args.resume:
            raise FileExistsError(
                f"{manifest_path} exists; pass --resume or --overwrite"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("version") != SCHEMA_VERSION:
            raise ValueError("Existing manifest schema does not match this generator")
        if existing.get("generation_config") != config:
            raise ValueError("Existing manifest generation_config does not match CLI")
        rows = {row["patient_id"]: row for row in existing.get("patients", [])}
    write_manifest(
        manifest_path,
        rows,
        total_scenarios=total_scenarios,
        config=config,
    )

    for geometry_index, mesh_path in enumerate(mesh_paths):
        first = geometry_index * args.scenarios_per_geometry
        last = first + args.scenarios_per_geometry
        selected = range(max(first, args.scenario_start), min(last, scenario_end))
        if selected.start >= selected.stop:
            continue
        geometry_id = geometry_ids[geometry_index]
        scene = build_scene_from_ct_mesh(
            mesh_path,
            geometry_id=geometry_id,
            E_true=5_000.0,
        )
        for global_scenario_index in selected:
            scenario_index = global_scenario_index - first
            spec = scenario_spec(
                scenario_index,
                args.scenarios_per_geometry,
                geometry_id=geometry_id,
            )
            scenario_id = spec["patient_id"]
            existing_row = rows.get(scenario_id)
            if args.resume and existing_row is not None:
                complete = all(
                    (args.out / experiment["relative_path"]).exists()
                    for experiment in existing_row["experiments"]
                )
                if complete:
                    print(f"{scenario_id} already complete; skipping", flush=True)
                    continue
            scenario_dir = args.out / scenario_id
            if args.overwrite and scenario_dir.exists():
                shutil.rmtree(scenario_dir)
            elif existing_row is None and scenario_dir.exists():
                raise FileExistsError(
                    f"{scenario_dir} is not represented in the manifest; "
                    "pass --overwrite to replace it"
                )

            patient_scene, E_nodes, inclusion_center, inclusion_radius = build_patient(
                spec, ct_mesh_scene=scene
            )
            experiments = [
                generate_experiment(
                    args.out,
                    spec,
                    patient_scene,
                    E_nodes,
                    inclusion_center,
                    inclusion_radius,
                    experiment,
                    experiment_index=index,
                    resolution=args.resolution,
                    motion_noise_std=args.motion_noise_std,
                    schema_version=SCHEMA_VERSION,
                    save_images=not args.no_images,
                    multiview=True,
                    force_prior_fraction=args.force_prior_fraction,
                )
                for index, experiment in enumerate(EXPERIMENTS)
            ]
            rows[scenario_id] = {**spec, "experiments": experiments}
            write_manifest(
                manifest_path,
                rows,
                total_scenarios=total_scenarios,
                config=config,
            )
            print(
                f"{scenario_id} geometry={geometry_index:02d} split=test "
                f"experiments={len(experiments)}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--geometry-limit", type=int)
    parser.add_argument("--scenarios-per-geometry", type=int, default=20)
    parser.add_argument("--scenario-start", type=int, default=0)
    parser.add_argument("--scenario-end", type=int)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--motion-noise-std", type=float, default=2.5e-4)
    parser.add_argument("--force-prior-fraction", type=float, default=0.05)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    torch.set_default_dtype(torch.float64)
    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
