"""Generate the expanded CT mechanics cohort in isolated parallel shards."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "experiments" / "generate_ion_ct_synthetic_mechanics.py"
DEFAULT_MESH_DIR = ROOT / "results" / "ion_geometry_ood" / "deidentified_meshes27"
DEFAULT_SPLIT = ROOT / "results" / "ion_geometry_ood" / "geometry_split_27.json"
DEFAULT_OUTPUT = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"


def geometry_ranges(geometry_count: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, geometry_count))
    base, remainder = divmod(geometry_count, workers)
    ranges = []
    start = 0
    for index in range(workers):
        width = base + int(index < remainder)
        ranges.append((start, start + width))
        start += width
    return ranges


def _run_shard(
    *,
    index: int,
    geometry_range: tuple[int, int],
    shard_root: Path,
    mesh_dir: Path,
    split_manifest: Path,
    scenarios_per_geometry: int,
    resolution: int,
    motion_noise_std: float,
    force_prior_fraction: float,
) -> Path:
    first_geometry, last_geometry = geometry_range
    scenario_start = first_geometry * scenarios_per_geometry
    scenario_end = last_geometry * scenarios_per_geometry
    command = [
        sys.executable,
        str(GENERATOR),
        "--mesh-dir",
        str(mesh_dir),
        "--geometry-split-manifest",
        str(split_manifest),
        "--out",
        str(shard_root),
        "--scenarios-per-geometry",
        str(scenarios_per_geometry),
        "--scenario-start",
        str(scenario_start),
        "--scenario-end",
        str(scenario_end),
        "--resolution",
        str(resolution),
        "--motion-noise-std",
        str(motion_noise_std),
        "--force-prior-fraction",
        str(force_prior_fraction),
        "--no-images",
        "--overwrite",
    ]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = shard_root.parent / f"shard_{index:02d}.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"Shard {index} failed with exit code {completed.returncode}; "
            f"see {log_path.name}"
        )
    return shard_root / "manifest.json"


def _merge_shards(
    manifests: list[Path],
    output: Path,
    *,
    expected_scenarios: int,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    template = None
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if template is None:
            template = payload
        elif payload["generation_config"] != template["generation_config"]:
            raise ValueError("Shard generation configurations differ")
        for row in payload["patients"]:
            patient_id = str(row["patient_id"])
            if patient_id in rows:
                raise ValueError(f"Duplicate generated scenario: {patient_id}")
            source = manifest_path.parent / patient_id
            destination = output / patient_id
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(source), str(destination))
            rows[patient_id] = row
    if template is None or len(rows) != expected_scenarios:
        raise ValueError(
            f"Expected {expected_scenarios} scenarios, found {len(rows)}"
        )
    merged = {
        **template,
        "patient_count": expected_scenarios,
        "generated_patient_count": len(rows),
        "experiment_count": sum(len(row["experiments"]) for row in rows.values()),
        "patients": [rows[key] for key in sorted(rows)],
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    temporary.replace(output / "manifest.json")
    return merged


def generate_parallel(args: argparse.Namespace) -> dict[str, Any]:
    split = json.loads(args.geometry_split_manifest.read_text(encoding="utf-8"))
    geometry_count = int(split["geometry_count"])
    expected_scenarios = geometry_count * args.scenarios_per_geometry
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    shard_parent = args.output.parent / f".{args.output.name}_shards"
    if shard_parent.exists():
        shutil.rmtree(shard_parent)
    shard_parent.mkdir(parents=True)
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, geometry_range in enumerate(
            geometry_ranges(geometry_count, args.workers)
        ):
            shard_root = shard_parent / f"shard_{index:02d}"
            jobs.append(
                executor.submit(
                    _run_shard,
                    index=index,
                    geometry_range=geometry_range,
                    shard_root=shard_root,
                    mesh_dir=args.mesh_dir,
                    split_manifest=args.geometry_split_manifest,
                    scenarios_per_geometry=args.scenarios_per_geometry,
                    resolution=args.resolution,
                    motion_noise_std=args.motion_noise_std,
                    force_prior_fraction=args.force_prior_fraction,
                )
            )
        manifests = [future.result() for future in as_completed(jobs)]
    result = _merge_shards(
        manifests,
        args.output,
        expected_scenarios=expected_scenarios,
    )
    shutil.rmtree(shard_parent)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument(
        "--geometry-split-manifest",
        type=Path,
        default=DEFAULT_SPLIT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenarios-per-geometry", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--motion-noise-std", type=float, default=2.5e-4)
    parser.add_argument("--force-prior-fraction", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = generate_parallel(args)
    print(
        json.dumps(
            {
                "geometry_count": result["generation_config"]["geometry_count"],
                "scenario_count": result["generated_patient_count"],
                "experiment_count": result["experiment_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
