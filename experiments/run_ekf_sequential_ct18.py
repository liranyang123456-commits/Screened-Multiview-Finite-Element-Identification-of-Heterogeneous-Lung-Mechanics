"""Sequential EKF baseline over the five nonzero load frames.

Default (unchanged): the 18 locked CT scenes (paired_scene_ids.json) on the
ct120 dataset with ratio and region locked exactly as in the official-prior
screened FEM.  State x = (log E_bg, log q).

Generalisations (2026-10-12, reviewer compute):
* ``--cohort ct120|synthetic50``.  synthetic50 uses the calibrator-strict
  initializer and the SAME ratio-lock decision / initial ratio / prior centre
  as the six-arm learned/learned matrix (``_low_contrast`` with
  contrast_ratio_lock = 1.20, prior_width_scale = 2.0).  When the ratio is not
  locked, log-ratio is a third EKF state with prior SD = matched
  material_prior_sigma[1] = max(0.05, 2 * log_ratio_std).
* ``--all-test`` runs every test-split scene (120 CT / 50 synthetic).
* ``--patient-start/--patient-limit`` and ``--shard i/n`` (round-robin) for
  parallel shards; each shard appends to its own ``records_shard{i}of{n}.jsonl``
  and resume reads every ``records*.jsonl`` in the output directory.
* ``--summarize`` merges shard files into ``records.jsonl`` (dedup by
  patient_id, test-split order) and writes ``summary.json`` including a
  PAIRED comparison against the matched batch arm.

Measurement h(x) = multiview UV projections of the isotropic FEM solve at
frame t under measured forces.  Jacobian by central differences in
log-parameters.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_screened_fem_locked import (  # noqa: E402
    CT_DATASET,
    CT_HYBRID,
    SYNTHETIC_DATASET,
    SYNTHETIC_INIT,
    _index_records,
    _low_contrast,
    _normalize_id,
    _official_2028_ratio_index,
    _official_ratio_key,
    _predicted_region,
)
from lung_inverse_rendering.evaluate_sim_lung_v2 import (  # noqa: E402
    force_scale_target,
    load_patient,
    measured_forces,
    relative_error,
)
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous  # noqa: E402
from rendering.gaussian_pbr import project  # noqa: E402

IDS = ROOT / "results" / "ct_reviewer_drivers_v6" / "paired_scene_ids.json"
OUT = ROOT / "results" / "ct_reviewer_drivers_v6" / "ekf_sequential"
FRAMES = (1, 2, 3, 4, 5)
TRACK_NOISE_PX = 0.05
PRIOR_WIDTH_SCALE = 2.0       # six-arm learned/learned and official-prior CT arms
CONTRAST_RATIO_LOCK = 1.20    # six-arm learned/learned contrast lock threshold
FORCE_PRIOR_SIGMA = 0.05
GLOBAL_BOUNDS = {"E": (1000.0, 15000.0), "ratio": (1.0, 5.0)}  # fit_patient global box

# Matched batch arms used for the paired comparison in --summarize.
BATCH_REFERENCES = {
    "ct120": {
        "official_prior_screened_fem120_v1": ROOT / "results" / "ion_ct_official_prior_screened_fem120_v1" / "records.jsonl",
    },
    "synthetic50": {
        "six_arm_learned_learned": ROOT / "results" / "lung_mesh_ai_v2" / "synthetic50_full_paired_controls_20261008" / "learned_learned" / "records.jsonl",
        "frozen_primary_screened_map_locked50_v1": ROOT / "results" / "lung_mesh_ai_v2" / "screened_fem_locked50_v1" / "records.jsonl",
    },
}
BATCH_18_REFERENCE = ROOT / "results" / "ct_reviewer_drivers_v6" / "ct_reviewer_drivers_v6" / "fem_paired" / "w2_s0p05_p2" / "records.jsonl"


def _predict_uv(scene, experiment, center, radius, log_E, log_q, ratio, frame, focal, H, W):
    E_nodes = make_heterogeneous_E_field(
        scene["nodes"], center, radius, math.exp(log_E), math.exp(log_E) * ratio
    )
    force = measured_forces(experiment)[frame].to(torch.float64) * math.exp(log_q)
    u = solve_nh_heterogeneous(scene["nodes"], scene["elems"], E_nodes, scene["nu_true"], force, scene["fixed"])
    points = (scene["nodes"] + u.view(-1, 3))[scene["surface_node_ids"]]
    rows = []
    for view in range(experiment["poses_multiview"].shape[1]):
        uv, _, _ = project(points, experiment["poses_multiview"][frame, view], focal=focal, H=H, W=W)
        rows.append(uv.to(torch.float64))
    return torch.stack(rows)  # (V, N, 2)


def _patient_seed(pid: str) -> int:
    """Same seeding rule as the batch fitter (int suffix, else stable hash)."""
    token = str(pid).rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return int(hashlib.sha256(str(pid).encode("utf-8")).hexdigest()[:8], 16) % 1_000_000


def _cohort_setup(cohort: str) -> dict:
    if cohort == "synthetic50":
        return {
            "dataset": SYNTHETIC_DATASET,
            "predictions": _index_records(SYNTHETIC_INIT),
            "official_ratios": {},
            "always_lock_ratio": False,
            "initializer_path": SYNTHETIC_INIT,
        }
    return {
        "dataset": CT_DATASET,
        "predictions": _index_records(CT_HYBRID, method="ours_hybrid_initializer"),
        "official_ratios": _official_2028_ratio_index(),
        "always_lock_ratio": True,
        "initializer_path": CT_HYBRID,
    }


def _select_ids(args, manifest: dict) -> list[str]:
    test_ids = [str(r["patient_id"]) for r in manifest["patients"] if r["split"] == "test"]
    if args.cohort == "synthetic50":
        if not args.all_test:
            print("[note] synthetic50 has no 18-scene lock file; running all test-split scenes", flush=True)
        ids = test_ids
    elif args.all_test:
        ids = test_ids
    else:
        ids = [str(pid) for pid in json.loads(IDS.read_text(encoding="utf-8"))]
    ids = ids[args.patient_start:]
    if args.patient_limit is not None:
        ids = ids[: args.patient_limit]
    if args.shard:
        index, count = args.shard
        ids = ids[index::count]
    return ids


def _done_ids(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for path in sorted(out_dir.glob("records*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(str(json.loads(line)["patient_id"]))
    return done


def run_ekf_patient(scene, experiments, prior, official, *, lock_ratio: bool, noise_px: float, pid: str) -> dict:
    center, radius = _predicted_region(scene, prior)
    initial_E_raw = float(prior["E_background_estimated"])
    initial_ratio_raw = float((official or prior)["inclusion_ratio_estimated"])
    # fit_patient clips the start point to its global box (E 1000-15000, ratio 1-5)
    # before locking/optimising; mirror that so the locked ratio is identical.
    initial_E = min(max(initial_E_raw, GLOBAL_BOUNDS["E"][0]), GLOBAL_BOUNDS["E"][1])
    initial_ratio = min(max(initial_ratio_raw, GLOBAL_BOUNDS["ratio"][0]), GLOBAL_BOUNDS["ratio"][1])
    intr = experiments[0]["render_intrinsics"]
    focal, H, W = float(intr["focal_px"]), int(intr["height_px"]), int(intr["width_px"])
    # prior state and covariance (same scales as batch material/force priors)
    sig_E = max(0.03, PRIOR_WIDTH_SCALE * float(prior.get("log_E_std", 0.06)))
    sig_r = max(0.05, PRIOR_WIDTH_SCALE * float((official or prior).get("log_ratio_std", 0.23)))
    if lock_ratio:
        x = torch.tensor([math.log(initial_E), 0.0], dtype=torch.float64)
        P = torch.diag(torch.tensor([sig_E ** 2, FORCE_PRIOR_SIGMA ** 2], dtype=torch.float64))
        Q = torch.diag(torch.tensor([1e-4, 1e-5], dtype=torch.float64))  # small random walk
        eps = (0.05, 0.02)
    else:
        x = torch.tensor([math.log(initial_E), 0.0, math.log(initial_ratio)], dtype=torch.float64)
        P = torch.diag(torch.tensor([sig_E ** 2, FORCE_PRIOR_SIGMA ** 2, sig_r ** 2], dtype=torch.float64))
        Q = torch.diag(torch.tensor([1e-4, 1e-5, 1e-4], dtype=torch.float64))
        eps = (0.05, 0.02, 0.05)
    n_state = int(x.numel())

    def _ratio_of(state: torch.Tensor) -> float:
        return initial_ratio if lock_ratio else math.exp(float(state[2]))

    steps = []
    evaluations = 0
    noise_gen = torch.Generator().manual_seed(93_000 + _patient_seed(pid))
    for experiment in experiments:
        conf = experiment["image_occlusion_confidence_multiview_seq"].to(torch.float64)
        obs_all = experiment["image_uv_deformed_multiview_seq"].to(torch.float64).clone()
        if noise_px > 0:
            obs_all = obs_all + noise_px * torch.randn(obs_all.shape, generator=noise_gen, dtype=obs_all.dtype)
        for frame in FRAMES:
            mask = (conf[frame] >= 0.2)  # (V, N)
            if int(mask.sum()) == 0:
                continue
            z = obs_all[frame][mask].reshape(-1)
            h0 = _predict_uv(scene, experiment, center, radius, float(x[0]), float(x[1]), _ratio_of(x), frame, focal, H, W)[mask].reshape(-1)
            evaluations += 1
            # central-difference Jacobian in log params
            cols = []
            for k in range(n_state):
                xp = x.clone(); xp[k] += eps[k]
                xm = x.clone(); xm[k] -= eps[k]
                hp = _predict_uv(scene, experiment, center, radius, float(xp[0]), float(xp[1]), _ratio_of(xp), frame, focal, H, W)[mask].reshape(-1)
                hm = _predict_uv(scene, experiment, center, radius, float(xm[0]), float(xm[1]), _ratio_of(xm), frame, focal, H, W)[mask].reshape(-1)
                evaluations += 2
                cols.append((hp - hm) / (2.0 * eps[k]))
            Hjac = torch.stack(cols, dim=1)  # (M, n_state)
            # predict
            P = P + Q
            # update with robust per-observation scale (RMS of observed motion vs rest)
            rest = experiment["image_uv_rest_multiview_seq"].to(torch.float64)[frame][mask].reshape(-1)
            scale = max(float(torch.sqrt((z - rest).square().mean())), TRACK_NOISE_PX)
            R_scalar = (TRACK_NOISE_PX ** 2) + (0.05 * scale) ** 2
            S = Hjac @ P @ Hjac.T + R_scalar * torch.eye(Hjac.shape[0], dtype=torch.float64)
            K = P @ Hjac.T @ torch.linalg.solve(S, torch.eye(Hjac.shape[0], dtype=torch.float64))
            innov = z - h0
            # Huber-like clipping of innovation to the 1.5*scale level
            clip = 1.5 * scale
            innov = innov.clamp(-clip, clip)
            x = x + K @ innov
            P = (torch.eye(n_state, dtype=torch.float64) - K @ Hjac) @ P
            step = {"experiment": experiment.get("experiment"), "frame": frame, "log_E": float(x[0]), "log_q": float(x[1])}
            if not lock_ratio:
                step["log_ratio"] = float(x[2])
            steps.append(step)
    E_est = math.exp(float(x[0]))
    q_est = math.exp(float(x[1]))
    ratio_est = _ratio_of(x)
    record = {
        "patient_id": pid,
        "method": "ekf_sequential_isotropic_locked_ratio" if lock_ratio else "ekf_sequential_isotropic_free_ratio",
        "observation_noise_px": noise_px,
        "E_background_estimated": E_est,
        "E_background_relative_error": relative_error(E_est, float(experiments[0]["E_background"])),
        "inclusion_ratio_relative_error": relative_error(ratio_est, float(experiments[0]["inclusion_ratio"])),
        "force_scale_estimated": q_est,
        "force_scale_relative_error": relative_error(q_est, force_scale_target(experiments)),
        "posterior_log_std": [math.sqrt(float(P[k, k])) for k in range(n_state)],
        "fem_solves": evaluations,
        "steps": steps,
        # matched-arm bookkeeping (added 2026-10-12; does not alter the estimate)
        "E_background_true": float(experiments[0]["E_background"]),
        "inclusion_ratio_true": float(experiments[0]["inclusion_ratio"]),
        "inclusion_ratio_estimated": ratio_est,
        "force_scale_true": force_scale_target(experiments),
        "lock_ratio": bool(lock_ratio),
        "initial_E_background": initial_E_raw,
        "initial_ratio": initial_ratio_raw,
        "start_E_background_clipped": initial_E,
        "start_ratio_clipped": initial_ratio,
        "prior_log_std": [sig_E, FORCE_PRIOR_SIGMA] + ([] if lock_ratio else [sig_r]),
    }
    return record


def run(args) -> None:
    out_dir: Path = args.out
    setup = _cohort_setup(args.cohort)
    dataset = setup["dataset"]
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = {str(r["patient_id"]): r for r in manifest["patients"]}
    predictions = setup["predictions"]
    official_ratios = setup["official_ratios"]
    ids = _select_ids(args, manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.shard:
        rec_path = out_dir / f"records_shard{args.shard[0]}of{args.shard[1]}.jsonl"
    else:
        rec_path = out_dir / "records.jsonl"
    done = _done_ids(out_dir)
    print(f"[{args.cohort}] {len(ids)} scenes selected, {len([i for i in ids if i in done])} already done -> {rec_path.name}", flush=True)
    with rec_path.open("a", encoding="utf-8") as handle:
        for pid in ids:
            if pid in done:
                print("skip", pid, flush=True)
                continue
            row = rows[pid]
            scene, experiments = load_patient(dataset, row)
            prior = predictions[_normalize_id(pid)]
            official = official_ratios.get(_official_ratio_key(row)) or official_ratios.get(_normalize_id(pid))
            if args.cohort == "ct120":
                # unchanged default: official 2028+bins3 ratio when available, else hybrid prior
                official = official or prior
                lock_ratio = True
            else:
                lock_ratio = setup["always_lock_ratio"] or _low_contrast(prior, official, CONTRAST_RATIO_LOCK)
            record = run_ekf_patient(
                scene, experiments, prior, official, lock_ratio=lock_ratio, noise_px=args.noise_px, pid=pid
            )
            record.update({
                "cohort": args.cohort,
                "geometry_id": row.get("geometry_id"),
                "scenario_id": row.get("scenario_id"),
                "shard": f"{args.shard[0]}/{args.shard[1]}" if args.shard else None,
            })
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                pid, round(record["E_background_relative_error"], 4),
                "ratio", round(record["inclusion_ratio_relative_error"], 4),
                "lock", int(lock_ratio), "solves", record["fem_solves"], flush=True,
            )


# --------------------------------------------------------------------------
# summary / merge
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return float(ordered[mid]) if n % 2 else float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _paired(ekf: dict[str, dict], batch_rows: list[dict], tie_tol: float = 1e-9) -> dict:
    batch = {str(r["patient_id"]): r for r in batch_rows}
    common = sorted(set(ekf) & set(batch))
    result = {"n_paired": len(common), "n_batch": len(batch), "endpoints": {}}
    for key in ("E_background_relative_error", "inclusion_ratio_relative_error", "force_scale_relative_error"):
        deltas = []
        better = worse = tied = 0
        for pid in common:
            if key not in ekf[pid] or key not in batch[pid]:
                continue
            d = float(ekf[pid][key]) - float(batch[pid][key])
            deltas.append(100.0 * d)
            if abs(d) <= tie_tol:
                tied += 1
            elif d < 0:
                better += 1
            else:
                worse += 1
        result["endpoints"][key] = {
            "better": better,
            "worse": worse,
            "tied": tied,
            "median_paired_delta_pp": round(_median(deltas), 4) if deltas else None,
            "ekf_median_percent": round(100.0 * _median([float(ekf[p][key]) for p in common if key in ekf[p]]), 4),
            "batch_median_percent": round(100.0 * _median([float(batch[p][key]) for p in common if key in batch[p]]), 4),
        }
    evals = [float(batch[p].get("function_evaluations", math.nan)) for p in common if "function_evaluations" in batch[p]]
    result["batch_median_function_evaluations"] = _median(evals) if evals else None
    # matched-protocol checks (same lock decision / initial point as the batch arm)
    checks = {"lock_ratio_equal": 0, "initial_E_equal": 0, "initial_ratio_equal": 0, "locked_ratio_value_equal": 0, "n_locked_in_both": 0, "n_checked": 0}
    for pid in common:
        e, b = ekf[pid], batch[pid]
        if "lock_ratio" not in e or "lock_ratio" not in b:
            continue
        checks["n_checked"] += 1
        checks["lock_ratio_equal"] += int(bool(e["lock_ratio"]) == bool(b["lock_ratio"]))
        if bool(e["lock_ratio"]) and bool(b["lock_ratio"]) and "inclusion_ratio_estimated" in e and "inclusion_ratio_estimated" in b:
            checks["n_locked_in_both"] += 1
            checks["locked_ratio_value_equal"] += int(abs(float(e["inclusion_ratio_estimated"]) - float(b["inclusion_ratio_estimated"])) <= 1e-9)
        if "initial_E_background" in e and "initial_E_background" in b:
            checks["initial_E_equal"] += int(abs(float(e["initial_E_background"]) - float(b["initial_E_background"])) <= 1e-9 * max(1.0, abs(float(b["initial_E_background"]))))
        if "initial_ratio" in e and "initial_ratio" in b:
            checks["initial_ratio_equal"] += int(abs(float(e["initial_ratio"]) - float(b["initial_ratio"])) <= 1e-9)
    result["matched_protocol_checks"] = checks
    return result


def summarize(args) -> None:
    out_dir: Path = args.out
    merged_path = out_dir / "records.jsonl"
    shard_paths = sorted(p for p in out_dir.glob("records_shard*.jsonl"))
    setup = _cohort_setup(args.cohort)
    manifest = json.loads((setup["dataset"] / "manifest.json").read_text(encoding="utf-8"))
    order = {str(r["patient_id"]): i for i, r in enumerate(r for r in manifest["patients"] if r["split"] == "test")}
    rows_by_id: dict[str, dict] = {}
    duplicates = 0
    sources = ([merged_path] if merged_path.exists() else []) + shard_paths
    for path in sources:
        for row in _read_jsonl(path):
            pid = str(row["patient_id"])
            if pid in rows_by_id:
                duplicates += 1
                continue
            rows_by_id[pid] = row
    if shard_paths:
        merged = [rows_by_id[p] for p in sorted(rows_by_id, key=lambda p: order.get(p, 10**9))]
        merged_path.write_text("".join(json.dumps(r) + "\n" for r in merged), encoding="utf-8")
    records = _read_jsonl(merged_path) if merged_path.exists() else []
    ekf = {str(r["patient_id"]): r for r in records}
    expected = [p for p in order] if (args.all_test or args.cohort == "synthetic50") else [str(p) for p in json.loads(IDS.read_text(encoding="utf-8"))]
    summary = {
        "protocol": (
            "EKF over log(E_bg), log(q)" + (", log(ratio) when the ratio is not contrast-locked" if args.cohort == "synthetic50" else "")
            + " with the same learned initializer, same prior scales (prior_width_scale 2.0, force prior 0.05), "
            + ("ratio locked (official 2028+bins3) and predicted region locked as in the official-prior screened FEM. "
               if args.cohort == "ct120" else
               "ratio contrast-lock decision (threshold 1.20), initial ratio and prior centre identical to the six-arm learned/learned matrix; "
               "unlocked ratio gets prior SD = material_prior_sigma[1] = max(0.05, 2*log_ratio_std). ")
            + "One measurement update per nonzero frame per load (five frames x four loads), multiview UV residuals, "
            + "central-difference Jacobians, Huber-style innovation clipping. No screening or rollback."
        ),
        "cohort": args.cohort,
        "n": len(records),
        "unique_ids": len(ekf),
        "expected_n": len(expected),
        "missing_ids": sorted(set(expected) - set(ekf)),
        "unexpected_ids": sorted(set(ekf) - set(expected)),
        "duplicate_rows_dropped_in_merge": duplicates,
        "shard_files": [p.name for p in shard_paths],
        "observation_noise_px": sorted({float(r.get("observation_noise_px", 0.0)) for r in records}),
        "ratio_locked_n": sum(bool(r.get("lock_ratio", True)) for r in records),
        "median_background_error_percent": round(100.0 * _median([r["E_background_relative_error"] for r in records]), 4) if records else None,
        "median_ratio_error_percent": round(100.0 * _median([r["inclusion_ratio_relative_error"] for r in records]), 4) if records else None,
        "median_force_error_percent": round(100.0 * _median([r["force_scale_relative_error"] for r in records]), 4) if records else None,
        "median_fem_solves": _median([float(r["fem_solves"]) for r in records]) if records else None,
        "paired_vs_batch": {},
        "records_sha256": _sha256(merged_path) if merged_path.exists() else None,
        "shard_records_sha256": {p.name: _sha256(p) for p in shard_paths},
    }
    references = dict(BATCH_REFERENCES[args.cohort])
    if args.cohort == "ct120":
        references["eighteen_scene_w2_s0p05_p2"] = BATCH_18_REFERENCE
    for name, path in references.items():
        if not path.exists():
            summary["paired_vs_batch"][name] = {"missing": str(path)}
            continue
        comparison = _paired(ekf, _read_jsonl(path))
        comparison["records_path"] = str(path.relative_to(ROOT))
        comparison["records_sha256"] = _sha256(path)
        summary["paired_vs_batch"][name] = comparison
    # EKF versus the (unrefined) learned initializer it started from
    init_pairs = [
        (abs(float(r["initial_E_background"]) - float(r["E_background_true"])) / float(r["E_background_true"]), float(r["E_background_relative_error"]))
        for r in records if "initial_E_background" in r and "E_background_true" in r
    ]
    ratio_pairs = [
        (abs(float(r["initial_ratio"]) - float(r["inclusion_ratio_true"])) / float(r["inclusion_ratio_true"]), float(r["inclusion_ratio_relative_error"]))
        for r in records if not bool(r.get("lock_ratio", True)) and "initial_ratio" in r and "inclusion_ratio_true" in r
    ]

    def _vs_init(pairs: list[tuple[float, float]]) -> dict | None:
        if not pairs:
            return None
        deltas = [100.0 * (f - i) for i, f in pairs]
        return {
            "n": len(pairs),
            "ekf_better_than_initializer": sum(f < i for i, f in pairs),
            "ekf_worse_than_initializer": sum(f > i for i, f in pairs),
            "tied": sum(f == i for i, f in pairs),
            "median_initializer_error_percent": round(100.0 * _median([i for i, _ in pairs]), 4),
            "median_ekf_error_percent": round(100.0 * _median([f for _, f in pairs]), 4),
            "median_paired_delta_pp": round(_median(deltas), 4),
        }

    summary["paired_vs_initializer"] = {
        "E_background_relative_error": _vs_init(init_pairs),
        "inclusion_ratio_relative_error_unlocked_only": _vs_init(ratio_pairs),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("protocol",)}, indent=2))


def _parse_shard(text: str) -> tuple[int, int]:
    index, count = text.split("/")
    index, count = int(index), int(count)
    if count < 1 or not 0 <= index < count:
        raise ValueError("--shard must be i/n with 0 <= i < n")
    return index, count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("ct120", "synthetic50"), default="ct120")
    parser.add_argument("--all-test", action="store_true", help="run every test-split scene instead of the 18 locked CT IDs")
    parser.add_argument("--patient-start", type=int, default=0)
    parser.add_argument("--patient-limit", type=int, default=None)
    parser.add_argument("--shard", type=_parse_shard, default=None, help="i/n round-robin shard of the selected scenes")
    parser.add_argument("--noise-px", type=float, default=0.0, help="add i.i.d. pixel noise to observed tracks (protocol parity with batch fitter)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summarize", action="store_true", help="merge shard records and write summary.json (no FEM runs)")
    args = parser.parse_args()
    if args.summarize:
        summarize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
