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
* ``--prior-width-scale`` (2026-10-12, reviewer fix).  Multiplier on the
  initializer log-std that sets the initial EKF material covariance,
  sig_E = max(0.03, w * log_E_std) and (unlocked ratio) sig_r = max(0.05,
  w * log_ratio_std) -- the same formula the batch fitter uses for
  material_prior_sigma.  Default 2.0 preserves every record/summary written
  before the flag existed.  The CT120 official-prior audit and the frozen
  primary synthetic screened-MAP run used width 1, so a matched EKF rerun
  needs ``--prior-width-scale 1.0``.  The width is recorded in each record
  (``prior_width_scale``, ``prior_log_std_inputs``) and in
  ``summary.json["prior_width_scale"]``; ``matched_protocol_checks`` compares
  the EKF prior SD with the batch arm's material_prior_sigma (width read from
  the arm's run_manifest.json / summary.json protocol, else the documented
  legacy value).

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
PRIOR_WIDTH_SCALE = 2.0       # default for --prior-width-scale (six-arm learned/learned value; keeps old records reproducible)
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
# Arms written before run_screened_fem_locked.py recorded prior_width_scale in
# run_manifest.json / summary.json["protocol"].  Values as documented in the
# manuscript (primary synthetic run and CT120 official audit: width multiplier 1).
BATCH_REFERENCE_PRIOR_WIDTH_DOCUMENTED = {
    "official_prior_screened_fem120_v1": 1.0,
    "frozen_primary_screened_map_locked50_v1": 1.0,
}
MATERIAL_PRIOR_FLOORS = (0.03, 0.05)  # same floors as run_screened_fem_locked material_prior_sigma


def _prior_std_inputs(prior: dict, official: dict | None) -> tuple[float, float]:
    """(log_E_std, log_ratio_std) exactly as the batch fitter reads them."""
    return (
        float(prior.get("log_E_std", 0.06)),
        float((official or prior).get("log_ratio_std", 0.23)),
    )


def _material_prior_sigma(width: float, std_inputs: tuple[float, float]) -> tuple[float, float]:
    """run_screened_fem_locked material_prior_sigma for a given width multiplier."""
    return (
        max(MATERIAL_PRIOR_FLOORS[0], float(width) * std_inputs[0]),
        max(MATERIAL_PRIOR_FLOORS[1], float(width) * std_inputs[1]),
    )


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


def _done_ids(out_dir: Path, prior_width_scale: float | None = None) -> set[str]:
    """IDs already written in the output directory.

    When ``prior_width_scale`` is given, refuse to resume into a directory whose
    existing records were produced with a different width (records written
    before the flag existed carry no field and count as the old hard-coded 2.0).
    """
    done: set[str] = set()
    widths: set[float] = set()
    for path in sorted(out_dir.glob("records*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add(str(row["patient_id"]))
                widths.add(float(row.get("prior_width_scale", PRIOR_WIDTH_SCALE)))
    if prior_width_scale is not None and any(abs(w - float(prior_width_scale)) > 1e-12 for w in widths):
        raise RuntimeError(
            f"{out_dir} already holds records with prior_width_scale {sorted(widths)}; "
            f"refusing to append records with prior_width_scale={prior_width_scale}"
        )
    return done


def run_ekf_patient(
    scene, experiments, prior, official, *, lock_ratio: bool, noise_px: float, pid: str,
    prior_width_scale: float = PRIOR_WIDTH_SCALE,
) -> dict:
    center, radius = _predicted_region(scene, prior)
    initial_E_raw = float(prior["E_background_estimated"])
    initial_ratio_raw = float((official or prior)["inclusion_ratio_estimated"])
    # fit_patient clips the start point to its global box (E 1000-15000, ratio 1-5)
    # before locking/optimising; mirror that so the locked ratio is identical.
    initial_E = min(max(initial_E_raw, GLOBAL_BOUNDS["E"][0]), GLOBAL_BOUNDS["E"][1])
    initial_ratio = min(max(initial_ratio_raw, GLOBAL_BOUNDS["ratio"][0]), GLOBAL_BOUNDS["ratio"][1])
    intr = experiments[0]["render_intrinsics"]
    focal, H, W = float(intr["focal_px"]), int(intr["height_px"]), int(intr["width_px"])
    # prior state and covariance (same formula as the batch material_prior_sigma,
    # with the width multiplier taken from --prior-width-scale)
    std_inputs = _prior_std_inputs(prior, official)
    sig_E, sig_r = _material_prior_sigma(prior_width_scale, std_inputs)
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
        # width multiplier and the initializer log-stds it was applied to (2026-10-12)
        "prior_width_scale": float(prior_width_scale),
        "prior_log_std_inputs": [std_inputs[0], std_inputs[1]],
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
    done = _done_ids(out_dir, args.prior_width_scale)
    print(
        f"[{args.cohort}] {len(ids)} scenes selected, {len([i for i in ids if i in done])} already done -> {rec_path.name}"
        f" (prior_width_scale={args.prior_width_scale}, noise_px={args.noise_px})",
        flush=True,
    )
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
                scene, experiments, prior, official, lock_ratio=lock_ratio, noise_px=args.noise_px, pid=pid,
                prior_width_scale=args.prior_width_scale,
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


def _batch_prior_width(name: str, records_path: Path) -> tuple[float | None, str]:
    """Width multiplier the batch arm used for material_prior_sigma, and where it came from."""
    arm_dir = records_path.parent
    manifest = arm_dir / "run_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("prior_width_scale") is not None:
            return float(payload["prior_width_scale"]), "run_manifest.json"
    summary_path = arm_dir / "summary.json"
    if summary_path.exists():
        protocol = json.loads(summary_path.read_text(encoding="utf-8")).get("protocol", {})
        if isinstance(protocol, dict) and protocol.get("prior_width_scale") is not None:
            return float(protocol["prior_width_scale"]), "summary.json protocol"
    if name in BATCH_REFERENCE_PRIOR_WIDTH_DOCUMENTED:
        return BATCH_REFERENCE_PRIOR_WIDTH_DOCUMENTED[name], "documented legacy value (arm predates prior_width_scale bookkeeping)"
    return None, "unknown (no run_manifest.json / summary.json protocol entry and no documented value)"


def _paired(
    ekf: dict[str, dict],
    batch_rows: list[dict],
    tie_tol: float = 1e-9,
    *,
    batch_prior_width: float | None = None,
    batch_prior_width_source: str | None = None,
    std_inputs: dict[str, tuple[float, float]] | None = None,
) -> dict:
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
    # prior-width check: EKF initial material SD vs the batch arm's material_prior_sigma
    # (batch sigma = max(floor, batch_width * initializer log-std), recomputed from the
    # same initializer log-stds the EKF record used; the batch records themselves do
    # not store material_prior_sigma, only the arm manifest stores the width).
    ekf_widths = sorted({float(e.get("prior_width_scale", PRIOR_WIDTH_SCALE)) for e in (ekf[p] for p in common)})
    prior_check = {
        "ekf_prior_width_scale": ekf_widths[0] if len(ekf_widths) == 1 else ekf_widths,
        "batch_prior_width_scale": batch_prior_width,
        "batch_prior_width_source": batch_prior_width_source,
        "width_scale_equal": (
            None if batch_prior_width is None or len(ekf_widths) != 1
            else bool(abs(ekf_widths[0] - float(batch_prior_width)) <= 1e-12)
        ),
        "n_prior_sd_checked": 0,
        "prior_sd_E_equal": 0,
        "n_ratio_sd_checked": 0,
        "prior_sd_ratio_equal": 0,
        "max_abs_prior_sd_E_difference": 0.0,
    }
    if batch_prior_width is not None:
        for pid in common:
            e = ekf[pid]
            inputs = (
                tuple(e["prior_log_std_inputs"]) if "prior_log_std_inputs" in e
                else (std_inputs or {}).get(pid)
            )
            if inputs is None or "prior_log_std" not in e:
                continue
            batch_sigma = _material_prior_sigma(batch_prior_width, (float(inputs[0]), float(inputs[1])))
            ekf_sigma = [float(v) for v in e["prior_log_std"]]
            prior_check["n_prior_sd_checked"] += 1
            diff_E = abs(ekf_sigma[0] - batch_sigma[0])
            prior_check["prior_sd_E_equal"] += int(diff_E <= 1e-9)
            prior_check["max_abs_prior_sd_E_difference"] = max(prior_check["max_abs_prior_sd_E_difference"], diff_E)
            if not bool(e.get("lock_ratio", True)) and len(ekf_sigma) >= 3:
                prior_check["n_ratio_sd_checked"] += 1
                prior_check["prior_sd_ratio_equal"] += int(abs(ekf_sigma[2] - batch_sigma[1]) <= 1e-9)
    prior_check["prior_sd_all_equal"] = (
        None if batch_prior_width is None or prior_check["n_prior_sd_checked"] == 0
        else bool(
            prior_check["prior_sd_E_equal"] == prior_check["n_prior_sd_checked"]
            and prior_check["prior_sd_ratio_equal"] == prior_check["n_ratio_sd_checked"]
        )
    )
    checks["prior_width"] = prior_check
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
    # width actually used by the records (legacy records predate the field -> 2.0)
    record_widths = sorted({float(r.get("prior_width_scale", PRIOR_WIDTH_SCALE)) for r in records})
    width = record_widths[0] if len(record_widths) == 1 else float(args.prior_width_scale)
    # initializer log-stds per scene (same lookup as run()) for the prior-SD check on legacy records
    std_inputs: dict[str, tuple[float, float]] = {}
    for row in manifest["patients"]:
        pid = str(row["patient_id"])
        if pid not in ekf:
            continue
        prior = setup["predictions"].get(_normalize_id(pid))
        if prior is None:
            continue
        official = setup["official_ratios"].get(_official_ratio_key(row)) or setup["official_ratios"].get(_normalize_id(pid))
        std_inputs[pid] = _prior_std_inputs(prior, official)  # falls back to prior when official is None, as run() does
    summary = {
        "protocol": (
            "EKF over log(E_bg), log(q)" + (", log(ratio) when the ratio is not contrast-locked" if args.cohort == "synthetic50" else "")
            + f" with the same learned initializer, same prior scales (prior_width_scale {width:g}, force prior 0.05; "
            + f"initial material SD = max(0.03, {width:g}*log_E_std) as in the batch material_prior_sigma), "
            + ("ratio locked (official 2028+bins3) and predicted region locked as in the official-prior screened FEM. "
               if args.cohort == "ct120" else
               "ratio contrast-lock decision (threshold 1.20), initial ratio and prior centre identical to the six-arm learned/learned matrix; "
               f"unlocked ratio gets prior SD = material_prior_sigma[1] = max(0.05, {width:g}*log_ratio_std). ")
            + "One measurement update per nonzero frame per load (five frames x four loads), multiview UV residuals, "
            + "central-difference Jacobians, Huber-style innovation clipping. No screening or rollback."
        ),
        "prior_width_scale": width,
        "prior_width_scale_in_records": record_widths,
        "prior_width_scale_recorded_per_record": sum("prior_width_scale" in r for r in records),
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
        batch_width, batch_width_source = _batch_prior_width(name, path)
        comparison = _paired(
            ekf, _read_jsonl(path),
            batch_prior_width=batch_width, batch_prior_width_source=batch_width_source, std_inputs=std_inputs,
        )
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
    parser.add_argument(
        "--prior-width-scale", type=float, default=PRIOR_WIDTH_SCALE,
        help="multiplier on the initializer log-std for the initial material covariance "
             "(sig_E = max(0.03, w*log_E_std)); 2.0 = six-arm learned/learned (legacy default), "
             "1.0 = CT120 official audit / frozen primary synthetic protocol",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summarize", action="store_true", help="merge shard records and write summary.json (no FEM runs)")
    args = parser.parse_args()
    if args.prior_width_scale <= 0:
        parser.error("--prior-width-scale must be positive")
    if args.summarize:
        summarize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
