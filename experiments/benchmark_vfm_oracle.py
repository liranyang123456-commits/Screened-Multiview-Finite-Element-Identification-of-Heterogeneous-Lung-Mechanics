"""Virtual Fields Method (VFM) oracle comparator (AIIM R3).

Classical full-field inverse-elasticity reference. VFM receives MORE
information than the proposed method and is therefore an upper-bound classical
comparator, not a matched competitor:

  - full interior nodal kinematics at the peak frame (oracle full field),
  - true inclusion region (oracle region) for the two-modulus split,
  - measured forces (5% measurement error, matched to the main protocol).

For compressible Neo-Hookean elasticity with fixed Poisson ratio the internal
force is linear in the shear modulus, so the Galerkin virtual-work equations
f_int(u; mu) . u* = f_ext . u* are linear in (mu_bg, mu_inclusion) and are
solved as a small least-squares system per patient (3 virtual fields x 4
loads = 12 equations). Virtual fields vanish on the fixed plane so that
reaction forces drop out of the external work.

Output: results/lung_mesh_ai_v2/vfm_oracle_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.evaluate_sim_lung_v2 import load_patient  # noqa: E402
from physics.fem import _local_fint, _shape_grads  # noqa: E402

DATASET = ROOT / "dataset" / "sim_lung_ai_v2_multiview250"
OUT = ROOT / "results" / "lung_mesh_ai_v2" / "vfm_oracle_benchmark.json"
NU = 0.45
LAM_OVER_MU = 2.0 * NU / (1.0 - 2.0 * NU)  # lambda/mu = 9


def affine_virtual_fields(
    X: torch.Tensor, fixed_nodes: torch.Tensor
) -> list[torch.Tensor]:
    """Ramps u*_k = ((X_k - lo_k)/L_k) * w(z), w zero on the fixed plane.

    Virtual fields must vanish on Dirichlet DOFs so that reaction forces drop
    out of the external virtual work. The fixed set is a single plane normal
    to axis z in this cohort, so w(z) = (z_fixed - z)/L_z.
    """
    lo = X.amin(dim=0)
    hi = X.amax(dim=0)
    extent = (hi - lo).clamp_min(1e-9)
    z_fixed = float(X[fixed_nodes][:, 2].median())
    weight = (z_fixed - X[:, 2]) / extent[2]
    fields = []
    for k in range(3):
        ramp = (X[:, k] - lo[k]) / extent[k]
        values = torch.zeros_like(X)
        values[:, k] = ramp * weight
        fields.append(values)
    return fields


def unit_internal_force(
    u: torch.Tensor,
    X: torch.Tensor,
    elems: torch.Tensor,
    dN: torch.Tensor,
    vol: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Assembled internal force with mu=1 (lambda=lambda/mu) on `mask` elements."""
    ne = elems.shape[0]
    mu = torch.where(mask, 1.0, 0.0).to(u.dtype)
    lam = mu * LAM_OVER_MU
    uloc = u[elems].reshape(ne, 12).detach().clone().requires_grad_(True)
    fe = _local_fint(uloc, X[elems], dN, vol, mu, lam, ne, 3).view(ne, 4, 3)
    out = torch.zeros(X.shape[0], 3, dtype=u.dtype)
    for a in range(4):
        out.index_add_(0, elems[:, a], fe[:, a, :])
    return out.detach()


def vfm_patient(
    scene: dict,
    experiments: list[dict],
    X: torch.Tensor,
    elems: torch.Tensor,
    dN: torch.Tensor,
    vol: torch.Tensor,
    fields: list[torch.Tensor],
    force_field: str = "forces_measured",
) -> tuple[float, float, float, float]:
    rows_a: list[list[float]] = []
    rows_b: list[float] = []
    for experiment in experiments:
        u_seq = experiment["u_seq"].double().view(7, -1, 3)
        force_key = "forces" if force_field == "forces" else "forces_measured"
        f_seq = experiment[force_key].double().view(7, -1, 3)
        frame = int(f_seq.norm(dim=(1, 2)).argmax())
        u = u_seq[frame]
        f = f_seq[frame]

        center = experiment["inclusion_center"].double().view(-1)
        radius = float(experiment["inclusion_radius"])
        centroids = X[elems].mean(dim=1)
        inside = (centroids - center).norm(dim=1) <= radius
        homogeneous = abs(float(experiment["inclusion_ratio"]) - 1.0) < 1e-9

        f_bg = unit_internal_force(u, X, elems, dN, vol, ~inside)
        f_in = (
            unit_internal_force(u, X, elems, dN, vol, inside)
            if not homogeneous
            else None
        )
        for values in fields:
            a_bg = float((f_bg * values).sum())
            a_in = 0.0 if homogeneous else float((f_in * values).sum())
            rows_a.append([a_bg, a_in])
            rows_b.append(float((f * values).sum()))

    A = torch.tensor(rows_a, dtype=torch.float64)
    b = torch.tensor(rows_b, dtype=torch.float64)
    solution = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    mu_bg = max(float(solution[0]), 1.0)
    mu_in = max(float(solution[1]), mu_bg) if solution.numel() > 1 else mu_bg
    e_bg = 2.0 * mu_bg * (1.0 + NU)
    ratio = mu_in / mu_bg
    truth_e = float(experiments[0]["E_background"])
    truth_ratio = float(experiments[0]["inclusion_ratio"])
    return e_bg, ratio, truth_e, truth_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-limit", type=int, default=50)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--force-field", choices=("forces_measured", "forces"), default="forces_measured",
        help="Use measured forces for matched comparison or true forces for noise-free ceiling.",
    )
    parser.add_argument("--seed-bootstrap", type=int, default=20261007)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    test = [row for row in manifest["patients"] if row["split"] == "test"]
    test = test[: args.patient_limit]

    started = time.time()
    records = []
    for index, patient in enumerate(test):
        scene, experiments = load_patient(DATASET, patient)
        X = scene["nodes"].double()
        elems = scene["elems"]
        xe = X[elems]
        dN = _shape_grads(xe, 3)
        cols = torch.stack([xe[:, k + 1] - xe[:, 0] for k in range(3)], dim=-1)
        vol = torch.linalg.det(cols).abs() / 6.0
        fixed_nodes = torch.tensor(
            sorted({int(d // 3) for d in scene["fixed"]}), dtype=torch.long
        )
        fields = affine_virtual_fields(X, fixed_nodes)
        e_est, ratio_est, e_true, ratio_true = vfm_patient(
            scene, experiments, X, elems, dN, vol, fields, args.force_field
        )
        records.append(
            {
                "patient_id": patient["patient_id"],
                "E_background_estimated": e_est,
                "E_background_true": e_true,
                "E_background_relative_error": abs(e_est / e_true - 1.0),
                "inclusion_ratio_estimated": ratio_est,
                "inclusion_ratio_true": ratio_true,
                "inclusion_ratio_relative_error": abs(ratio_est / ratio_true - 1.0),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(test)} patients", flush=True)

    e_errors = [row["E_background_relative_error"] for row in records]
    ratio_errors = [
        row["inclusion_ratio_relative_error"]
        for row in records
        if row["inclusion_ratio_true"] > 1.0 + 1e-9
    ]
    rng = np.random.default_rng(args.seed_bootstrap)
    e_boot = np.asarray([
        np.median(rng.choice(e_errors, size=len(e_errors), replace=True))
        for _ in range(args.bootstrap_replicates)
    ])
    ratio_boot = np.asarray([
        np.median(rng.choice(ratio_errors, size=len(ratio_errors), replace=True))
        for _ in range(args.bootstrap_replicates)
    ])
    result = {
        "method": "vfm_oracle_fullfield",
        "information": "oracle full interior kinematics + oracle region + selected force field",
        "force_field": args.force_field,
        "noise_free_true_force_ceiling": args.force_field == "forces",
        "patients": len(records),
        "runtime_seconds": time.time() - started,
        "bootstrap_seed": args.seed_bootstrap,
        "bootstrap_replicates": args.bootstrap_replicates,
        "E_background_median_relative_error": float(np.median(e_errors)),
        "E_background_median_bootstrap_95_ci": [float(x) for x in np.quantile(e_boot, [0.025, 0.975])],
        "inclusion_ratio_median_relative_error": float(np.median(ratio_errors)),
        "inclusion_ratio_median_bootstrap_95_ci": [float(x) for x in np.quantile(ratio_boot, [0.025, 0.975])],
        "E_background_mean_relative_error": float(np.mean(e_errors)),
        "inclusion_ratio_mean_relative_error": float(np.mean(ratio_errors)),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
