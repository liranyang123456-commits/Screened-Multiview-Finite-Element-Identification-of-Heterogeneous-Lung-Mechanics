
# Heterogeneous Lung Mechanics from Multiview Motion

Reference implementation and frozen numerical evidence for:

> **Calibrated Multiview Inverse Finite-Element Identification of
> Heterogeneous Lung Mechanics with Screened Physics Auditing**

Repository:
https://github.com/liranyang123456-commits/Screened-Multiview-Finite-Element-Identification-of-Heterogeneous-Lung-Mechanics

The project estimates background Young's modulus, inclusion contrast, region
center/radius, and node-level material fields from synchronized multiview motion
under known loads. The primary method uses a train-only-selected PCA--Ridge
response calibrator as a prior for screened inverse FEM.

## Evidence boundary

- Material-property truth is synthetic.
- The main benchmark contains 250 procedural CT-like surrogate geometries,
  split 150/50/50 by simulated patient.
- The CT-geometry benchmark contains 540 synthetic-mechanics scenes instantiated
  on 27 de-identified patient-derived geometries with an 18/3/6 split.
- The CT geometries are not patient material ground truth.
- No raw DICOM, clinical video, patient identifier, patient-derived mesh/pixel
  asset, model checkpoint, or generated tensor dataset is distributed here.

## Included

- Core simulation, graph, learning, inverse-FEM, evaluation, and test code.
- Frozen aggregate and scene-level numerical JSON evidence.
- Dataset manifests and protocol/QC reports without source paths.
- Publication-safe quantitative figures.
- Scripts for smoke testing and reproducing summary tables.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt  # DICOM/visualization utilities
```

## Quick verification

```bash
python scripts/verify_release.py
python scripts/summarize_ct_benchmark.py
pytest experiments/test_benchmark_ion_ct_synthetic_mechanics.py \
       experiments/test_export_ion_ct_figure_assets.py \
       experiments/test_generate_ion_ct_synthetic_mechanics.py \
       lung_inverse_rendering/test_lung_pipeline.py \
       evaluation/test_material_uncertainty_metrics.py \
       evaluation/test_aggregate_multiseed.py -q
```

## Reproducing experiments

1. Generate the procedural multiview synthetic cohort using
   `lung_inverse_rendering/generate_sim_lung_v2.py`.
2. Train the response calibrator with
   `experiments/train_lung_response_calibrator.py`.
3. Train/evaluate MeshGNN with `experiments/train_lung_mesh_gnn.py` and
   `evaluation/evaluate_lung_mesh_gnn.py`.
4. Run matched inverse-FEM evaluation with
   `lung_inverse_rendering/evaluate_sim_lung_v2.py`.
5. Aggregate the external benchmark using
   `experiments/benchmark_ion_ct_synthetic_mechanics.py`.

Large generated tensors and checkpoints are intentionally not versioned. The
frozen JSON files under `results/` reproduce manuscript statistics without
requiring restricted source data.

## Reviewer-revision experiments

Every first-round-revision experiment is driven by a committed script. The
complete command list with flags and archived output directories is
`experiments/reproduce_reviewer_arms.ps1` (dry run by default; `-Section
<name>` selects a block, `-Execute` runs it). Scene-level `records.jsonl`
files and their `run_manifest.json` protocol files are shipped byte-for-byte,
so the `records_sha256` / `run_manifest_sha256` fields quoted in the group
summaries can be re-hashed, and every group summary can be regenerated from
records without running FEM:

```bash
python experiments/summarize_reviewer_arms.py --check          # all groups below
python experiments/run_assumption_sensitivity_ct18.py --summarize-only
python experiments/reconcile_ct_reviewer_evidence.py
```

Locked cohorts: 18 CT-derived test scenes (six held-out geometries x three
scenarios; `results/ct_reviewer_drivers_v6/paired_scene_ids.json`), the 50
procedural synthetic test cases, and the 120 locked CT test scenes. The
18-scene and six-arm FEM arms below use `experiments/run_screened_fem_locked.py`
with `--prior-width-scale 2.0 --screen-threshold 0.05 --probe-count 2
--max-nfev 48` unless the flag is the varied factor; the 120-scene official
audit and the frozen primary run use prior-width multiplier 1 (see
`results/ct_reviewer_drivers_v6/ct120_audit_mechanics.json`,
`prior_width_scale_evidence`), so acceptance rates are protocol-specific
and are not comparable across protocols.

| Experiment | Driver / flags | Summary (and records) |
| --- | --- | --- |
| Primary 50-case screened MAP vs fixed-init; no-screen counterfactual | `lung_inverse_rendering/evaluate_sim_lung_v2.py` (96 evaluations, `--minimum-refinement-cost-reduction 0.05`); `experiments/run_aiim_sensitivity_sweeps.py --configs noscreen` | `results/lung_mesh_ai_v2/fair_multiview_fem50_strict/fair_multiview_summary.json`; `results/lung_mesh_ai_v2/aiim_sensitivity/sensitivity_noscreen.json` |
| Six-arm 50-case matched controls (init x prior centre, no material prior, residual objective/screen) | `run_screened_fem_locked.py --cohort synthetic50 --initialization-mode {learned,fixed} --prior-center-mode {learned,fixed}`; `--material-prior-weight 0 --prior-width-scale 1`; `--screening-metric residual` | `results/lung_mesh_ai_v2/synthetic50_full_paired_controls_20261008/summary_50patient_paired.json` (+ six `*/records.jsonl`) |
| 50-case early-threshold sweep 0 / 0.01 / 0.025 / 0.10 / 0.20 | `--cohort synthetic50 --screen-threshold <t>` (final acceptance stays 0.05) | `results/lung_mesh_ai_v2/synthetic50_threshold_sweep/summary.json` |
| 18-scene baseline, six matched controls, threshold/probe/width sensitivity (11 arms), material-prior-weight sweep, nine-arm force/yaw mismatch | `experiments/run_ct_reviewer_drivers.py --run --patient-ids-file results/ct_reviewer_drivers_v6/paired_scene_ids.json [--include-controls] --config-id <arm>`; mismatch via `--model-force-bias {-0.20,-0.10,0.05,0.10,0.20}` and `--pose-rotation-error-deg {-10,-5,5,10}` | `results/ct_reviewer_drivers_v6/reconciled_paired_experiments.json` (registry with SHA-256 of every records file) |
| Camera-yaw +/-10 deg arms re-run at the common 48-evaluation budget (the archived 16-evaluation arms are kept as reference) | `run_screened_fem_locked.py --cohort ct120 --pose-rotation-error-deg {-10,10} --max-nfev 48 --patient-ids <18 locked ids>` | `results/ct_reviewer_drivers_v6/model_mismatch_n48/summary.json` (+ per-arm `records.jsonl`, `run_manifest.json`) |
| 18-scene assumption / proxy arms: Poisson ratio 0.40/0.49, fixation quantile 0.70/0.90, ellipsoid aspect 1.5/2.0, joint +10 % force / +5 deg yaw, 20 % correspondence dropout, 2 % mesh jitter, 15 % contact shift | `experiments/run_assumption_sensitivity_ct18.py` (`--model-nu`, `--posterior-fixation-quantile`, `--region-aspect-xy`, `--tracking-error-model dropout --tracking-dropout-rate`, `--mesh-jitter-fraction`, `--contact-shift-fraction`) | `results/ct_reviewer_drivers_v6/assumption_sensitivity/summary.json` |
| Corrected ellipsoid-aspect arms (region attention always from the predicted centre/radius; the archived arms fell back to the true region) and the occupancy sanity check | `run_screened_fem_locked.py --cohort ct120 --region-aspect-xy {1.5,2.0} --max-nfev 48 --patient-ids <18 locked ids>`; `experiments/aspect_arm_sanity.py` | `results/ct_reviewer_drivers_v6/assumption_sensitivity_fixed/summary.json`; `results/ct_reviewer_drivers_v6/assumption_sensitivity/aspect_arm_sanity.json` |
| Constitutive mismatch (fiber / viscous observation laws, isotropic fitter) | `experiments/audit_constitutive_mismatch.py --law {fiber,viscous}` (`physics/constitutive_mismatch.py`) | `results/ct_reviewer_drivers_v6/constitutive_mismatch/summary.json` |
| Structural mismatch (two inclusions, diffuse field, sliding boundary, matched control) | `experiments/audit_structural_mismatch.py --mode {matched,two_inclusions,diffuse,sliding_bc}` | `results/ct_reviewer_drivers_v6/structural_mismatch/summary.json` |
| Learned initializer re-inferred on the mismatched observations of the 18 scenes | `experiments/reinfer_mismatch_initializer.py` | `results/ct_reviewer_drivers_v6/mismatch_initializer_reinference/summary.json` (+ `records.jsonl`) |
| Sequential EKF, 18 scenes (0 / 0.05 px observation noise) | `experiments/run_ekf_sequential_ct18.py [--noise-px 0.05 --out results/ct_reviewer_drivers_v6/ekf_sequential_noise0p05]` | `results/ct_reviewer_drivers_v6/ekf_sequential/summary.json`, `.../ekf_sequential_noise0p05/summary.json` |
| Sequential EKF, cohort level: all 120 CT scenes (official ratio locked, prior width 1) and the 50 synthetic cases (six-arm lock rule) | `experiments/run_ekf_sequential_ct18.py --cohort ct120 --all-test --noise-px 0.05 --shard k/4`; `--cohort synthetic50 --all-test --noise-px 0.05 --shard k/4` | `results/ct_reviewer_drivers_v6/ekf_sequential_ct120_all/summary.json`; `results/lung_mesh_ai_v2/ekf_sequential_synthetic50/summary.json` (its `superseded_v0_unclipped_initial_ratio/` is the discarded first run, see `WHY_SUPERSEDED.txt`) |
| Decoupled screening on the 18 scenes (residual screen, MAP objective) | `--screening-metric residual --objective-metric map` | `results/ct_reviewer_drivers_v6/decoupled_screen/summary.json` |
| Decoupled residual screen + acceptance on the 50 cases (width 1 / 96 evaluations and width 2 / 48 evaluations; plus the plain MAP arm at width 1 / 96) | `run_screened_fem_locked.py --cohort synthetic50 --screening-metric residual --objective-metric map --acceptance-metric residual --prior-width-scale {1.0,2.0} --max-nfev {96,48}` | `results/lung_mesh_ai_v2/synthetic50_decoupled_residual/comparison.json` (+ per-arm `summary.json`, `records.jsonl`) |
| Record-only audits: accepted-update harm, initializer-only reference on the 18 scenes, four-bin calibration test sensitivity, CT 120 audit mechanics (prior-width-1 evidence) | `experiments/accepted_update_harm.py` (harm statistics); the other three are record-only recomputations from the shipped `records.jsonl` files, each JSON listing its input paths and SHA-256 digests | `results/ct_reviewer_drivers_v6/accepted_update_harm.json`, `initializer_reference_18.json`, `calibration_fourbin_test_sensitivity.json`, `ct120_audit_mechanics.json` |
| Population-mean prior centre | `--prior-center-mode fixed --fixed-prior-E 4425.1653061175575` | `results/ct_reviewer_drivers_v6/population_prior/summary.json`, `population_prior_E.json` |
| Training-only nested calibration resampling (two seeds) and LOO refit | `experiments/audit_nested_calibration_resampling.py --repeats 1000 --heldout-geometries 4 --seed {20261008,20261009}`; `--loo-refit-only` | `results/ct_reviewer_drivers_v6/repeated_outer_audit_seed*_1000.json`, `calibration_stability_loo_refit.json` |
| CT 120-scene official-prior screened FEM and leave-one-geometry-out influence | `run_screened_fem_locked.py --cohort ct120 --max-nfev 48 --prior-width-scale 1.0` (the audit used width multiplier 1; the current driver default is 2.0); `experiments/audit_ct_geometry_influence.py` | `results/ion_ct_official_prior_screened_fem120_v1/summary.json`, `records.jsonl`, `geometry_influence_audit.json`; `results/ct_reviewer_drivers_v6/ct120_audit_mechanics.json` |
| Seven-parameter local identifiability | `experiments/audit_joint_identifiability_ct.py` | `results/ct_reviewer_drivers_v6/joint_identifiability_audit_v1.json` |
| Temporal amplitude rank-one projection audit | `experiments/audit_temporal_amplitude.py` | `results/lung_mesh_ai_v2/temporal_amplitude_audit_test50.json` |
| VFM true-force oracle ceiling | `experiments/benchmark_vfm_oracle.py --patient-limit 50 --force-field forces` | `results/lung_mesh_ai_v2/vfm_oracle_trueforce_50.json` |
| Identity / provenance of the 27 CT geometries | `experiments/audit_ion_dataset.py`, `lung_inverse_rendering/prepare_ion_ct_meshes.py --export`, `experiments/build_ion_geometry_split.py`, `experiments/validate_ion_geometry_cohort.py`, `experiments/validate_ion_ct_synthetic_mechanics.py` | requires the restricted source collection; released outputs contain opaque `geom_*` IDs and hashes only (`results/ion_geometry_ood/`, `dataset/ion_ct_synthetic_mechanics540/manifest.json`) |

All medians are arithmetic (mean of the two central values for even n). The
18-scene CT pilots lock the official calibrated ratio prediction, whose median
error on that 18-scene subset is 54.39 % (17.77 % over all 120 scenes); the
pilots use prior-width multiplier 2 whereas the 120-scene audit uses
multiplier 1, so the two must not be pooled. The screened batch audit does not
improve background accuracy on either main cohort (3.82 -> 4.11 % synthetic,
8.55 -> 11.70 % CT), whereas the all-frame sequential EKF started from the same
initializer does (2.68 % and 3.12 %); the EKF has no screening or rollback and
is not a replacement of the frozen primary results.

## Citation

See `CITATION.cff`.

## License

Code is released under the MIT License. Numerical evidence and figures are
provided for research reproducibility with attribution. Third-party datasets
and patient-derived source assets are not redistributed.
