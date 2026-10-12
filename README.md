
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
- Every reviewer-revision driver, audit and summary script, with the exact
  command list (`experiments/reproduce_reviewer_arms.ps1`).
- Frozen aggregate and scene-level numerical JSON evidence, including the
  byte-identical `records.jsonl` / `run_manifest.json` files whose SHA-256
  digests are quoted in the supplement.
- Dataset manifests and protocol/QC reports without source paths.
- Publication-safe quantitative figures and the scripts that draw them
  (`paper_aiim/figures/`), plus the manuscript build helpers used for the
  revision (`paper_aiim/revision_tools/`).
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

## Figures

`paper_aiim/figures/` holds the scripts that drew every figure of the
manuscript and supplement (the supplement cites them under this path). All
scripts resolve the repository root from their own location, so they run
unchanged from a checkout; pass `--output-dir figures` (where supported) to
regenerate into the shipped figure folder. Figures that need the generated
tensor cohort, the de-identified CT meshes or model checkpoints cannot be
redrawn from this repository and are marked below.

| Figure stem(s) | Script | Inputs (all shipped unless noted) |
| --- | --- | --- |
| `fig_pipeline`, `fig_dataset_protocol`, `fig_main_results` (original illustrated designs, redrawn at the printed 6.5 in width with all labels >= 7 pt) | `generate_aiim_figures.py --only <stem>` | `dataset/sim_lung_ai_v2_multiview250/manifest.json`, `results/lung_mesh_ai_v2/publication_gates_final.json`, `results/lung_mesh_ai_v2/node_response_calibrator_strict.json` |
| `*_simplified`, `fig_ct_controlled_distributions` (auxiliary reduced drawings explored during the revision; not compiled into the manuscript) | `plot_revision_overview.py` | `dataset/sim_lung_ai_v2_multiview250/manifest.json`, `results/lung_mesh_ai_v2/node_response_calibrator_strict.json`, `results/ion_ct_expanded30_physics16/benchmark.json` |
| `fig_review_sensitivity` (main text, review-driven sensitivity) | `plot_revision_sensitivity.py` | `results/ct_reviewer_drivers_v6/reconciled_paired_experiments.json`, `results/lung_mesh_ai_v2/synthetic50_threshold_sweep/summary.json` |
| `fig_fem_multistart`, `fig_ct_ratio_calibration`, `fig_structured_tracking` | `generate_new_result_figures.py` | `results/lung_mesh_ai_v2/fem_multistart50_v1/summary.json`, `results/ion_ct_ratio_optimization/fair_calibrate/official_lock.json`, `results/lung_mesh_ai_v2/structured_tracking_v1/mixed/metrics_*.json` |
| `fig_inverse_leaderboard` (main text, post-lock inverse arms and 120-scene official-prior audit) | `plot_inverse_leaderboard.py` | `results/inverse_arm_scoreboard_20260825.json`, `results/ion_ct_ratio_optimization/fair_calibrate/official_lock.json`, `results/ion_ct_official_prior_screened_fem120_v1/records.jsonl` |
| `fig_fair_fem`, `fig_calibration`, `fig_response_baselines`, `fig_multiseed_ood`, `fig_development_ablation`, `fig_aiim_sensitivity`, `fig_ion_ct_method_comparison`, `fig_ion_ct_error_distributions`, `fig_ion_ct_node_material_metrics`, `fig_ion_ct_physics_tradeoff`, `fig_ion_ct_material_recovery`, `fig_ion_ct_synthetic_pipeline` | `generate_aiim_figures.py --only <stem>` | frozen JSON under `results/lung_mesh_ai_v2/`, `results/ion_geometry_ood/formal_validation/`, `results/ion_ct_expanded120_final/`, `results/ion_ct_expanded30_physics16/` and `dataset/ion_ct_synthetic_mechanics540/manifest.json` (the script also loads `dataset/sim_lung_ai_v2_multiview250/manifest.json`) |
| `fig_qualitative_case`, `fig_case_gallery`, `fig_real_ion_cases`, `fig_ion_ct_multiview_tracks`, `fig_ion_ct_physics_material_maps` | `generate_aiim_figures.py --only <stem>` | **not redrawable here**: need the generated `dataset/sim_lung_ai_v2_multiview250/*/gt.pt` tensors, the de-identified CT meshes (`results/ion_geometry_ood/deidentified_meshes27/`) or the de-identified figure assets, none of which are distributed |
| `fig_network_fem` / `meshmaterial_gnn` (MeshMaterialGNN schematic) | `plot_meshmaterial_gnn.py` (also called by `generate_aiim_figures.py --only fig_network_fem`) | needs a PlotNeuralNet checkout (`PLOTNEURALNET_DIR`) and `pdflatex`; the compiled `figures/meshmaterial_gnn.tex` / `.pdf` are shipped |
| `graphical_abstract_array` (Array submission graphical abstract, 2656 x 1062 px; not a manuscript figure; bars show the prior-matched width-1 EKF) | `generate_graphical_abstract.py` (writes into `figures/` in this release) | `results/lung_mesh_ai_v2/node_response_calibrator_strict.json`, `results/ion_ct_official_prior_screened_fem120_v1/summary.json`, `results/ct_reviewer_drivers_v6/ekf_sequential_ct120_all_w1/summary.json`, `results/lung_mesh_ai_v2/ekf_sequential_synthetic50_w1/summary.json` (the script asserts prior width 1 and the scene-wise prior-SD match) |

## Reviewer-revision experiments

Every revision experiment (first-round responses and the second revision of
2026-10-12) is driven by a committed script. The complete command list with
flags and archived output directories is
`experiments/reproduce_reviewer_arms.ps1` (dry run by default; `-Section
<name>` selects a block, `-Execute` runs it; second-revision sections are
`ekf_cohort`, `ct18_yaw48`, `ct18_aspect_fixed`, `decoupled50`,
`ct18_reinference`, `record_audits`). Scene-level `records.jsonl` files, their
`run_manifest.json` protocol files, the shard records and launch records of
the sharded runs, and every per-arm `summary.json` whose digest is quoted are
shipped byte-for-byte, so the `records_sha256` / `run_manifest_sha256` /
`summary_json_sha256` fields quoted in the group summaries can be re-hashed.
Every group summary can be regenerated (eight first-round groups) or verified
field-by-field (eighteen second-revision groups) from the records without
running FEM:

```bash
python experiments/summarize_reviewer_arms.py --check          # all 26 groups
python experiments/run_assumption_sensitivity_ct18.py --summarize-only
python experiments/reconcile_ct_reviewer_evidence.py
```

Locked cohorts: 18 CT-derived test scenes (six held-out geometries x three
scenarios; `results/ct_reviewer_drivers_v6/paired_scene_ids.json`), the 50
procedural synthetic test cases (the complete test split of
`dataset/sim_lung_ai_v2_multiview250/manifest.json`), and the 120 locked CT
test scenes. The 18-scene and six-arm FEM arms below use
`experiments/run_screened_fem_locked.py` with `--prior-width-scale 2.0
--screen-threshold 0.05 --probe-count 2 --max-nfev 48` unless the flag is the
varied factor; the 120-scene official audit and the frozen primary run use
prior-width multiplier 1 (see
`results/ct_reviewer_drivers_v6/ct120_audit_mechanics.json`,
`prior_width_scale_evidence`), so acceptance rates are protocol-specific
and are not comparable across protocols. "Supplement" loci refer to section
titles of the supplementary material; "Main" loci to the manuscript.

| Experiment | Driver / flags | Summary (and records) | Manuscript locus |
| --- | --- | --- | --- |
| Primary 50-case screened MAP vs fixed-init; no-screen counterfactual | `lung_inverse_rendering/evaluate_sim_lung_v2.py` (96 evaluations, `--minimum-refinement-cost-reduction 0.05`); `experiments/run_aiim_sensitivity_sweeps.py --configs noscreen` | `results/lung_mesh_ai_v2/fair_multiview_fem50_strict/fair_multiview_summary.json`; `results/lung_mesh_ai_v2/aiim_sensitivity/sensitivity_noscreen.json`; per-case metrics `results/lung_mesh_ai_v2/fem_multiview_screened50_strict/metrics_*.json` | Main: Matched inverse-FEM comparison; Screen counterfactual. Supplement: Incremental FEM contribution |
| Six-arm 50-case matched controls (init x prior centre, no material prior, residual objective/screen) | `run_screened_fem_locked.py --cohort synthetic50 --initialization-mode {learned,fixed} --prior-center-mode {learned,fixed}`; `--material-prior-weight 0 --prior-width-scale 1`; `--screening-metric residual` | `results/lung_mesh_ai_v2/synthetic50_full_paired_controls_20261008/summary_50patient_paired.json` (+ six `*/records.jsonl`) | Main: Matched initialization and prior controls. Supplement: Matched controls |
| 50-case early-threshold sweep 0 / 0.01 / 0.025 / 0.10 / 0.20 | `--cohort synthetic50 --screen-threshold <t>` (final acceptance stays 0.05) | `results/lung_mesh_ai_v2/synthetic50_threshold_sweep/summary.json` | Main: Continuation threshold, probes, and prior strength. Supplement: Continuation threshold on the 50-case synthetic protocol |
| 18-scene baseline, six matched controls, threshold/probe/width sensitivity (11 arms), material-prior-weight sweep, nine-arm force/yaw mismatch | `experiments/run_ct_reviewer_drivers.py --run --patient-ids-file results/ct_reviewer_drivers_v6/paired_scene_ids.json [--include-controls] --config-id <arm>`; mismatch via `--model-force-bias {-0.20,-0.10,0.05,0.10,0.20}` and `--pose-rotation-error-deg {-10,-5,5,10}` | `results/ct_reviewer_drivers_v6/reconciled_paired_experiments.json` (registry with SHA-256 of every records file) | Main: Review-driven sensitivity and model-mismatch analyses; Fig. `fig_review_sensitivity`. Supplement: Matched controls; Continuation threshold and probe directions; Material-prior strength; Force, camera, and mechanical mismatch |
| Camera-yaw +/-10 deg arms re-run at the common 48-evaluation budget (the archived 16-evaluation arms are kept as reference) | `run_screened_fem_locked.py --cohort ct120 --pose-rotation-error-deg {-10,10} --max-nfev 48 --patient-ids <18 locked ids>` | `results/ct_reviewer_drivers_v6/model_mismatch_n48/summary.json` (+ per-arm `records.jsonl`, `run_manifest.json`, `summary.json`) | Supplement: Force, camera, and mechanical mismatch. Responses R3 C5, R4 C6 |
| 18-scene assumption / proxy arms: Poisson ratio 0.40/0.49, fixation quantile 0.70/0.90, ellipsoid aspect 1.5/2.0, joint +10 % force / +5 deg yaw, 20 % correspondence dropout, 2 % mesh jitter, 15 % contact shift | `experiments/run_assumption_sensitivity_ct18.py` (`--model-nu`, `--posterior-fixation-quantile`, `--region-aspect-xy`, `--tracking-error-model dropout --tracking-dropout-rate`, `--mesh-jitter-fraction`, `--contact-shift-fraction`) | `results/ct_reviewer_drivers_v6/assumption_sensitivity/summary.json` | Main: Assumptions and systematic observation-law mismatch. Supplement: Force, camera, and mechanical mismatch |
| Corrected ellipsoid-aspect arms (region attention always from the predicted centre/radius; the archived arms fell back to the true region) and the occupancy sanity check | `run_screened_fem_locked.py --cohort ct120 --region-aspect-xy {1.5,2.0} --max-nfev 48 --patient-ids <18 locked ids>`; `experiments/aspect_arm_sanity.py` | `results/ct_reviewer_drivers_v6/assumption_sensitivity_fixed/summary.json`; `results/ct_reviewer_drivers_v6/assumption_sensitivity/aspect_arm_sanity.json` | Main: Assumptions and systematic observation-law mismatch. Supplement: Force, camera, and mechanical mismatch. Responses R3 C5, R4 C6/C12 |
| Constitutive mismatch (fiber / viscous observation laws, isotropic fitter) | `experiments/audit_constitutive_mismatch.py --law {fiber,viscous}` (`physics/constitutive_mismatch.py`) | `results/ct_reviewer_drivers_v6/constitutive_mismatch/summary.json` | Main: Assumptions and systematic observation-law mismatch. Supplement: Force, camera, and mechanical mismatch |
| Structural mismatch (two inclusions, diffuse field, sliding boundary, matched control) | `experiments/audit_structural_mismatch.py --mode {matched,two_inclusions,diffuse,sliding_bc}` | `results/ct_reviewer_drivers_v6/structural_mismatch/summary.json` | Main: Assumptions and systematic observation-law mismatch. Supplement: Structural truth mismatch and residual-based detection |
| Learned initializer re-inferred on the mismatched observations of the 18 scenes | `experiments/reinfer_mismatch_initializer.py --shard 0 --shard-count 1 --scratch <dir>`; `--merge` (needs the CT tensors and the training checkpoints) | `results/ct_reviewer_drivers_v6/mismatch_initializer_reinference/summary.json` (+ `records.jsonl`) | Main: Assumptions and systematic observation-law mismatch. Supplement: Learned initializer re-inferred on mismatched observations. Responses R3 C5, R4 C6 |
| Sequential EKF, 18 scenes (0 / 0.05 px observation noise) | `experiments/run_ekf_sequential_ct18.py [--noise-px 0.05 --out results/ct_reviewer_drivers_v6/ekf_sequential_noise0p05]` | `results/ct_reviewer_drivers_v6/ekf_sequential/summary.json`, `.../ekf_sequential_noise0p05/summary.json` | Main: Sequential EKF on the 18 scenes and on both main cohorts. Supplement: Sequential EKF on the locked 18 scenes and on both main cohorts |
| Sequential EKF, cohort level, width-2 sensitivity: all 120 CT scenes (official ratio locked) and the 50 synthetic cases (six-arm lock rule) with material-prior width multiplier 2 (the value hard-coded before `--prior-width-scale` existed; the first cohort runs, now reported only as a sensitivity) | `experiments/run_ekf_sequential_ct18.py --cohort ct120 --all-test --noise-px 0.05 --shard k/4`, then `--summarize`; `--cohort synthetic50 --all-test --noise-px 0.05 --shard k/4`, then `--summarize` | `results/ct_reviewer_drivers_v6/ekf_sequential_ct120_all/summary.json`; `results/lung_mesh_ai_v2/ekf_sequential_synthetic50/summary.json` (+ `records_shard*.jsonl`, `launch_command.txt`; its `superseded_v0_unclipped_initial_ratio/` is the discarded first run, see `WHY_SUPERSEDED.txt`) | Main: Sequential EKF on the 18 scenes and on both main cohorts (sensitivity sentence); Principal finding; Limitations; Conclusion. Supplement: Sequential EKF section, "Width-2 sensitivity" paragraph and table rows. Responses R1 C4, R2 C1, R3 C8, R4 C2/C4/C5 |
| Sequential EKF, cohort level, prior-matched width-1 runs (the PRIMARY EKF statement of the manuscript): the same filter with the material-prior SD matched scene by scene to the CT120 official-prior audit and to the frozen primary run (`matched_protocol_checks.prior_width.prior_sd_all_equal = true`), plus the width-1 vs width-2 side-by-side on identical scenes | `experiments/run_ekf_sequential_ct18.py --cohort ct120 --all-test --noise-px 0.05 --prior-width-scale 1.0 --shard k/4 --out results/ct_reviewer_drivers_v6/ekf_sequential_ct120_all_w1` (k = 0..3), then `--summarize`; `--cohort synthetic50 --all-test --noise-px 0.05 --prior-width-scale 1.0 --shard k/2 --out results/lung_mesh_ai_v2/ekf_sequential_synthetic50_w1` (k = 0..1), then `--summarize` (flags as recorded in the first line of each shipped `run_shard*of*.log`; no `launch_command.txt` was written for these runs) | `results/ct_reviewer_drivers_v6/ekf_sequential_ct120_all_w1/summary.json`; `results/lung_mesh_ai_v2/ekf_sequential_synthetic50_w1/summary.json` (+ `records.jsonl`, `records_shard*.jsonl`, `run_shard*.log`); `results/ct_reviewer_drivers_v6/ekf_width_comparison.json` | Main: Abstract; Sequential EKF on both main cohorts; Principal finding; Limitations; Conclusion (headline values 4.74 % / 2.73 %). Supplement: Sequential EKF section, table rows "Prior-matched EKF (w1; primary)" and the "Width-2 sensitivity" paragraph. Graphical abstract. Cover letter and highlights. Responses R1 C4, R2 C1, R3 C8, R4 C2/C4/C5 |
| Decoupled screening on the 18 scenes (residual screen, MAP objective) | `--screening-metric residual --objective-metric map` | `results/ct_reviewer_drivers_v6/decoupled_screen/summary.json` | Supplement: Matched controls. Response R3 C2 |
| Decoupled residual screen + acceptance on the 50 cases (width 1 / 96 evaluations and width 2 / 48 evaluations; plus the plain MAP arm at width 1 / 96; three shards per arm) | `run_screened_fem_locked.py --cohort synthetic50 --screening-metric residual --objective-metric map --acceptance-metric residual --prior-width-scale {1.0,2.0} --max-nfev {96,48} --patient-start {0,17,34} --patient-limit {17,17,16}` | `results/lung_mesh_ai_v2/synthetic50_decoupled_residual/comparison.json` (+ per-arm `summary.json`, `records.jsonl`, `shard*/records.jsonl`, `shard*/run_manifest.json`, `shard*/command.txt`) | Main: Matched initialization and prior controls. Supplement: Decoupled residual screen and acceptance on the 50 cases. Responses R2 C2, R3 C2/C3, R4 C3 |
| Record-only audits: accepted-update harm, initializer-only reference on the 18 scenes, four-bin calibration test sensitivity, CT 120 audit mechanics (prior-width-1 evidence) | `experiments/accepted_update_harm.py`; the other three are verified field-by-field by `experiments/summarize_reviewer_arms.py --check --groups initializer_reference_18 calibration_fourbin_test_sensitivity ct120_audit_mechanics` from the shipped records, manifests and MeshGNN prediction files | `results/ct_reviewer_drivers_v6/accepted_update_harm.json`, `initializer_reference_18.json`, `calibration_fourbin_test_sensitivity.json`, `ct120_audit_mechanics.json` | Main: Incremental contribution relative to initialization; Development-only geometry-resampling analysis. Supplement: Incremental FEM contribution; Review-driven paired experiments; Complete frozen publication gates; Reproducibility and data provenance. Responses R1 C4/C7, R2 C2, R3 C2/C4, R4 C4/C7 |
| Second-round text facts (CT120 search-box edge counts 73/61/46, calibration pairing 61/59, bin crossings 1/9/0/2/5, identifiability direction, decoupled-vs-primary IDs, accepted-update harm, EKF prior width in the records, width-1 rerun, latexdiff baseline) | one-off script over the shipped records and summaries (not part of the repository); every value is read from files that `summarize_reviewer_arms.py --check` verifies | `results/ct_reviewer_drivers_v6/second_round_facts.json` | Main: Sec. 3.7, 4.4, 5.1, 5.3; Table 7/17 captions. Supplement: Sec. 9.5, 17. Responses R1-R4 (second-round edits) |
| Population-mean prior centre | `--prior-center-mode fixed --fixed-prior-E 4425.1653061175575` | `results/ct_reviewer_drivers_v6/population_prior/summary.json`, `population_prior_E.json` | Supplement: Matched controls. Response R4 C3 |
| Training-only nested calibration resampling (two seeds) and LOO refit | `experiments/audit_nested_calibration_resampling.py --repeats 1000 --heldout-geometries 4 --seed {20261008,20261009}`; `--loo-refit-only` | `results/ct_reviewer_drivers_v6/repeated_outer_audit_seed*_1000.json`, `calibration_stability_loo_refit.json` | Main: Development-only geometry-resampling analysis. Supplement: Training-only calibration resampling. Responses R1 C7, R3 C4, R4 C7 |
| CT 120-scene official-prior screened FEM and leave-one-geometry-out influence | `run_screened_fem_locked.py --cohort ct120 --max-nfev 48 --prior-width-scale 1.0` (the audit used width multiplier 1; the current driver default is 2.0); `experiments/audit_ct_geometry_influence.py` | `results/ion_ct_official_prior_screened_fem120_v1/summary.json`, `records.jsonl`, `geometry_influence_audit.json`; `results/ct_reviewer_drivers_v6/ct120_audit_mechanics.json` | Main: Official calibrated-prior audit and geometry influence. Supplement: Official calibrated-prior CT FEM audit; Fixed-prediction leave-one-geometry-out influence |
| Post-lock inverse-arm controls: locked-ratio screened FEM and unscreened PINN on the 50 cases; PINN on the 120 CT scenes (official and hybrid-initializer starts); validation-selected ratio stacking | `run_screened_fem_locked.py --cohort synthetic50`; `experiments/run_pinn_inverse_arm.py`; `experiments/stack_ct_ratio_calibrator.py`; ratio lock `experiments/calibrate_ct_ratio_fair.py` | `results/lung_mesh_ai_v2/screened_fem_locked50_v1/`, `results/lung_mesh_ai_v2/pinn_inverse50_v1/`, `results/ion_ct_pinn_inverse120_v1/`, `results/ion_ct_pinn_inverse120_officialstart_v1/` (summary, comparison, records), `results/inverse_arm_scoreboard_20260825.json`, `results/ion_ct_ratio_optimization/fair_calibrate/official_lock.json`, `results/ion_ct_ratio_optimization/stack_covariate_v1/selection.json` | Supplement: Additional inverse-arm controls after the official lock; Validation-selected ratio stacking; Fig. `fig_inverse_leaderboard` |
| Seven-parameter local identifiability | `experiments/audit_joint_identifiability_ct.py` | `results/ct_reviewer_drivers_v6/joint_identifiability_audit_v1.json` | Main: Seven-parameter local identifiability. Supplement: Seven-parameter local sensitivity. Response R4 C11 |
| Temporal amplitude rank-one projection audit | `experiments/audit_temporal_amplitude.py` | `results/lung_mesh_ai_v2/temporal_amplitude_audit_test50.json` | Supplement: Held-out FEM comparison. Response R3 C8 |
| VFM true-force oracle ceiling | `experiments/benchmark_vfm_oracle.py --patient-limit 50 --force-field forces` | `results/lung_mesh_ai_v2/vfm_oracle_trueforce_50.json` | Main: Classical full-field comparator. Supplement: Virtual fields method full-field reference comparator. Response R4 C10 |
| Identity / provenance of the 27 CT geometries | `experiments/audit_ion_dataset.py`, `lung_inverse_rendering/prepare_ion_ct_meshes.py --export`, `experiments/build_ion_geometry_split.py`, `experiments/validate_ion_geometry_cohort.py`, `experiments/validate_ion_ct_synthetic_mechanics.py` | requires the restricted source collection; released outputs contain opaque `geom_*` IDs and hashes only (`results/ion_geometry_ood/`, `dataset/ion_ct_synthetic_mechanics540/manifest.json`) | Main: Clinical data provenance and participant scope. Supplement: Reproducibility and data provenance. Responses R3 C7, R4 C14/C15 |

All reported medians are arithmetic (mean of the two central values for even
n); see `REPRODUCIBILITY.md` for the two internal fields that use the
upper-middle element instead. The 18-scene CT pilots lock the official
calibrated ratio prediction, whose median error on that 18-scene subset is
54.39 % (17.77 % over all 120 scenes); the pilots use prior-width multiplier 2
whereas the 120-scene audit uses multiplier 1, so the two must not be pooled.
The screened batch audit does not improve background accuracy on either main
cohort (3.82 -> 4.11 % synthetic, 8.55 -> 11.70 % CT), whereas the all-frame
sequential EKF started from the same initializer does. The primary EKF
statement in the manuscript (abstract, Results, Discussion, Conclusion,
Supplement Sec. 9.5 table, graphical abstract) is the prior-matched run with
material-prior width multiplier 1, the value of the audit and of the primary
run (prior SD equal scene by scene, `prior_sd_all_equal = true`;
`ekf_sequential_*_w1/`): 4.74 % on the 120 CT scenes (audit 11.70 %,
initializer 8.55 %; better than the audit in 100/120 and than the
initializer in 92/120) and 2.73 % on the 50 synthetic cases (primary 4.11 %,
initializer 3.82 %; 31/50 and 30/50). The first EKF runs used width
multiplier 2 (`ekf_sequential_*/`, twice the audit's SD) and gave 3.12 % and
2.68 %; they are reported only as a sensitivity (Supplement Sec. 9.5;
side-by-side in `ekf_width_comparison.json`, width 1 minus width 2: CT
+1.35 pp with width 1 worse in 86/120, synthetic -0.16 pp). The EKF has no
screening or rollback and is not a replacement of the frozen primary
results.

## Citation

See `CITATION.cff`.

## License

Code is released under the MIT License. Numerical evidence and figures are
provided for research reproducibility with attribution. Third-party datasets
and patient-derived source assets are not redistributed.
