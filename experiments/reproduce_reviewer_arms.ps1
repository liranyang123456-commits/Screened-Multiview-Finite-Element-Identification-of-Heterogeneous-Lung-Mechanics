<#
.SYNOPSIS
  Exact command list for every reviewer-revision experiment arm of the
  ARRAY-D-26-05373 revision (first-round responses R1-R4).

.DESCRIPTION
  Each entry gives the committed driver script, the exact flags, and the
  archived output directory that the manuscript / supplement / responses cite.
  Arms that were originally launched by ad-hoc PowerShell (assumption proxies,
  structural mismatch, EKF, decoupled screen, population prior, threshold
  sweep) are listed with the same flags that are frozen in their
  run_manifest.json / launch_command.txt files.

  By default the script only PRINTS the commands (dry run). Pass -Execute to
  run them; FEM arms take hours and will REFUSE to overwrite an archived
  directory unless you point --results elsewhere (run_screened_fem_locked.py
  resumes a directory only if the protocol manifest is identical).

  All summary.json files can be regenerated from the archived records without
  running FEM:  -Section summaries

.PARAMETER Execute
  Actually run the commands instead of printing them.

.PARAMETER Section
  One or more section names to print/run. Default: all sections.
  Names: primary50, sixarm50, threshold50, ct18_controls, ct18_sensitivity,
         ct18_weight, ct18_mismatch, ct18_assumption, ct18_constitutive,
         ct18_structural, ct18_ekf, ct18_decoupled, ct18_population,
         calibration, ct120, audits, summaries, tests

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File experiments/reproduce_reviewer_arms.ps1
  powershell -ExecutionPolicy Bypass -File experiments/reproduce_reviewer_arms.ps1 -Section summaries -Execute
#>
[CmdletBinding()]
param(
    [switch]$Execute,
    [string[]]$Section = @()
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

# ---------------------------------------------------------------- locked ID sets
$Ids18File = 'results/ct_reviewer_drivers_v6/paired_scene_ids.json'
$Ids18 = Get-Content $Ids18File -Raw | ConvertFrom-Json
if ($Ids18.Count -ne 18) { throw "$Ids18File must list 18 scene IDs" }
$Ids18Arg = '--patient-ids ' + ($Ids18 -join ' ')
# The 50 synthetic cases are the complete test split of dataset/sim_lung_ai_v2_multiview250
# (lung_patient_004 ... lung_patient_249, step 5); run_screened_fem_locked.py --cohort synthetic50
# selects them automatically, so no explicit --patient-ids is needed.

$Fem = "$Py experiments/run_screened_fem_locked.py"
$Ct18Base = "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.05 --probe-count 2 --max-nfev 48"
$V6 = 'results/ct_reviewer_drivers_v6'
$LM = 'results/lung_mesh_ai_v2'

$Sections = [ordered]@{}

# ---------------------------------------------------------------- primary 50-case screened MAP + no-screen counterfactual
$Sections.primary50 = @(
    "# Primary screened-MAP arm (50 synthetic test cases, 96 evaluations) -> $LM/fair_multiview_fem50_strict/",
    "$Py lung_inverse_rendering/evaluate_sim_lung_v2.py --dataset dataset/sim_lung_ai_v2_multiview250 --results $LM/fair_multiview_fem50_strict --patient-limit 50 --observation image_tracks --multiview-tracks --minimum-track-confidence 0.2 --region-track-weight 8.0 --track-noise-px 0.05 --all-track-frames --temporal-track-regression --track-smoothing-iterations 2 --max-nfev 96 --initial-predictions $LM/node_response_calibrator_strict.json --use-predicted-region --force-prior-sigma 0.05 --material-prior-weight 0.05 --minimum-refinement-cost-reduction 0.05 --benchmark-method calibrated_multiview_screened_map_strict",
    "# Fixed-init pairing + aggregation (writes fair_multiview_summary.json)",
    "$Py experiments/run_fair_multiview_fem.py --dataset dataset/sim_lung_ai_v2_multiview250 --predictions $LM/node_response_calibrator_strict.json --ai-result <metrics_*calibrated_multiview_screened_map_strict*.json from the previous step> --results $LM/fair_multiview_fem50_strict",
    "# No-screen / no-rollback counterfactual (screening 0, acceptance 0) -> $LM/aiim_sensitivity/sensitivity_noscreen.json",
    "$Py experiments/run_aiim_sensitivity_sweeps.py --configs noscreen --results $LM/aiim_sensitivity"
)

# ---------------------------------------------------------------- six-arm 50-patient matched controls
$Six = "$LM/synthetic50_full_paired_controls_20261008"
$SixBase = "$Fem --cohort synthetic50 --prior-width-scale 2.0 --screen-threshold 0.05 --probe-count 2 --max-nfev 48"
$Sections.sixarm50 = @(
    "# Six-arm matched controls, 50 synthetic test cases -> $Six/<arm>/records.jsonl",
    "$SixBase --initialization-mode learned --prior-center-mode learned --results $Six/learned_learned",
    "$SixBase --initialization-mode fixed   --prior-center-mode fixed   --results $Six/fixed_fixed",
    "$SixBase --initialization-mode fixed   --prior-center-mode learned --results $Six/fixed_learned",
    "$SixBase --initialization-mode learned --prior-center-mode fixed   --results $Six/learned_fixed",
    "$Fem --cohort synthetic50 --prior-width-scale 1 --screen-threshold 0.05 --probe-count 2 --max-nfev 48 --material-prior-weight 0 --results $Six/no_material_prior",
    "$SixBase --screening-metric residual --results $Six/residual_only",
    "# Group summary (medians, paired deltas, 20000-replicate bootstrap, seed 20261008) -> $Six/summary_50patient_paired.json",
    "$Py experiments/summarize_reviewer_arms.py --check --groups six_arm_synthetic50"
)

# ---------------------------------------------------------------- 50-case early-threshold sweep
$Sw = "$LM/synthetic50_threshold_sweep"
$Sections.threshold50 = @(
    "# Early continuation threshold sweep (final acceptance stays 0.05) -> $Sw/<arm>/",
    "$Fem --cohort synthetic50 --prior-width-scale 2.0 --probe-count 2 --max-nfev 48 --initialization-mode learned --prior-center-mode learned --screen-threshold 0.0   --results $Sw/s0p00",
    "$Fem --cohort synthetic50 --prior-width-scale 2.0 --probe-count 2 --max-nfev 48 --initialization-mode learned --prior-center-mode learned --screen-threshold 0.01  --results $Sw/s0p01",
    "$Fem --cohort synthetic50 --prior-width-scale 2.0 --probe-count 2 --max-nfev 48 --initialization-mode learned --prior-center-mode learned --screen-threshold 0.025 --results $Sw/s0p025",
    "$Fem --cohort synthetic50 --prior-width-scale 2.0 --probe-count 2 --max-nfev 48 --initialization-mode learned --prior-center-mode learned --screen-threshold 0.1   --results $Sw/s0p10",
    "$Fem --cohort synthetic50 --prior-width-scale 2.0 --probe-count 2 --max-nfev 48 --initialization-mode learned --prior-center-mode learned --screen-threshold 0.2   --results $Sw/s0p20",
    "# threshold 0.05 reference = $Six/learned_learned (see sixarm50)",
    "$Py experiments/summarize_reviewer_arms.py --check --groups synthetic50_threshold_sweep"
)

# ---------------------------------------------------------------- 18-scene CT pilot: baseline + matched controls
$Drv = "$V6/ct_reviewer_drivers_v6"
$Sections.ct18_controls = @(
    "# Baseline w2_s0p05_p2 (learned init / learned prior centre) -> $Drv/fem_paired/w2_s0p05_p2/",
    "$Ct18Base --results $Drv/fem_paired/w2_s0p05_p2 $Ids18Arg",
    "# Six matched controls -> $Drv/paired_controls/<arm>/",
    "$Ct18Base --initialization-mode fixed   --prior-center-mode fixed   --results $Drv/paired_controls/control_fixed_init_fixed_prior $Ids18Arg",
    "$Ct18Base --initialization-mode learned --prior-center-mode fixed   --results $Drv/paired_controls/control_learned_init_fixed_prior $Ids18Arg",
    "$Ct18Base --initialization-mode fixed   --prior-center-mode learned --results $Drv/paired_controls/control_fixed_init_learned_prior $Ids18Arg",
    "$Ct18Base --initialization-mode learned --prior-center-mode learned --results $Drv/paired_controls/control_learned_init_learned_prior $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 1 --screen-threshold 0.05 --probe-count 2 --max-nfev 48 --material-prior-weight 0 --results $Drv/paired_controls/control_no_material_prior $Ids18Arg",
    "$Ct18Base --screening-metric residual --results $Drv/paired_controls/control_residual_only $Ids18Arg",
    "# Equivalent driver form (writes driver_manifest.json per arm):",
    "$Py experiments/run_ct_reviewer_drivers.py --run --patient-ids-file $Ids18File --output-root $Drv --include-controls --config-id w2_s0p05_p2 --config-id control_fixed_init_fixed_prior --config-id control_learned_init_fixed_prior --config-id control_fixed_init_learned_prior --config-id control_learned_init_learned_prior --config-id control_no_material_prior --config-id control_residual_only"
)

# ---------------------------------------------------------------- 18-scene sensitivity (threshold / probes / width), 11 arms incl. baseline
$Sens = 'results/ct_reviewer_drivers_v1/recovered_raw/fem_paired'
$Sections.ct18_sensitivity = @(
    "# One-factor sensitivity arms -> $Sens/<config>/ (baseline w2_s0p05_p2 shared with ct18_controls)",
    "$Fem --cohort ct120 --prior-width-scale 1.0 --screen-threshold 0.05  --probe-count 2 --max-nfev 48 --results $Sens/w1_s0p05_p2 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 3.0 --screen-threshold 0.05  --probe-count 2 --max-nfev 48 --results $Sens/w3_s0p05_p2 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.0   --probe-count 2 --max-nfev 48 --results $Sens/w2_s0_p2 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.01  --probe-count 2 --max-nfev 48 --results $Sens/w2_s0p01_p2 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.025 --probe-count 2 --max-nfev 48 --results $Sens/w2_s0p025_p2 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.1   --probe-count 2 --max-nfev 48 --results $Sens/w2_s0p1_p2 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.2   --probe-count 2 --max-nfev 48 --results $Sens/w2_s0p2_p2 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.05  --probe-count 4 --max-nfev 48 --results $Sens/w2_s0p05_p4 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.05  --probe-count 6 --max-nfev 48 --results $Sens/w2_s0p05_p6 $Ids18Arg",
    "# Equivalent driver form: $Py experiments/run_ct_reviewer_drivers.py --run --patient-ids-file $Ids18File --output-root results/ct_reviewer_drivers_v1/recovered_raw --config-id <config>"
)

# ---------------------------------------------------------------- 18-scene material-prior-weight sweep
$W = "$V6/material_prior_weight_sweep"
$Sections.ct18_weight = @(
    "# Material-prior weight sweep at width 2 -> $W/<arm>/",
    "$Ct18Base --material-prior-weight 0.01  --results $W/w0p01 $Ids18Arg",
    "$Ct18Base --material-prior-weight 0.025 --results $W/w0p025 $Ids18Arg",
    "$Ct18Base --material-prior-weight 0.05  --results $W/w0p05 $Ids18Arg",
    "$Ct18Base --material-prior-weight 0.1   --results $W/w0p1 $Ids18Arg",
    "$Ct18Base --material-prior-weight 0.2   --results $W/w0p2 $Ids18Arg"
)

# ---------------------------------------------------------------- 18-scene nine-arm force / yaw mismatch
$MM = "$V6/model_mismatch_completed"
$Sections.ct18_mismatch = @(
    "# Force-bias arms (48 evaluations) -> $MM/<arm>/ and $Drv/model_mismatch/mismatch_force_m10_baseline/",
    "$Ct18Base --model-force-bias -0.20 --results $MM/force_m20 $Ids18Arg",
    "$Ct18Base --model-force-bias -0.10 --results $Drv/model_mismatch/mismatch_force_m10_baseline $Ids18Arg",
    "$Ct18Base --model-force-bias 0.05  --results $MM/force_p5 $Ids18Arg",
    "$Ct18Base --model-force-bias 0.10  --results $MM/force_p10 $Ids18Arg",
    "$Ct18Base --model-force-bias 0.20  --results $MM/force_p20 $Ids18Arg",
    "# Signed camera-yaw arms. NOTE: the archived arms used in reconciled_paired_experiments.json were run with --max-nfev 16",
    "#   (all 18 scenes reached the 16-evaluation budget); their run_manifest.json records max_nfev=16.",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.05 --probe-count 2 --max-nfev 16 --pose-rotation-error-deg -10 --results $MM/pose_yaw_m10_v2 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.05 --probe-count 2 --max-nfev 16 --pose-rotation-error-deg -5  --results $MM/pose_yaw_m5_v1 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.05 --probe-count 2 --max-nfev 16 --pose-rotation-error-deg 5   --results $MM/pose_yaw_p5_v1 $Ids18Arg",
    "$Fem --cohort ct120 --prior-width-scale 2.0 --screen-threshold 0.05 --probe-count 2 --max-nfev 16 --pose-rotation-error-deg 10  --results $MM/pose_yaw_p10_v1 $Ids18Arg",
    "# 48-evaluation +/-5 deg yaw arms also exist (not used by the reconciled registry): $Drv/model_mismatch/mismatch_pose_m5_baseline, mismatch_pose_p5_baseline",
    "$Ct18Base --pose-rotation-error-deg -5 --results $Drv/model_mismatch/mismatch_pose_m5_baseline $Ids18Arg",
    "$Ct18Base --pose-rotation-error-deg 5  --results $Drv/model_mismatch/mismatch_pose_p5_baseline $Ids18Arg",
    "# Registry of controls / mismatch / sensitivity / weight sweep / resampling with SHA-256 of every records file:",
    "$Py experiments/reconcile_ct_reviewer_evidence.py"
)

# ---------------------------------------------------------------- 18-scene assumption / proxy arms
$AS = "$V6/assumption_sensitivity"
$Sections.ct18_assumption = @(
    "# Assumption-sensitivity and observation/geometry proxies -> $AS/<arm>/ (driver: run_assumption_sensitivity_ct18.py)",
    "$Ct18Base --model-nu 0.40 --results $AS/nu_0p40 $Ids18Arg",
    "$Ct18Base --model-nu 0.49 --results $AS/nu_0p49 $Ids18Arg",
    "$Ct18Base --posterior-fixation-quantile 0.70 --results $AS/fix_q0p70 $Ids18Arg",
    "$Ct18Base --posterior-fixation-quantile 0.90 --results $AS/fix_q0p90 $Ids18Arg",
    "$Ct18Base --region-aspect-xy 1.5 --results $AS/aspect_1p5 $Ids18Arg",
    "$Ct18Base --region-aspect-xy 2.0 --results $AS/aspect_2p0 $Ids18Arg",
    "$Ct18Base --model-force-bias 0.10 --pose-rotation-error-deg 5.0 --results $AS/joint_force_p10_yaw_p5 $Ids18Arg",
    "$Ct18Base --tracking-error-model dropout --tracking-dropout-rate 0.20 --results $AS/corr_dropout_0p20 $Ids18Arg",
    "$Ct18Base --mesh-jitter-fraction 0.02 --results $AS/mesh_jitter_0p02 $Ids18Arg",
    "$Ct18Base --contact-shift-fraction 0.15 --results $AS/contact_shift_0p15 $Ids18Arg",
    "# Same ten arms through the committed driver (prints identical commands with --dry-run):",
    "$Py experiments/run_assumption_sensitivity_ct18.py --dry-run",
    "# Group summary -> $AS/summary.json",
    "$Py experiments/run_assumption_sensitivity_ct18.py --summarize-only"
)

# ---------------------------------------------------------------- constitutive mismatch (fiber / viscous observation laws)
$Sections.ct18_constitutive = @(
    "# Observations regenerated with physics/constitutive_mismatch.py; isotropic fitter -> $V6/constitutive_mismatch/<law>/records.jsonl",
    "$Py experiments/audit_constitutive_mismatch.py --law fiber",
    "$Py experiments/audit_constitutive_mismatch.py --law viscous",
    "$Py experiments/summarize_reviewer_arms.py --check --groups constitutive_mismatch"
)

# ---------------------------------------------------------------- structural mismatch
$Sections.ct18_structural = @(
    "# Truth-structure mismatch with the production solver -> $V6/structural_mismatch/<mode>/records.jsonl",
    "$Py experiments/audit_structural_mismatch.py --mode matched",
    "$Py experiments/audit_structural_mismatch.py --mode two_inclusions",
    "$Py experiments/audit_structural_mismatch.py --mode diffuse",
    "$Py experiments/audit_structural_mismatch.py --mode sliding_bc",
    "$Py experiments/summarize_reviewer_arms.py --check --groups structural_mismatch"
)

# ---------------------------------------------------------------- EKF sequential diagnostic
$Sections.ct18_ekf = @(
    "# Sequential EKF over (log E_bg, log q), ratio/region locked -> $V6/ekf_sequential/ and $V6/ekf_sequential_noise0p05/",
    "$Py experiments/run_ekf_sequential_ct18.py",
    "$Py experiments/run_ekf_sequential_ct18.py --noise-px 0.05 --out $V6/ekf_sequential_noise0p05",
    "$Py experiments/summarize_reviewer_arms.py --check --groups ekf_sequential ekf_sequential_noise0p05"
)

# ---------------------------------------------------------------- decoupled screening (R3 C2)
$Sections.ct18_decoupled = @(
    "# Residual screen + MAP objective -> $V6/decoupled_screen/residual_screen_map_objective/",
    "$Ct18Base --screening-metric residual --objective-metric map --results $V6/decoupled_screen/residual_screen_map_objective $Ids18Arg",
    "$Py experiments/summarize_reviewer_arms.py --check --groups decoupled_screen"
)

# ---------------------------------------------------------------- population-prior centre (R4 C3)
$Sections.ct18_population = @(
    "# Prior centre = training-split mean of true E_background (4425.1653061175575 Pa; see $V6/population_prior/population_prior_E.json)",
    "$Py -c `"import json,statistics; m=json.load(open('dataset/ion_ct_synthetic_mechanics540/manifest.json')); v=[float(r['E_background']) for r in m['patients'] if r['split']=='train']; print(len(v), statistics.fmean(v), statistics.median(v))`"",
    "$Ct18Base --initialization-mode learned --prior-center-mode fixed --fixed-prior-E 4425.1653061175575 --results $V6/population_prior/learned_init_population_center $Ids18Arg",
    "$Py experiments/summarize_reviewer_arms.py --check --groups population_prior"
)

# ---------------------------------------------------------------- calibration resampling (training geometries only)
$Sections.calibration = @(
    "# Two independent 1000-repeat nested 14/4 outer resamplings -> $V6/repeated_outer_audit_seed*_1000.json",
    "$Py experiments/audit_nested_calibration_resampling.py --repeats 1000 --heldout-geometries 4 --seed 20261008 --output $V6/repeated_outer_audit_seed20261008_1000.json",
    "$Py experiments/audit_nested_calibration_resampling.py --repeats 1000 --heldout-geometries 4 --seed 20261009 --output $V6/repeated_outer_audit_seed20261009_1000.json",
    "# Deterministic leave-one-training-geometry-out refit (run_ct_reviewer_drivers.calibration_stability) -> $V6/calibration_stability_loo_refit.json",
    "$Py experiments/audit_nested_calibration_resampling.py --loo-refit-only --output $V6/calibration_stability_loo_refit.json"
)

# ---------------------------------------------------------------- CT120 official-prior screened FEM + geometry influence
$Sections.ct120 = @(
    "# 120 locked CT test scenes, official calibrated ratio and predicted region fixed -> results/ion_ct_official_prior_screened_fem120_v1/",
    "$Fem --cohort ct120 --max-nfev 48",
    "# Fixed-prediction leave-one-geometry-out influence (no refit) -> results/ion_ct_official_prior_screened_fem120_v1/geometry_influence_audit.json",
    "$Py experiments/audit_ct_geometry_influence.py"
)

# ---------------------------------------------------------------- other audits
$Sections.audits = @(
    "# Seven-parameter local identifiability (18 scenes) -> $V6/joint_identifiability_audit_v1.json",
    "$Py experiments/audit_joint_identifiability_ct.py",
    "# Temporal amplitude rank-one projection audit (50 synthetic test cases) -> $LM/temporal_amplitude_audit_test50.json",
    "$Py experiments/audit_temporal_amplitude.py",
    "# VFM true-force oracle ceiling (50 synthetic test cases) -> $LM/vfm_oracle_trueforce_50.json",
    "$Py experiments/benchmark_vfm_oracle.py --patient-limit 50 --force-field forces --out $LM/vfm_oracle_trueforce_50.json --seed-bootstrap 20261007 --bootstrap-replicates 5000",
    "# Identity / provenance chain (private source collection required for stages 1-2; outputs contain only opaque geom_* ids and hashes)",
    "$Py experiments/audit_ion_dataset.py --root <private de-identified source root>",
    "$Py lung_inverse_rendering/prepare_ion_ct_meshes.py --export",
    "$Py experiments/build_ion_geometry_split.py",
    "$Py experiments/validate_ion_geometry_cohort.py",
    "$Py experiments/validate_ion_ct_synthetic_mechanics.py"
)

# ---------------------------------------------------------------- summaries only (no FEM)
$Sections.summaries = @(
    "$Py experiments/run_assumption_sensitivity_ct18.py --summarize-only",
    "$Py experiments/summarize_reviewer_arms.py --check",
    "$Py experiments/reconcile_ct_reviewer_evidence.py"
)

# ---------------------------------------------------------------- fast tests
$Sections.tests = @(
    "$Py -m pytest -q lung_inverse_rendering/test_sim_lung_v2.py lung_inverse_rendering/test_pinn_inverse.py experiments/test_generate_ion_ct_synthetic_mechanics.py experiments/test_assumption_summary_statistics.py experiments/test_solver_convergence.py",
    "$Py experiments/run_screened_fem_locked.py --help",
    "$Py experiments/run_assumption_sensitivity_ct18.py --help",
    "$Py experiments/summarize_reviewer_arms.py --help"
)

# ---------------------------------------------------------------- dispatch
$Selected = if ($Section.Count -eq 0) { @($Sections.Keys) } else { @($Section | ForEach-Object { $_ -split '[,;\s]+' } | Where-Object { $_ }) }
foreach ($name in $Selected) {
    if (-not $Sections.Contains($name)) { throw "unknown section '$name'; known: $($Sections.Keys -join ', ')" }
    Write-Host ""
    Write-Host "=== [$name] ===" -ForegroundColor Cyan
    foreach ($line in $Sections[$name]) {
        if ($line.StartsWith('#')) { Write-Host $line -ForegroundColor DarkGray; continue }
        Write-Host $line
        if ($Execute) {
            if ($line -match '<[^>]+>') { Write-Warning 'placeholder argument; skipped'; continue }
            Invoke-Expression $line
            if ($LASTEXITCODE -ne 0) { throw "command failed with exit code ${LASTEXITCODE}: $line" }
        }
    }
}
