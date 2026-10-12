
# Reproducibility

The repository separates:

1. source code and tests;
2. generated datasets/checkpoints (not versioned);
3. frozen numerical evidence under `results/`;
4. privacy-safe quantitative figures under `figures/`.

All reported external CT benchmark methods are tagged as `common_input`,
`secondary`, or `oracle` in `benchmark.json`. Do not pool these evidence tiers.
Geometry-cluster bootstrap intervals are descriptive because the locked test
set contains six independent CT geometries.

Run:

```bash
python scripts/verify_release.py
python scripts/summarize_ct_benchmark.py
```

The verification script rejects raw-data/checkpoint extensions, known local
source paths, unreadable JSON, and oversized files.

## Software environment of the reported numbers

All revision experiments were run on one Windows 11 workstation with

- Python 3.13.9 (Anaconda build), PyTorch 2.11.0+cu128 (FEM, EKF and the
  learned initializer re-inference were run on CPU; CUDA was available but
  only the MeshGNN training used it), NumPy 2.2.6, SciPy 1.16.3,
  scikit-learn 1.7.2, Matplotlib 3.7.2;
- `OMP_NUM_THREADS=3` / `MKL_NUM_THREADS=3` per process (the launch records
  shipped next to the sharded runs, `launch_command.txt` and
  `shard*/command.txt`, repeat this), with up to four processes in parallel.

`requirements.txt` pins only lower bounds; the GitHub Actions smoke workflow
runs on Python 3.11 / Linux with the current PyPI releases. Two frozen
metrics files keep their original 160-180-character names
(`results/lung_mesh_ai_v2/fem_multiview_screened50_strict/metrics_*.json`,
`results/lung_mesh_ai_v2/structured_tracking_v1/mixed/metrics_*.json`)
because their paths and digests are quoted in `comparison.json` and in the
figure scripts; on Windows clone with `git config --global core.longpaths
true` into a short directory (or enable the `LongPathsEnabled` policy),
otherwise `summarize_reviewer_arms.py --check` skips the frozen-primary
reference block it cannot open. Floating-point
results of the FEM search are deterministic for a given BLAS build but may
differ in the last digits across platforms; the shipped `records.jsonl`
files are therefore the reference, and `summarize_reviewer_arms.py --check`
reproduces every quoted summary statistic from them exactly.

## Expected runtimes (wall clock, per process at 3 threads)

| Run | Scenes | Median time per scene | Total |
| --- | --- | --- | --- |
| 18-scene CT pilot arm, 48 evaluations (`run_screened_fem_locked.py --cohort ct120 --patient-ids ...`) | 18 | 2-3 min (`wall_time_seconds` in the records) | 40-60 min |
| 50-case synthetic arm, 48 evaluations (six-arm / threshold sweep) | 50 | 1-2.5 min | 1-2 h |
| 50-case synthetic arm, 96 evaluations, residual acceptance (decoupled arms; three shards) | 50 | ~1 min (most cases stop after the 5-7 screening evaluations) | 20-50 min with three shards |
| 120-scene official-prior audit | 120 | ~2 min | ~4 h |
| Cohort-level EKF, 120 CT scenes (100 FEM solves per scene; identical for the width-1 rerun) | 120 | ~4 min | ~2 h with four shards |
| Cohort-level EKF, 50 synthetic cases (140 FEM solves per case; identical for the width-1 rerun, two shards) | 50 | ~5 min | ~1 h with four shards (~1.5 h with two) |
| Learned initializer re-inference (CPU, 11 condition x variant blocks) | 18 x 11 | seconds | ~30 min |
| `summarize_reviewer_arms.py --check` (26 groups) | - | - | < 15 s |
| Fast pytest set (`pytest -q` on the files listed in the workflow) | - | - | 1-2 min |

Constitutive / structural mismatch audits and the nested calibration
resampling (1000 repeats) each take 0.5-2 h.

## Median convention

Every endpoint reported in the manuscript, supplement, responses and the
group summaries is an arithmetic median: the mean of the two central values
for even sample sizes (18 scenes, 50 cases, 120 scenes). Two internal fields
written by older helpers use the upper-middle element instead
(`sorted(values)[n // 2]`) and are reproduced verbatim by the checker so that
the archived files verify: the inner leave-one-geometry-out selection
statistic `inner_loo_median` in `calibration_fourbin_test_sensitivity.json`
(and in `run_ct_reviewer_drivers.py --only-calibration`), and the four
descriptive `median_*` / `wall_time_median_*` values of
`aspect_arm_sanity.json`. Neither is quoted as a result in the manuscript.
Percentages in the group summaries are rounded to two (first-round groups)
or four decimals (second-revision groups); the raw fractions are in the
records.

## Reviewer-revision arms

`experiments/reproduce_reviewer_arms.ps1` lists the exact command (driver,
flags, locked scene IDs, output directory) of every revision experiment,
including the second-revision sections `ekf_cohort`, `ct18_yaw48`,
`ct18_aspect_fixed`, `decoupled50`, `ct18_reinference` and `record_audits`;
see the "Reviewer-revision experiments" table in `README.md`. Per-scene
`records.jsonl`, `run_manifest.json`, shard records, launch records and the
per-arm `summary.json` files whose digests are quoted are shipped
byte-for-byte so that every SHA-256 in the group summaries and in the
supplement can be re-hashed. Regenerate (first-round groups) and verify
field-by-field (second-revision groups) every group summary from records
without FEM:

```bash
python experiments/summarize_reviewer_arms.py --check
python experiments/run_assumption_sensitivity_ct18.py --summarize-only
python experiments/reconcile_ct_reviewer_evidence.py
python -m pytest experiments/test_constitutive_mismatch.py \
       experiments/test_assumption_summary_statistics.py -q
```

The checker prints, for every second-revision group, how many values were
recomputed and which fields are not recomputable from the shipped files
(occupancy node sets that need the mesh tensors, prose conclusions,
checkpoint digests, file time stamps).

## Figures

See the "Figures" section of `README.md` and `paper_aiim/figures/README.md`
for the script that draws each figure and the inputs it needs.
