
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

## Reviewer-revision arms

`experiments/reproduce_reviewer_arms.ps1` lists the exact command (driver,
flags, locked scene IDs, output directory) of every first-round-revision
experiment; see the "Reviewer-revision experiments" table in `README.md`.
Per-scene `records.jsonl` and `run_manifest.json` files are shipped
byte-for-byte so that the SHA-256 digests quoted in the group summaries and
in the supplement can be re-hashed. Regenerate and verify every group summary
from records without FEM:

```bash
python experiments/summarize_reviewer_arms.py --check
python experiments/run_assumption_sensitivity_ct18.py --summarize-only
python experiments/reconcile_ct_reviewer_evidence.py
python -m pytest experiments/test_constitutive_mismatch.py \
       experiments/test_assumption_summary_statistics.py -q
```
