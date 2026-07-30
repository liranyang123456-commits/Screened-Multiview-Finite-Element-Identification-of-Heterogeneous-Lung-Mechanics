
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
