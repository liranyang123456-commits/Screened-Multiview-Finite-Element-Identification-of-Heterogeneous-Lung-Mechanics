
# GitHub Repository and Release Updates

The reproducibility package is published at:

https://github.com/liranyang123456-commits/Screened-Multiview-Finite-Element-Identification-of-Heterogeneous-Lung-Mechanics

Before updating the public repository, run:

```bash
python scripts/verify_release.py
pytest experiments/test_benchmark_ion_ct_synthetic_mechanics.py \
       experiments/test_export_ion_ct_figure_assets.py \
       experiments/test_generate_ion_ct_synthetic_mechanics.py \
       lung_inverse_rendering/test_lung_pipeline.py \
       evaluation/test_material_uncertainty_metrics.py \
       evaluation/test_aggregate_multiseed.py -q
```

Commit only the verified privacy-safe package and push it to the `main` branch.
Do not add raw clinical data, patient-derived CT pixels/meshes, generated tensor
cohorts, or checkpoints. The final HTTPS URL is already included in the
manuscript and data-availability statement.
