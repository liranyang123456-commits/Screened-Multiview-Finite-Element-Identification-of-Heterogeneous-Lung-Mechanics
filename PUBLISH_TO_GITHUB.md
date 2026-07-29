
# Publishing to GitHub

The folder is already self-contained. Review `README.md`, then either upload it
through GitHub or use Git:

```bash
cd heterogeneous-lung-mechanics
git init
git add .
git commit -m "Release reproducibility package"
git branch -M main
git remote add origin https://github.com/<account>/<repository>.git
git push -u origin main
```

Before pushing:

```bash
python scripts/verify_release.py
pytest experiments/test_benchmark_ion_ct_synthetic_mechanics.py \
       experiments/test_export_ion_ct_figure_assets.py \
       experiments/test_generate_ion_ct_synthetic_mechanics.py \
       lung_inverse_rendering/test_lung_pipeline.py \
       evaluation/test_material_uncertainty_metrics.py \
       evaluation/test_aggregate_multiseed.py -q
```

After pushing, replace the pending repository-URL wording in the manuscript and
data-availability statement with the final HTTPS URL. Do not add raw clinical
data, patient-derived CT pixels/meshes, generated tensor cohorts, or checkpoints.
