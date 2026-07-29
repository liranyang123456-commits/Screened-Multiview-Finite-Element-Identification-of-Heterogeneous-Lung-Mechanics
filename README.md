
# Heterogeneous Lung Mechanics from Multiview Motion

Reference implementation and frozen numerical evidence for:

> **Calibrated Response Learning as a Prior for Screened Multiview
> Finite-Element Identification of Heterogeneous Lung Mechanics**

The project estimates background Young's modulus, inclusion contrast, region
center/radius, and node-level material fields from synchronized multiview motion
under known loads. The primary method uses a train-only-selected PCA--Ridge
response calibrator as a prior for screened inverse FEM.

## Evidence boundary

- Material-property truth is synthetic.
- The main benchmark contains 250 programmatic CT-like surrogate geometries,
  split 150/50/50 by simulated patient.
- The external benchmark contains 60 synthetic-mechanics scenes instantiated on
  three de-identified patient-derived CT geometries.
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

1. Generate the programmatic multiview synthetic cohort using
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

## Citation

See `CITATION.cff`.

## License

Code is released under the MIT License. Numerical evidence and figures are
provided for research reproducibility with attribution. Third-party datasets
and patient-derived source assets are not redistributed.
