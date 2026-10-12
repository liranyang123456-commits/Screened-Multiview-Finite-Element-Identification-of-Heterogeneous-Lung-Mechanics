
# Figure scripts

These are the development scripts that drew every figure of the manuscript
and supplement, copied unchanged except for three release-specific edits:
`plot_meshmaterial_gnn.py` reads the PlotNeuralNet checkout from the
`PLOTNEURALNET_DIR` environment variable, and the two `plot_revision_*.py`
scripts write into the repository's `figures/` folder instead of the
manuscript source tree. Each script resolves the repository root from its
own location (`Path(__file__).resolve().parents[2]`), so run them from a
checkout:

```bash
python paper_aiim/figures/plot_revision_overview.py                    # fig_pipeline, fig_dataset_protocol, fig_main_results
python paper_aiim/figures/plot_revision_sensitivity.py                 # fig_review_sensitivity
python paper_aiim/figures/generate_new_result_figures.py --output-dir figures
python paper_aiim/figures/plot_inverse_leaderboard.py --output-dir figures
python paper_aiim/figures/generate_aiim_figures.py --output-dir figures --only fig_fair_fem fig_calibration
```

`generate_aiim_figures.py` loads `dataset/sim_lung_ai_v2_multiview250/manifest.json`
and the frozen `results/` JSON files; the stems `fig_qualitative_case`,
`fig_case_gallery`, `fig_real_ion_cases`, `fig_ion_ct_multiview_tracks` and
`fig_ion_ct_physics_material_maps` additionally need the generated tensor
cohort or the de-identified CT meshes, which are not distributed, and will
stop with a `FileNotFoundError` naming the missing asset. `fig_network_fem`
needs PlotNeuralNet and `pdflatex`. The README table lists stem -> script ->
inputs for every figure.
