
# Figure scripts

These are the development scripts that drew every figure of the manuscript
and supplement, copied unchanged except for five release-specific edits:
`plot_meshmaterial_gnn.py` reads the PlotNeuralNet checkout from the
`PLOTNEURALNET_DIR` environment variable (and a comment in
`generate_aiim_figures.py` names that variable instead of a local path), and
the two `plot_revision_*.py` scripts and `generate_graphical_abstract.py` write
into the repository's `figures/` folder instead of the manuscript source tree. Each script
resolves the repository root from its own location
(`Path(__file__).resolve().parents[2]`), so run them from a checkout:

```bash
# every figure that is drawn from the shipped frozen JSON and manifests
python paper_aiim/figures/generate_aiim_figures.py --output-dir figures \
    --only fig_pipeline fig_dataset_protocol fig_main_results fig_response_baselines \
           fig_calibration fig_fair_fem fig_aiim_sensitivity fig_development_ablation \
           fig_multiseed_ood fig_ion_ct_error_distributions fig_ion_ct_physics_tradeoff \
           fig_ion_ct_node_material_metrics fig_ion_ct_method_comparison
python paper_aiim/figures/plot_inverse_leaderboard.py --output-dir figures
python paper_aiim/figures/generate_new_result_figures.py --output-dir figures \
    --only fig_fem_multistart fig_structured_tracking fig_ct_ratio_calibration
python paper_aiim/figures/plot_revision_sensitivity.py                 # fig_review_sensitivity (main Fig. 22)
python paper_aiim/figures/plot_meshmaterial_gnn.py                     # meshmaterial_gnn / fig_network_fem (main Fig. 3; PlotNeuralNet + pdflatex)
python paper_aiim/figures/generate_graphical_abstract.py               # graphical_abstract_array (2656 x 1409 px; Figure 1 stages + width-1 EKF bars)
python paper_aiim/figures/plot_revision_overview.py                    # auxiliary *_simplified drawings only (not compiled)
```

`generate_aiim_figures.py` loads `dataset/sim_lung_ai_v2_multiview250/manifest.json`
and the frozen `results/` JSON files; the stems `fig_qualitative_case`,
`fig_case_gallery`, `fig_real_ion_cases`, `fig_ion_ct_multiview_tracks`,
`fig_ion_ct_synthetic_pipeline`, `fig_ion_ct_material_recovery` and
`fig_ion_ct_physics_material_maps` additionally need the generated tensor
cohort, the per-scene CT experiment tensors, the de-identified CT meshes or
the de-identified figure assets, which are not distributed, and will stop
with a `FileNotFoundError` naming the missing asset (the compiled PDFs of
these figures are shipped in `figures/`). `fig_network_fem` needs
PlotNeuralNet and `pdflatex`. The README table lists stem -> script ->
inputs for every figure.
