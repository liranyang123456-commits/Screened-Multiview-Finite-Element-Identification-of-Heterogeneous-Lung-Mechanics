
# Model Card

## Primary pipeline

The primary estimator is a PCA--Ridge response calibrator selected using only
the synthetic training cohort. Validation residuals calibrate marginal
conformal intervals and material-prior scales. Screened inverse FEM either
retains the prior or performs bounded physical refinement.

## Secondary models

- Ridge, PLS, and ExtraTrees common-input response baselines.
- MeshMaterialGNN with temporal/view/load attention and node material heads.
- Fixed and deterministic-multistart inverse FEM.
- True-region/true-force oracle, reported only as an undeployable upper bound.

## Limitations

The primary model is accurate on the procedural synthetic distribution. After
independent-geometry training, the expanded 27-geometry cohort meets
background, center, and radius criteria on six held-out geometries, while
inclusion-ratio error remains above criterion. Measured-force FEM improves
global stiffness but not material contrast. These limitations are part of the
frozen evidence rather than omitted failure cases.
