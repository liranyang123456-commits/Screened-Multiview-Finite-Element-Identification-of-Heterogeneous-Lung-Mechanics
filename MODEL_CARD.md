
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

The primary model is accurate on the programmatic synthetic distribution but
does not transfer reliably to the three CT-derived geometries. Measured-force
FEM recovers global background stiffness on those synthetic-mechanics scenes,
while material partition and contrast remain unresolved. These limitations are
part of the frozen evidence rather than omitted failure cases.
