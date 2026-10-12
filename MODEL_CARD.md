
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
- True-region/true-force reference, reported only as an undeployable, information-advantaged comparator.
- Sequential extended Kalman filter over the material state (no screening or rollback), reported at cohort level.

## Limitations

The primary model is accurate on the procedural synthetic distribution. After
independent-geometry training, the calibrated hybrid initializer reaches
8.55 / 17.77 / 9.60 / 11.46 % median errors on six held-out CT-derived
geometries (raw MeshGNN ratio error 26.95 %); the 25 % ratio gate is a
development criterion, not a clinical tolerance. The official-prior screened
FEM audit raised background error from 8.55 % to 11.70 % with the ratio fixed,
and screening did not detect constitutive or structural mismatch. All
material-accuracy targets are synthetic. These limitations are part of the
frozen evidence rather than omitted failure cases.
