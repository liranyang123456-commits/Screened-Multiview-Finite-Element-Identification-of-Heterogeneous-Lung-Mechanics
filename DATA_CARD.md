
# Data Card

## Programmatic synthetic cohort

- 250 independent programmatic CT-like lung-wall surrogate geometries.
- Patient-level split: 150 training, 50 validation, 50 test.
- Four loads, three calibrated views, seven synchronized frames.
- Synthetic heterogeneous Neo-Hookean mechanics and known force.

The generated tensor cohort is excluded because of size; its code, schema,
manifests, and frozen numerical outputs are included.

## Real CT geometry with synthetic mechanics

- Three de-identified CT-derived geometries.
- Twenty matched material/load templates per geometry (60 scenes).
- Four loads per scene (240 load experiments).
- Synthetic material fields, forces, deformation, and accuracy truth.

Only the sanitized manifest, protocol/QC reports, and numerical benchmark are
included. Raw CT, DICOM metadata, patient-derived pixels, and meshes are not
distributed. Twenty scenarios on one geometry are repeated mechanical trials,
not twenty independent patients.

## Intended use

Reproducibility, inverse-mechanics method development, and audit of the reported
synthetic-mechanics claims.

## Prohibited interpretation

The data do not establish patient-specific modulus accuracy, diagnostic
performance, clinical utility, or population-level anatomical generalization.
