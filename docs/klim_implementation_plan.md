# KLIM Implementation Plan

No implementation is authorized by this document. Blocking mathematical decisions are listed in [klim_algorithm_spec.md](klim_algorithm_spec.md#ambiguities-and-decisions).

## Proposed package

```text
adbench/baseline/KLIM/
|-- __init__.py
|-- run.py
|-- model.py
|-- anchor_mapping.py
|-- scatter.py
|-- solver.py
`-- scoring.py
```

## Module contracts

| File | Responsibility and public surface | Inputs -> outputs | Dependencies | Primary tests |
|---|---|---|---|---|
| `__init__.py` | Minimal package export; no heavy side effects. | imports -> selected public names | none | import test |
| `anchor_mapping.py` | `fit_anchors`, `compute_similarity`, `normalize_similarity`, `transform_anchor_space`. | `(n,d), params -> U:(m,d), Z:(n,m)` | NumPy, scikit-learn MiniBatchKMeans | shape, row sums, determinism, no refit |
| `scatter.py` | Construct `H`, source-defined graph terms, `S_wreg`, and `S_T`. | `Z:(n,m), alpha -> (m,m)` matrices | NumPy/SciPy | dimensions, symmetry, PSD, alpha boundaries |
| `solver.py` | Source-defined range SVD and null eigensolver. | two `(m,m)` scatters -> `V:(m,p)`, spectrum, diagnostics | NumPy/SciPy | rank, singularity, subspace determinism |
| `scoring.py` | Assign subclasses, build centroids, and score batches. | `Z,V,B -> assignments, B, scores` | NumPy | empty clusters, non-negative scores |
| `model.py` | `KLIMModel`; validate, orchestrate mathematical stages, store fitted state. | `fit(X_normal)->self`; `decision_function(X_test)->(n_t,)` | modules above | standalone end-to-end tests |
| `run.py` | Thin `KLIM` ADBench adapter. | `fit(X_train,y_train)->self`; `predict_score(X_test)->(n_t,)` | `KLIMModel` | detector-contract and pipeline smoke tests |

## Public APIs

### Standalone model

```text
KLIMModel.fit(X_normal) -> self
KLIMModel.decision_function(X_test) -> scores
```

The model owns validation, optional standalone scaling, anchors, mapping, scatter construction, projection, subclasses, centroids, and scores. It does not accept evaluation labels.

### ADBench adapter

```text
KLIM.__init__(seed, model_name=None, ...)
KLIM.fit(X_train, y_train) -> self
KLIM.predict_score(X_test) -> scores
```

First-version proposal: `X_normal = X_train[y_train == 0]`. This uses `y_train` only to select rows presented as normal; `y_test` remains exclusively in ADBench evaluation. It is a novelty-detection protocol when zero labels are trusted. In ADBench's unsupervised setting, zero also represents hidden anomalies, so it becomes contaminated one-class training. The intended category and policy remain blocking decisions.

## Implementation stages

| Stage | Files changed | Tests added | Completion criterion | Main risks | Rollback boundary |
|---|---|---|---|---|---|
| 1. Anchor mapping | package init, `anchor_mapping.py` | validation and anchor-mapping unit tests | Deterministic finite `Z`, rows sum to one, transform never refits | sigma underflow, duplicate anchors | Remove isolated mapping module |
| 2. Scatter construction | `scatter.py` | exact small-matrix reference tests | Source-approved formulas, symmetric finite scatters | unresolved Laplacian and `O(n^2)` memory | Revert scatter module only |
| 3. Eigen solver | `solver.py` | range/nullity and sign-invariant tests | Finite `V` with verified range and null residuals | tolerance, repeated eigenvalues, zero nullity | Revert solver only |
| 4. Subclass scoring | `scoring.py` | centroid and score tests | Valid centroids; finite non-negative `(n_t,)` scores | empty classes, batch memory | Revert scoring only |
| 5. Standalone model | `model.py` | synthetic end-to-end tests | Fit/predict state contract and reproducibility pass | double scaling, invalid fitted state | Remove model orchestration |
| 6. ADBench adapter | `run.py` | adapter-contract tests | Constructor, `fit`, and `predict_score` match ADBench | label semantics and registration category | Remove adapter; standalone remains |
| 7. Pipeline smoke test | authorized integration point plus tests | one detector, one dataset, three seeds | finite scores/metrics/times and CSVs | central registry behavior, eager imports | Remove only registration/integration change |

## Sequencing gates

1. Resolve all blocking formula decisions before Stage 2.
2. Freeze one mathematical reference case with hand-computed small matrices.
3. Keep the standalone numerical model independent of ADBench.
4. Integrate only after the standalone test suite passes.
5. Do not modify `DataGenerator`, `Utils.metric`, metric definitions, timing boundaries, or CSV schemas.

