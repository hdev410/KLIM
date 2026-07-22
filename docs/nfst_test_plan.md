# NFST Test Plan

Tests follow [nfst_algorithm_spec.md](nfst_algorithm_spec.md) and compare eigenspaces rather than raw eigenvector signs.

## Unit tests

- Interface: public imports, stored constructor values, clear validation errors, adapter signatures, and seed propagation.
- Anchor mapping: finite 2-D inputs, deterministic anchors, stable Gaussian normalization, row sums, stored-sigma reuse, and no prediction refit.
- Scatter: hand-computed `W`, `D`, `L`, `S_T`, and `S_wreg`; symmetry, PSD tolerance, alpha boundaries, and dense-memory rejection.
- Solver: rank-deficient `S_T`, known reduced nullity, scale-aware thresholds, zero-rank and zero-nullity errors, null-mode caps, explicit smallest-eigen selection, required component count, residuals, and subspace equivalence.
- Scoring: deterministic argmax assignments, projected subclass means, empty subclass exclusion, exact nearest-centroid squared distances, batching, and invalid/non-finite input rejection.

## Standalone tests

- Fit and score controlled one-mode and two-mode normal datasets.
- Verify finite non-negative `(n_test,)` scores and qualitatively larger scores for constructed anomalies.
- Verify same-seed reproducibility.
- Verify prediction before fit fails.
- Verify a failed refit clears all learned state.

## Adapter and isolation tests

- Verify `fit(X_train, y_train)` delegates all training rows and does not interpret masked labels.
- Verify `predict_score(X_test)` delegates without refitting.
- Verify package imports have no TensorFlow, Torch, Keras, or PyOD dependency.
- Verify no ADBench core file is modified and no registration is added.

## Current targeted command

```powershell
& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' -m unittest discover -s tests -p 'test_nfst*.py' -v
```

ADBench integration and multi-dataset benchmark tests remain out of scope until registration is explicitly authorized.
