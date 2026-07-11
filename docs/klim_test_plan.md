# KLIM Test Plan

Tests must be written against the formula decision record in [klim_algorithm_spec.md](klim_algorithm_spec.md). Formula-dependent expected values cannot be finalized while blocking decisions remain open.

## Unit tests - input validation

- Reject non-2D, scalar, empty-row, and zero-feature inputs.
- Reject NaN, positive infinity, and negative infinity.
- Reject prediction with a feature count different from `n_features_in_`.
- Reject invalid `n_anchors`, `sigma`, `alpha`, `null_dim`, `gamma`, batch size, and tolerance.
- Reject fitting with no selected training rows.
- Reject prediction before fit.

## Unit tests - anchor mapping

- Assert `anchors_.shape == (m,d)` and `Z.shape == (n,m)`.
- Assert every entry is finite and non-negative.
- Assert row sums equal one within a scale-aware tolerance.
- Assert identical seed and input yield identical anchors/memberships within tolerance.
- Assert transform leaves anchors, sigma, and fitted-state fingerprints unchanged.
- Test zero distance, extreme distance, underflow protection, and valid positive sigma.
- Reject `m > n`; test duplicate/degenerate samples and diagnostics.

## Unit tests - scatter matrices

- Assert `Psi_G` and `Psi_L` are `(m,n)` under the PDF formula.
- Assert PDF concatenated `Psi` is `(m,2n)`.
- Assert `S_wreg` and `S_T` are finite `(m,m)` and symmetric.
- Assert `Psi @ Psi.T` is PSD within numerical tolerance.
- Cross-check Eqs. (15) and (16) on a small exact fixture.
- At `alpha=1`, assert the global scatter; at `alpha=0`, assert the local scatter.
- Reject `alpha` outside `[0,1]`.
- Add separate tests only if a non-PDF signed formulation is explicitly approved.

## Unit tests - eigen solver

- Assert `V.shape == (m,p)` and finite values.
- Assert reduced eigenvalues are ascending.
- Verify `V` lies in `range(S_T)` and selected residuals are near zero.
- Test singular, rank-zero, and nearly singular `S_T`.
- Test zero, one, and multiple numerical null directions.
- Test tolerance changes at a known spectral boundary.
- Compare deterministic projection subspaces using projector or principal-angle distance, not raw eigenvector signs.
- If a generalized solver is approved, separately test `gamma`, regularized definiteness, and residuals.

## Unit tests - base points

- Assert assignment shape `(n,)` and valid IDs `0..m-1`.
- Force empty subclasses and verify the approved handling policy.
- Assert `1 <= q <= m`, `B.shape == (q,p)`, and no NaN/inf.
- Assert a point exactly at a valid base point has near-zero score.
- Verify deterministic tie-breaking for equal memberships.

## Unit tests - anomaly scores

- Assert shape `(n_test,)`, floating dtype, finiteness, and non-negativity.
- Assert squared Euclidean reference values on a hand-computed fixture.
- Assert a controlled distant projected sample scores higher than a near-centroid sample.
- Assert batched and unbatched vectorized strategies agree.
- Assert no fitted state changes during scoring.

## Synthetic validation scenarios

| Scenario | Purpose | Expected qualitative check |
|---|---|---|
| One Gaussian normal cluster, distant anomalies | Basic separation | Anomalies receive larger scores and finite AUCs. |
| Two separated normal clusters, anomalies between them | Multimodal centroids | Multiple valid subclasses outperform a single center. |
| Two-moons normal manifold | Local geometry | Local term changes projection/scoring meaningfully. |
| Irrelevant feature noise | Scale/robustness | Performance degradation and sensitivity are recorded. |
| Contaminated normal training set | ADBench unsupervised semantics | No crash; contamination effect is quantified. |
| More anchors than meaningful modes | Empty/duplicate subclasses | Policy remains deterministic and finite. |
| Small-sample high-dimensional data | Rank deficiency | Solver returns a valid subspace or explicit diagnostic failure. |

## Ablation tests

Compare under identical splits and seeds:

1. global center only;
2. multiple anchor centers without projection;
3. `alpha=1`;
4. `alpha=0`;
5. full source-approved KLIM.

Record ROC-AUC, PR-AUC, fit/inference time, valid subclasses, selected dimension, and spectral diagnostics. Ablations must not be described as KLIM unless they match the approved source formulation.

## Reliability tests

- Run multiple fixed seeds and report mean and standard deviation.
- Require zero failed runs and no NaN/inf scores.
- Record numerical rank, selected eigenvalues, null residuals, and valid subclass count.
- Compare subspaces across repeat runs with the same seed.
- Record fit and inference time using ADBench boundaries.
- Track peak memory or analytical risk for `W,L:(n,n)`; gate datasets that exceed an approved limit.
- Verify larger scores mean more anomalous before metric integration.
- Verify ADBench writes all four result CSV files without changing schemas.

## Integration acceptance

KLIM integration is acceptable only when the detector contract passes, every blocking mathematical decision is documented as resolved, unit and synthetic suites pass, three-seed results are finite, and no training component is refitted during prediction.

