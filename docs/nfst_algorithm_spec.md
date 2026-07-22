# NFST Algorithm Specification

## Authority

The sole mathematical source of truth is `references/KLIM_Group.pdf`, especially Eqs. (9)-(17) and Algorithm 1. The detector's official project name is **NFST**: Anchor-based Kernelized One-Class LPP-NFST. The historical PDF filename is intentionally unchanged.

## Problem and data contract

`NFSTModel.fit(X_normal)` accepts a finite, non-empty `float64` matrix `(n,d)`. `NFSTModel.decision_function(X_test)` returns finite scores `(n_test,)`, where larger values mean more anomalous. The standalone model performs no feature scaling; callers must use a consistent training-fitted transformation when scaling is required.

The thin ADBench adapter is unsupervised: `NFST.fit(X_train, y_train)` fits on all rows of `X_train` and ignores `y_train`. This avoids treating ADBench's masked unsupervised labels as ground truth. ADBench owns splitting, scaling, metrics, timing, and CSV output.

## Source-defined stages

Let `U:(m,d)` be MiniBatch K-Means anchors. PDF Eq. (9) defines

```math
s_{ij}=\exp\left(-\frac{\lVert x_i-u_j\rVert_2^2}{\sigma}\right),
```

and Eq. (10) defines normalized memberships

```math
z_{ij}=\frac{s_{ij}}{\sum_{k=1}^{m}s_{ik}},\qquad Z\in\mathbb{R}^{n\times m}.
```

The implementation evaluates this ratio as a stable row-wise softmax of `-dist2/sigma`. A supplied `sigma` must be finite and positive. When `sigma=None`, training resolves it once from the median positive nearest-anchor squared distance and stores `sigma_`; prediction reuses that value.

With `H=I-11^T/n`, Eqs. (12)-(16) define

```math
\Psi_G=Z^T H,
\quad W=ZZ^T,
\quad D_{ii}=\sum_j W_{ij},
\quad L=D-W,
```

```math
S_{wreg}^{\phi}
=\alpha Z^T HZ+(1-\alpha)Z^T LZ,
\qquad
S_T^{\phi}=Z^T HZ.
```

Self-similarities are included in the degree sum. The code constructs the equivalent scatter expression directly and symmetrizes only floating-point roundoff. It does not implement the rejected reconstruction expression, signed global/local subtraction, generalized eigenproblem, ridge `gamma`, or a fixed `null_dim`.

## NFST projection solver

The source two-stage solver is:

1. Symmetrically decompose `S_T` and retain its numerical range basis `Q:(m,r)`.
2. Form `M=Q.T @ S_wreg @ Q:(r,r)`.
3. Symmetrically solve `M a_i=lambda_i a_i`, sorted ascending.
4. In source-strict mode, select only eigenvectors satisfying the scale-aware near-zero rule.
5. Return `V=Q @ A:(m,p)`.

Rank selection uses `rank_tolerance * max(1, max(abs(eigenvalues(S_T))))`. Null selection uses `null_tolerance * max(1, max(abs(eigenvalues(M))))`. `max_components`, when provided, only caps already-valid null directions; it never admits a non-null eigenvector. The solver raises a clear error when `rank(S_T)=0` or the reduced nullity is zero. No explicit inverse or generalized eigensolver is used.

Diagnostics retain the total and reduced spectra, applied thresholds, selected eigenvalues, range rank, nullity, orthogonality error, and residual norms.

The default `selection_mode="null"` is the source-strict behavior above. Real datasets can have zero reduced nullity, particularly for `alpha>0`. An explicit practical variant, `selection_mode="smallest"`, selects the `max_components` eigenvectors with the smallest reduced eigenvalues without calling them null vectors. This mode requires an explicit positive `max_components`, records its mode and selected eigenvalues in diagnostics, and is an algorithmic low-eigen extension rather than the exact-null rule in the PDF. It must never be activated silently.

## Pseudo-subclasses and scoring

PDF Eq. (11) assigns each training row by

```math
c_i=\arg\max_j z_{ij}.
```

Ties use NumPy's deterministic first-maximum rule. Projected training samples are `Y_train=ZV`. For each non-empty subclass,

```math
b_j=\operatorname{mean}_{i\in C_j}(z_iV).
```

Empty anchor IDs are excluded from `base_points_` and recorded through `valid_cluster_ids_` and counts. Test samples reuse the fitted anchors, `sigma_`, projection, and base points:

```math
A(x^*)=\min_j\lVert z(x^*)V-b_j\rVert_2^2.
```

Scoring is batched. Tiny negative squared distances caused by cancellation are clamped to zero; materially negative or non-finite values raise errors.

## Fitted state

Successful fitting stores at least `anchors_`, `sigma_`, `projection_`, `eigenvalues_`, `base_points_`, `valid_cluster_ids_`, `n_features_in_`, and `diagnostics_`. `is_fitted_` becomes true only after every stage succeeds. A failed refit clears all prior learned state, preventing partially fitted reuse. Prediction never refits training components.

## Public hyperparameters

| Parameter | Contract |
|---|---|
| `n_anchors` | Positive integer; must not exceed training rows. |
| `sigma` | `None` or a finite positive Eq. (9) denominator. |
| `alpha` | Finite value in `[0,1]`. |
| `random_state` | Integer seed used by anchor generation. |
| `batch_size` | Positive MiniBatch K-Means batch size. |
| `tol` | Positive numerical tolerance used by mapping/scatter validation. |
| `rank_tolerance` | Positive relative threshold for `range(S_T)`. |
| `null_tolerance` | Positive relative threshold for reduced null eigenvalues. |
| `max_components` | Optional positive cap on valid null directions. |
| `selection_mode` | `"null"` by default; opt-in `"smallest"` practical low-eigen variant requires `max_components`. |
| `max_graph_bytes` | Positive memory guard for dense `W` and `L`. |
| `score_batch_size` | Positive test-scoring batch size. |

## Complexity and known limitations

Anchor mapping costs `O(nmd)`. The exact source graph uses dense `W,L:(n,n)`, requiring `O(n^2)` memory and `O(n^2m)`-scale work. The implementation estimates the two dense float64 matrices and rejects work above `max_graph_bytes`; it does not invent a sparse approximation.

The strict source null-space rule can legitimately yield zero valid projection dimensions, especially when regularization removes the reduced null space. Strict mode reports this as a fit error. The opt-in low-eigen mode permits a practical projection but changes the selection rule and must be reported as a variant. Feature scaling, bandwidth choice, `alpha`, component count, and numerical tolerances remain dataset-sensitive integration risks.

## Resolved decisions

| Issue | Implemented policy |
|---|---|
| Mathematical authority | PDF Eqs. (9)-(17) and Algorithm 1 only. |
| Local/global scatter | Direct source-equivalent `alpha Z^T HZ + (1-alpha) Z^T LZ`. |
| Degree | `D_ii=sum_j W_ij`, including the diagonal. |
| Solver | Range of `S_T`, then ordinary symmetric reduced null eigensystem. |
| Projection size | Numerical nullity, optionally capped by `max_components`. |
| Empty subclasses | Exclude them from centroids and scoring. |
| Standalone scaling | No internal scaler. |
| ADBench labels | Ignore `y_train`; fit all supplied training rows. |
| Scalability | Exact dense graph with an explicit memory guard. |
