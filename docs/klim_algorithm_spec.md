# KLIM Algorithm Specification

## Authority and notation

The mathematical authority is `references/KLIM_Group.pdf`, especially Eqs. (9)-(17) and Algorithm 1. Requested formulations that conflict with it are recorded under [Ambiguities and decisions](#ambiguities-and-decisions); none is silently substituted.

| Symbol | Shape | Meaning |
|---|---:|---|
| `n`, `d` | scalar | Training samples and input features. |
| `m`, `p` | scalar | Anchors and selected projection dimensions. |
| `X` | `(n,d)` | Row-oriented training matrix. |
| `U` | `(m,d)` | Anchor matrix. |
| `S`, `Z` | `(n,m)` | Unnormalized similarities and normalized anchor memberships. |
| `H` | `(n,n)` | Centering matrix `I_n - 11^T/n`. |
| `W`, `D`, `L` | `(n,n)` | Similarity, degree, and graph-Laplacian matrices. |
| `Q` | `(m,r)` | Orthonormal basis for `range(S_T^phi)`; `r = rank(S_T^phi)`. |
| `A` | `(r,p)` | Selected reduced-space eigenvectors. |
| `V` | `(m,p)` | Anchor-space projection `QA`. |

## Stage 0 - Problem definition

KLIM is a one-class novelty detector. `fit(X_train, y_train)` receives ADBench's scaled training matrix `(n_train,d)` and visible labels `(n_train,)`; `predict_score(X_test)` receives `(n_test,d)` and returns finite `float64` scores `(n_test,)`, with larger values more anomalous. ADBench owns loading, split, preprocessing, metrics, timing, and CSV persistence.

The proposed adapter selects `X_normal = X_train[y_train == 0]`. Scientifically, this is known-normal one-class training only when `y_train == 0` reliably denotes known normal samples. In ADBench unsupervised mode, unavailable anomalies are also relabeled zero, so the same expression trains on the contaminated unlabeled set. This protocol distinction is `REQUIRES DECISION` before implementation.

## Stage 1 - Input validation and scaling

Accept a non-empty, finite, real numeric two-dimensional array. Convert to contiguous `float64`; reject NaN, infinity, zero features, feature-count mismatch, and insufficient samples. Store `n_features_in_`.

ADBench already applies training-fitted MinMax scaling. A standalone `KLIMModel` still needs a defined scaling contract. Candidate A fits and stores an internal scaler; candidate B assumes pre-scaled input and stores none. Test data must reuse training state and must never refit. Euclidean anchor distances are scale-sensitive, so unbalanced feature ranges can dominate both K-means and Eq. (9). The PDF specifies no scaler: `REQUIRES DECISION`.

## Stage 2 - Anchor generation

Fit MiniBatch K-Means to `X_normal:(n,d)` and store cluster centers `U:(m,d)` as `anchors_`.

- `1 <= n_anchors <= n` and normally `m << n`.
- Pass `random_state` and a deterministic `n_init` policy explicitly.
- Store the fitted centers, not the estimator unless diagnostics require it.
- Empty/reassigned clusters follow scikit-learn behavior during fitting; duplicate final centers must be detected diagnostically.
- Prediction uses saved anchors only.

The PDF specifies Mini-batch K-means but not batch size, initialization count, convergence settings, or duplicate-center policy.

## Stage 3 - Gaussian anchor similarity

PDF Eq. (9):

```math
s_{ij}=\exp\left(-\frac{\lVert x_i-u_j\rVert_2^2}{\sigma}\right).
```

Here `x_i:(d,)`, `u_j:(d,)`, squared distance is non-negative, and `S:(n,m)` lies in `(0,1]` mathematically. A zero distance gives `s_ij=1`. NumPy-oriented computation is `S = exp(-squared_distances(X, U) / sigma)`.

The denominator is `sigma`, not `sigma^2` or `2 sigma^2`. The PDF does not define whether its symbol denotes a bandwidth, a variance-like quantity, or how it is chosen. Require finite `sigma > 0`. Large distance-to-bandwidth ratios underflow to zero; compute normalization with a stable rowwise softmax of `-dist2/sigma`, or protect the denominator, only after confirming mathematical equivalence policy. A median positive nearest-anchor squared distance is a candidate heuristic, not a PDF formula. Supplied versus learned sigma is `REQUIRES DECISION`.

## Stage 4 - Normalized anchor representation

PDF Eq. (10):

```math
z_{ij}=\frac{s_{ij}}{\sum_{k=1}^{m}s_{ik}}, \qquad Z\in\mathbb{R}^{n\times m}.
```

Each row is a non-negative soft membership with invariant `sum_j z_ij ~= 1`. Training and inference use the same mapping and saved `U` and `sigma`. A zero/underflowed row denominator must not produce NaN. Stable exponent shifting preserves the normalized ratio; an epsilon-denominator fallback changes it slightly and must be tested.

## Stage 5 - Global error

PDF Eq. (12):

```math
H=I_n-\frac{1}{n}\mathbf{1}\mathbf{1}^T, \qquad \Psi_G=Z^T H.
```

Dimensions: `(m,n) = (m,n)(n,n)`. NumPy: `Psi_G = Z.T @ H`, or the algebraically equivalent centered transpose `Psi_G = (Z - Z.mean(axis=0)).T`. It represents deviations of anchor memberships from the global mean. Expected properties are finite values and `Psi_G @ 1 ~= 0`.

## Stage 6 - Local error

PDF Eq. (13), exactly:

```math
W=ZZ^T, \qquad L=D-W, \qquad \Psi_L=Z^T L^{1/2}.
```

Dimensions: `W,D,L,L^(1/2):(n,n)` and `Psi_L:(m,n)`. A literal NumPy strategy forms `W = Z @ Z.T`, a degree matrix `D`, a symmetric PSD square root of `L`, then `Psi_L = Z.T @ L_sqrt`.

The PDF calls this local-manifold preservation, not anchor reconstruction. It does not define `D_ii`, graph sparsification, or which matrix square root is intended. If `D_ii=sum_j W_ij`, a symmetric eigendecomposition is a candidate for the principal PSD square root. Cost and memory are `O(n^2m + n^3)` and `O(n^2)`, conflicting with the stated scalability motivation.

The requested alternative `Z.T @ Z @ Z.T` has dimensions `(m,n)(n,m)(m,n)=(m,n)` but is absent from the PDF and is not adopted. Its intended role is `REQUIRES DECISION`.

## Stage 7 - Combined regularized error

PDF Eq. (14), exactly:

```math
\Psi=\left[\sqrt{\alpha}\Psi_G\ \middle|\ \sqrt{1-\alpha}\Psi_L\right],
\qquad 0\leq\alpha\leq1.
```

Thus `Psi:(m,2n)`, `alpha=1` retains global error only, and `alpha=0` retains local error only. Concatenation deliberately produces no global-local cross terms in `Psi @ Psi.T`.

The requested `Psi = alpha*Psi_G - (1-alpha)*Psi_L` instead has `(m,n)` shape, uses unsquared weights, introduces negative global-local cross terms, and is neither Eq. (14) nor algebraically equivalent to Eq. (16). It is `REQUIRES DECISION` and must not replace the PDF formula silently.

## Stage 8 - Regularized scatter

PDF Eqs. (15)-(16):

```math
S_{wreg}^{\phi}=\Psi\Psi^T
=\alpha Z^T H Z+(1-\alpha)Z^T L Z\in\mathbb{R}^{m\times m}.
```

NumPy: `S_wreg = alpha*(Z.T @ H @ Z) + (1-alpha)*(Z.T @ L @ Z)`. It is symmetric and PSD in exact arithmetic when `L` is symmetric PSD. Symmetrize only for roundoff (`(S+S.T)/2`) and accept eigenvalues no lower than a documented negative tolerance. Direct scatter construction avoids explicitly forming `L^(1/2)` but still requires `L`. Dense cost is `O(n^2m + nm^2)`.

## Stage 9 - Total scatter

Algorithm 1 defines:

```math
S_T^{\phi}=Z^T H Z\in\mathbb{R}^{m\times m}.
```

It preserves total centered anchor-space variance and is also the global term in `S_wreg`. Store it for diagnostics or retain its range basis `Q`. Expected rank is at most `min(m,n-1)`.

## Stage 10 - NFST projection solver

The PDF does **not** define the requested generalized problem

```math
S_{wreg}v_i=\lambda_i(S_T+\gamma I)v_i.
```

Instead, PDF Eq. (17) and Algorithm 1 specify:

1. Compute `Q:(m,r)`, an orthonormal basis for `range(S_T^phi)`, via SVD.
2. Form `M = Q.T @ S_wreg @ Q:(r,r)`.
3. Solve `M a_i = lambda_i a_i`.
4. Select eigenvectors whose smallest eigenvalues satisfy `lambda_i ~= 0`.
5. Return `V = Q @ A:(m,p)`.

Use a symmetric solver; never form an explicit inverse. Sort eigenvalues ascending. Fixing eigenvector signs can stabilize serialization, but tests must compare subspaces because signs and repeated-eigenvalue bases are non-unique. Singular `S_T` is handled through its numerical range; near-rank and near-zero tolerances must be defined. `p <= r`, and the PDF implies `p` is determined by near-zero eigenvalues rather than a fixed `null_dim`.

Introducing `gamma`, a generalized solver, or a fixed projection dimension changes the source algorithm. Solver choice, tolerances, and the requested `gamma/null_dim` interface are `REQUIRES DECISION`.

## Stage 11 - Subclass assignment

PDF Eq. (11):

```math
c_i=\arg\max_{j\in\{1,\ldots,m\}}z_{ij}.
```

Assignments have shape `(n,)` and integer IDs `0..m-1` in code. The strongest anchor provides a hard pseudo-subclass for centroid construction. Ties use a deterministic first-maximum policy. Empty IDs are excluded from valid base points and retained in diagnostics.

## Stage 12 - Projected training representation

Use row vectors:

```math
Y_{train}=ZV\in\mathbb{R}^{n\times p}.
```

Row `z_i V` is the transpose-equivalent of column notation `V^T z_i`.

## Stage 13 - Subclass base points

Algorithm 1 defines:

```math
b_j=\operatorname{mean}_{i\in C_j}(z_iV)\in\mathbb{R}^{p}.
```

Stack non-empty centroids into `base_points_:(q,p)`, `1 <= q <= m`, and store matching `valid_cluster_ids_:(q,)`. Empty subclasses produce no centroid. Base points summarize projected normal modes.

## Stage 14 - Inference

```text
X_test -> saved scaling state -> saved anchors and sigma -> Z_test
       -> saved projection V -> Y_test -> saved valid base points -> scores
```

No scaler, anchor, bandwidth, projection, subclass, or centroid may be refitted in `predict_score()`.

## Stage 15 - Anomaly score

Algorithm 1 defines:

```math
A(x^*)=\min_j\lVert y^*-b_j\rVert_2^2.
```

For `Y_test:(n_t,p)` and `B:(q,p)`, calculate squared distances by batched broadcasting or `||Y||^2 + ||B||^2 - 2YB^T`, clamp tiny negative roundoff to zero, and reduce across valid base points. Output is finite `float64 (n_t,)`; larger means more anomalous. Batch over test rows when `n_t*q` is memory-sensitive.

## Stored model state

| Attribute | Shape/type | Purpose |
|---|---:|---|
| `scaler_` | fitted object or `None` | Training-only scaling state, pending policy. |
| `anchors_` | `(m,d)` | MiniBatch K-Means centers. |
| `sigma_` | scalar | Positive bandwidth used by every transform. |
| `projection_` | `(m,p)` | `V = QA`. |
| `eigenvalues_` | `(r,)` | Sorted reduced-scatter spectrum. |
| `valid_cluster_ids_` | `(q,)` | Non-empty subclass IDs. |
| `base_points_` | `(q,p)` | Projected subclass centroids. |
| `n_features_in_` | integer | Prediction shape validation. |
| `is_fitted_` | Boolean | Fit-state guard. |
| `diagnostics_` | mapping | Ranks, tolerances, empty clusters, invariants, and conditioning. |

## Hyperparameters

Defaults below are design candidates, not PDF-defined values.

| Parameter | Meaning / valid range | Initial candidate | Tuning candidates | Too small / too large |
|---|---|---:|---|---|
| `n_anchors` | `m`, integer `1..n` | `min(32,n)` | `8,16,32,64` | Underfits modes / increases memory and unstable empty subclasses. |
| `sigma` | Eq. (9) denominator, finite `>0`, or explicit heuristic token | median positive nearest-anchor `dist^2` | `0.5x,1x,2x` heuristic | One-hot/underflow / nearly uniform memberships. |
| `alpha` | Global-local weight in `[0,1]` | `0.5` | `0,0.25,0.5,0.75,1` | Local-only / global-only. |
| `null_dim` | Requested fixed `p`; absent from PDF | `None` | decision-dependent | No subspace / may exceed numerical nullity. |
| `gamma` | Requested generalized-eigen ridge; absent from PDF, `>=0` | `0` | decision-dependent | No regularization / changes geometry and dominates `S_T`. |
| `random_state` | Deterministic integer | adapter seed | ADBench seeds | Irrelevant magnitude; missing value loses reproducibility. |
| `batch_size` | MiniBatch K-Means batch, integer `>=1` | `min(1024,n)` | `256,512,1024` | Noisy updates / memory and less frequent updates. |
| `tol` | Rank/nullity/PSD numerical tolerance, finite `>0` | scale-aware `eps*size*max_eigenvalue` | `1e-10..1e-6` relative | False rank / missed null vectors. |

## Computational complexity

Let `t` be K-means iterations and `q <= m` valid subclasses.

| Operation | Time | Memory |
|---|---:|---:|
| MiniBatch K-Means | approximately `O(t * batch_size * m * d)` | `O(md)` plus batch |
| Anchor mapping | `O(nmd)` | `O(nm)` |
| Dense `W=ZZ^T`, `L` | `O(n^2m)` | `O(n^2)` |
| Scatter after dense `L` | `O(n^2m + nm^2)` | `O(n^2+m^2)` |
| Range SVD / reduced eigensolve | `O(m^3)` | `O(m^2)` |
| Assignment and base points | `O(nm+np)` | `O(n+qp)` |
| Test mapping and scoring | `O(n_tmd+n_tmp+n_tqp)` | batch-dependent |

Anchor mapping replaces an `n x n` kernel decomposition with an `m x m` projection problem, but the PDF's dense `W` and `L` restore `O(n^2)` storage and potentially `O(n^3)` work for `L^(1/2)`. A scalable graph construction is not specified and cannot be invented silently.

## Ambiguities and decisions

| Issue | Original/source formula | Dimension analysis | Proposed interpretations | Status |
|---|---|---|---|---|
| Local error conflict | PDF: `Psi_L=Z^T L^(1/2)`; request mentions `Z^T Z Z^T` | Both `(m,n)`, different operators | A: PDF Laplacian; B: requested reconstruction operator with new derivation | **REQUIRES DECISION - BLOCKING** |
| Combined-error conflict | PDF: `[sqrt(alpha)Psi_G | sqrt(1-alpha)Psi_L]`; request: `alpha Psi_G-(1-alpha)Psi_L` | `(m,2n)` vs `(m,n)`; latter adds cross terms | A: PDF concatenation; B: signed formulation as a different algorithm | **REQUIRES DECISION - BLOCKING** |
| Solver conflict | PDF: range basis plus null eigensystem; request: generalized eigenproblem with `gamma I` | `r x r` ordinary vs `m x m` generalized | A: PDF NFST; B: regularized generalized variant | **REQUIRES DECISION - BLOCKING** |
| Degree matrix | PDF: `L=D-W`, `W=ZZ^T` | `D:(n,n)` unspecified | Degree diagonal `D_ii=sum_j W_ij`; decide self-edge policy | **REQUIRES DECISION - BLOCKING** |
| Laplacian square root | PDF: `L^(1/2)` | Multiple square roots exist | Principal symmetric PSD root; or bypass through Eq. (16) | **REQUIRES DECISION** |
| Scalability contradiction | Dense `W,L:(n,n)` | `O(n^2)` memory | Exact dense source; or separately specified sparse/anchor graph approximation | **REQUIRES DECISION - BLOCKING FOR LARGE n** |
| Sigma semantics | `exp(-dist^2/sigma)` | Positive scalar | Supplied value; median squared-distance heuristic; no `sigma^2` substitution | **REQUIRES DECISION** |
| Scaling | Not defined in PDF | Affects all distances | Trust ADBench scaling; or internal standalone scaler without double scaling | **REQUIRES DECISION** |
| Training normals | One-class text; ADBench masks unavailable anomalies to zero | `X[y==0]` may equal all training data | Known-normal selection when labels are trustworthy; contaminated unlabeled protocol otherwise | **REQUIRES DECISION - BLOCKING** |
| Projection dimension | PDF selects `lambda_i ~= 0` | `p` depends on tolerance | Numerical nullity; capped requested dimension; fail if nullity zero | **REQUIRES DECISION** |
| Empty subclasses | Not specified | `q <= m` | Drop empty IDs and score against valid centroids | **REQUIRES DECISION** |

