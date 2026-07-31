# TMPNFST Algorithm Specification

## Scope and source

`TMPNFST` is a separate unsupervised baseline derived from
`references/KLIM_Group (1).pdf`. It does not replace or modify the anchor-based
`NFST` baseline.

The source describes four stages:

1. construct a k-nearest-neighbor graph;
2. calculate minimum-rank graph-contraction lifetimes and obtain spectral
   pseudo-components;
3. construct ordinary within-, between-, and total-scatter matrices from the
   pseudo-labels and solve an NFST projection;
4. score a test sample by its squared distance to the closest projected
   component centroid.

## Implemented training flow

```text
X_train
  -> mutual k-NN Gaussian graph W
  -> repair isolated vertices using their strongest outgoing k-NN edge
  -> symmetric normalized Laplacian
  -> c smallest eigenvectors
  -> K-means on spectral rows
  -> pseudo-labels
  -> Sw, Sb, St = Sw + Sb in original feature space
  -> range-space / low-eigen NFST solver
  -> projected component centroids
```

The PDF writes the random-walk Laplacian `I-D^-1 W`. The implementation uses
the symmetric normalized Laplacian `I-D^-1/2 W D^-1/2`, which has the
corresponding spectrum and permits a stable symmetric sparse eigensolver.

The lifetime values are calculated and exposed as `lifetimes_`. They are not
used to alter the spectral labels because the PDF's Algorithm 1 does not define
a lifetime threshold or connect the lifetime calculation to a later step. The
diagnostic `lifetime_used_for_partitioning` is therefore explicitly false.

## NFST solver policy

The new model reuses the existing numerical NFST solver without changing the
old detector. It passes `St` as total scatter and `Sw` as within scatter.

The benchmark configuration explicitly uses `selection_mode="smallest"` with
two projection components. This is necessary for ordinary real datasets where
`Sw` frequently has zero exact numerical nullity. The mode is recorded in
solver diagnostics and is not represented as an exact-null result.

The multi-dataset benchmark uses `n_neighbors=15`,
`eigen_tolerance=1e-6`, and `eigen_maxiter=50000`. The sparse symmetric
Laplacian solver requests the smallest algebraic eigenvalues (`which="SA"`).
For this positive-semidefinite matrix they are also the smallest-magnitude
values, while being more robust for ARPACK on weakly connected graphs.

## Public parameters

| Parameter | Meaning |
|---|---|
| `n_neighbors` | Number of neighbors used to build the training graph. |
| `n_components` | Number of spectral pseudo-components. |
| `sigma` | Gaussian denominator; `None` resolves the median positive squared k-NN distance. |
| `n_init` | K-means initialization count in spectral space. |
| `eigen_tolerance` | Sparse Laplacian eigensolver tolerance. |
| `eigen_maxiter` | Optional maximum ARPACK iteration count. |
| `rank_tolerance` | Numerical total-scatter rank threshold. |
| `null_tolerance` | Numerical null-eigenvalue threshold. |
| `max_components` | Projection-dimension cap. |
| `selection_mode` | Exact `null` or explicit practical `smallest` mode. |
| `score_batch_size` | Optional inference batch size. |

## Benchmark coverage

`scripts/compare_nfst_tmpnfst_7_models.py` compares `NFST`, `TMPNFST`, and the
seven previously selected PyOD models with seeds 1, 2, and 3. Its nine default
datasets are Hepatitis, Ionosphere, Vowels, Cardio, InternetAds, Fault, Glass,
Yeast, and WDBC. The five additions deliberately span extreme high dimension,
high and low anomaly ratios, and small and large compatible sample counts.
`--all-compatible` discovers every local classical dataset that stays below
the original NFST dense-graph guard.

For each dataset, models are ranked by mean AUC-PR descending, with mean
AUC-ROC descending as the tie-break. The terminal prints a separate ranking
and winner block for every dataset. The output directory receives one formatted
`<dataset>_model_ranking.xlsx` workbook per dataset, in addition to the combined
CSV summary and JSON diagnostics. The XLSX export uses only Python's standard
library so the verified benchmark environment does not need an extra Excel
package.

The comparison tests are intentionally non-executing configuration checks.
Model and benchmark execution must be initiated explicitly by the user.
