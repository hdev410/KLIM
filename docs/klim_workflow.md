# KLIM Workflow

This document summarizes the data flow. Mathematical authority and unresolved choices are in [klim_algorithm_spec.md](klim_algorithm_spec.md).

## Problem overview

KLIM maps normal training data into a normalized anchor space, finds an NFST null subspace that balances global compactness and local graph structure, builds pseudo-subclass centroids, and scores test samples by squared distance to the nearest valid centroid.

## Training workflow

```mermaid
flowchart TD
    X["X_normal (n x d)"] --> SC["scaler fit - policy pending"]
    SC --> XS["X_scaled (n x d)"]
    XS --> KM["MiniBatch K-Means"]
    KM --> U["anchors U (m x d)"]
    XS --> SIM["Gaussian similarities S (n x m)"]
    U --> SIM
    SIM --> Z["normalized Z (n x m)"]
    Z --> PG["Psi_G = Z^T H (m x n)"]
    Z --> PL["Psi_L = Z^T L^(1/2) (m x n)"]
    PG --> PSI["PDF concatenated Psi (m x 2n)"]
    PL --> PSI
    PSI --> SW["S_wreg (m x m)"]
    Z --> ST["S_T = Z^T H Z (m x m)"]
    SW --> NFST["range SVD + reduced null eigensolver"]
    ST --> NFST
    NFST --> V["V (m x p)"]
    Z --> Y["Y_train = ZV (n x p)"]
    V --> Y
    Z --> C["argmax subclass assignments (n,)"]
    C --> B["valid base points B (q x p)"]
    Y --> B
```

The diagram follows the PDF. The signed-error and generalized-eigen alternatives requested elsewhere remain unresolved and are not represented as KLIM source formulas.

## Inference workflow

```mermaid
flowchart LR
    XT["X_test (nt x d)"] --> SC["saved scaler policy"]
    SC --> U["saved anchors and sigma"]
    U --> Z["Z_test (nt x m)"]
    Z --> V["saved V"]
    V --> Y["Y_test (nt x p)"]
    Y --> B["saved base points (q x p)"]
    B --> D["nearest squared distance"]
    D --> S["scores (nt,)"]
```

Prediction validates feature count and finiteness, then reuses every fitted component. It performs no fitting or state mutation.

## ADBench integration

```mermaid
sequenceDiagram
    participant A as ADBench
    participant K as KLIM adapter
    participant M as KLIMModel
    participant U as Utils.metric

    A->>K: fit(X_train, y_train)
    K->>K: select one-class training rows per decided protocol
    K->>M: fit(X_normal)
    M-->>K: fitted model
    A->>K: predict_score(X_test)
    K->>M: decision_function(X_test)
    M-->>K: scores (n_test,)
    K-->>A: scores
    A->>U: metric(y_test, scores)
    U-->>A: ROC-AUC and PR-AUC
    A->>A: write result CSV files
```

## Matrix shapes

| Object | Shape | Created during | Retained |
|---|---:|---|---|
| `X_normal`, `X_scaled` | `(n,d)` | fit | no |
| `U` | `(m,d)` | fit | yes |
| `S`, `Z` | `(n,m)` | fit | no, unless diagnostic |
| `H`, `W`, `D`, `L` | `(n,n)` | fit | no |
| `Psi_G`, `Psi_L` | `(m,n)` | fit | no |
| PDF `Psi` | `(m,2n)` | fit | no |
| `S_wreg`, `S_T` | `(m,m)` | fit | diagnostic optional |
| `Q` | `(m,r)` | fit | diagnostic optional |
| `V` | `(m,p)` | fit | yes |
| assignments | `(n,)` | fit | diagnostic optional |
| `Y_train` | `(n,p)` | fit | no |
| `B` | `(q,p)` | fit | yes |
| `Z_test` | `(n_t,m)` | predict | no |
| `Y_test` | `(n_t,p)` | predict | no |
| scores | `(n_t,)` | predict | returned |

## Fitted state

| State | Used during prediction |
|---|---|
| `scaler_` or explicit no-scaler policy | Validate/reproduce training scaling. |
| `anchors_`, `sigma_` | Produce `Z_test`. |
| `projection_` | Produce `Y_test`. |
| `valid_cluster_ids_`, `base_points_` | Calculate nearest-centroid distance. |
| `n_features_in_`, `is_fitted_` | Guard API usage. |
| `eigenvalues_`, `diagnostics_` | Audit fit; not required for scores. |

