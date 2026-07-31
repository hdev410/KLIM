# Detector Contract

## Required interface

`RunPipeline.model_fit()` expects:

```python
class Detector:
    def __init__(self, seed: int, model_name: str | None = None):
        ...

    def fit(self, X_train, y_train):
        return self

    def predict_score(self, X_test):
        return scores
```

The pipeline constructs and fits detectors with keyword arguments, so adapters must preserve these parameter names.

| Element | Requirement |
|---|---|
| Constructor | Preserve the integer experiment `seed`; accept `model_name`. |
| Training input | `X_train:(N_train,D)`, `y_train:(N_train,)`. |
| Training output | The fitted detector instance. |
| Score input | `X_test:(M,D)`. |
| Score output | Finite numeric values `(M,)`; larger means more anomalous. |
| Labels | Unsupervised detectors must not use masked `y_train` as ground truth. |
| Randomness | Pass `seed` to detector-owned stochastic components. |
| Errors | Invalid inputs and fit failures raise informative exceptions. |

In unsupervised ADBench runs, hidden anomalies can remain in `X_train` while their unavailable labels are masked to zero. Fitting time covers `fit()` and inference time covers `predict_score()`; dataset work, metrics, persistence, and cleanup are outside those calls.

## Lifecycle

```mermaid
sequenceDiagram
    participant RP as RunPipeline.model_fit
    participant D as Detector
    participant M as Utils.metric
    RP->>D: Detector(seed=seed, model_name=name)
    RP->>D: fit(X_train, y_train)
    D-->>RP: self
    RP->>D: predict_score(X_test)
    D-->>RP: scores (N_test,)
    RP->>M: metric(y_test, scores, pos_label=1)
    M-->>RP: {aucroc, aucpr}
```

## Existing examples

`adbench/baseline/PyOD.py` wraps PyOD estimators behind the `PYOD` adapter. `adbench/baseline/Customized/run.py` demonstrates the external-detector contract; its optional internal module split is organizational rather than required by `RunPipeline`.

## NFST adapter

The standalone detector and thin adapter live under:

```text
adbench/baseline/NFST/
|-- __init__.py
|-- run.py
|-- model.py
|-- anchor_mapping.py
|-- scatter.py
|-- solver.py
`-- scoring.py
```

`NFST.fit(X_train, y_train)` ignores `y_train` and delegates all supplied training rows to `NFSTModel.fit`. `NFST.predict_score(X_test)` delegates without refitting. The adapter honors the seed and returns nearest-base-point squared distances.

NFST is registered in the unsupervised `RunPipeline` registry and can also be
passed as a custom detector. Its public adapter contract remains unchanged.

## TMPNFST adapter

`adbench/baseline/TMPNFST/` provides a separate topological/spectral NFST
baseline. It follows the same unsupervised adapter contract, ignores `y_train`,
honors the supplied seed for spectral K-means/eigensolver initialization, and
returns one finite nearest-centroid squared-distance score per test row.

Both `NFST` and `TMPNFST` are registered in the unsupervised `RunPipeline`
registry. Their algorithms and fitted state are independent; TMPNFST only
reuses NFST's generic range/null-space numerical solver.
