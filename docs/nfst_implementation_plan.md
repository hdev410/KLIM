# NFST Implementation Plan

The standalone implementation follows `references/KLIM_Group.pdf` and the resolved policies in [nfst_algorithm_spec.md](nfst_algorithm_spec.md).

## Package and responsibilities

```text
adbench/baseline/NFST/
|-- __init__.py       # public exports only
|-- run.py            # thin ADBench adapter
|-- model.py          # validation, orchestration, fitted state
|-- anchor_mapping.py # anchors and normalized Gaussian memberships
|-- scatter.py        # source-defined graph and scatter matrices
|-- solver.py         # two-stage NFST null-space solver
`-- scoring.py        # subclasses, base points, anomaly scores
```

## Public APIs

```text
NFSTModel.fit(X_normal) -> self
NFSTModel.decision_function(X_test) -> scores

NFST.__init__(seed, model_name=None, ...)
NFST.fit(X_train, y_train) -> self
NFST.predict_score(X_test) -> scores
```

`NFSTModel` owns every mathematical stage and does not scale data internally. `NFST` preserves the ADBench seed, ignores unsupervised `y_train`, fits all rows supplied in `X_train`, and delegates scoring. It is intentionally not registered in `RunPipeline` during the standalone phase.

## Completed implementation stages

| Stage | Main file | Verification focus | Status |
|---|---|---|---|
| Detector skeleton | `model.py`, `run.py` | imports, constructor, API | Completed |
| Anchor mapping | `anchor_mapping.py` | shapes, row sums, determinism, no refit | Completed |
| Scatter construction | `scatter.py` | exact small matrices, symmetry, PSD, memory guard | Completed |
| Two-stage solver | `solver.py` | strict null selection plus explicit low-eigen practical mode, rank, residuals, sign-invariant subspaces | Completed |
| Base points and scoring | `scoring.py` | empty subclasses, exact distances, batching | Completed |
| Standalone orchestration | `model.py` | end-to-end fit/predict, atomic failed refit | Completed |
| Thin adapter | `run.py` | seed, signatures, full-row delegation | Completed |
| ADBench registration and benchmark | central pipeline | integration smoke tests and datasets | Not Started |

## Integration gate

Before registration, select the ADBench supervision category and run pipeline smoke tests without changing split, scaling, label masking, metrics, timing, or CSV behavior. Dataset-level work must explicitly handle the source solver's possible zero-nullity error and the dense graph's `O(n^2)` memory limit.
