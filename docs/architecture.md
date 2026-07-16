# ADBench Architecture

## Directory overview

| Path | Responsibility |
|---|---|
| `adbench/run.py` | Builds experiment grids, selects detector families, coordinates training and evaluation, and saves results. |
| `adbench/datasets/data_generator.py` | Discovers and loads datasets, applies sampling and corruption settings, splits data, scales features, and controls training-label visibility. |
| `adbench/myutils.py` | Provides seeding, metrics, downloading, sampling, plotting, and result-analysis utilities. |
| `adbench/baseline/PyOD.py` | Adapts PyOD detectors to the ADBench detector interface. |
| `adbench/baseline/Customized/` | Demonstrates the interface for a user-supplied detector. |
| `adbench/baseline/*/` | Contains supervised, semi-supervised, and deep baseline implementations. |
| `adbench/datasets/*/` | Stores classical datasets and precomputed CV/NLP embeddings as `.npz` files. |
| `adbench/result/` | Runtime-created destination for metric and timing CSV files. |

## Pipeline overview

```mermaid
flowchart TD
    DS["Dataset (.npz or X/y dictionary)"] --> DG["DataGenerator.generator()"]
    DG --> RP["RunPipeline.run()"]
    RP --> DET["Detector constructor"]
    DET --> FIT["fit(X_train, y_train)"]
    FIT --> SCORE["predict_score(X_test)"]
    SCORE --> METRIC["Utils.metric(y_test, score)"]
    METRIC --> TABLES["Result DataFrames"]
    TABLES --> CSV["adbench/result/*.csv"]
```

`RunPipeline` is the coordinator. Data policy remains in `DataGenerator`, detector-specific behavior remains behind a small adapter interface, and metric calculation remains centralized in `Utils.metric`.

## Dataset lifecycle

```mermaid
flowchart LR
    A["Load X: (N,D), y: (N,)"] --> B["Optional resampling"]
    B --> C["Optional synthetic anomalies or noise"]
    C --> D["Stratified train/test split"]
    D --> E["Fit MinMaxScaler on X_train"]
    E --> F["Transform X_train and X_test"]
    F --> G["Mask unavailable training labels"]
    G --> H["X_train, y_train, X_test, y_test"]
```

`DataGenerator.generator()` loads `X` and `y` from a repository dataset or accepts a supplied dictionary through `RunPipeline.run()`. Small datasets may be resampled to the configured threshold and large datasets may be subsampled. The default split is stratified with a 30% test set. Scaling is fitted only on training features. Training anomaly labels are then exposed or masked according to the experiment's labeled-anomaly setting; test labels remain ground truth.

Repository datasets are loaded from `datasets/Classical/`, `datasets/CV_by_ResNet18/`, or `datasets/NLP_by_BERT/`; supplied dictionaries must contain aligned `X` and `y` arrays. Optional preprocessing can add a configured realistic synthetic anomaly type or robustness noise before the split.

## Detector lifecycle

```mermaid
stateDiagram-v2
    [*] --> Registered
    Registered --> Constructed: detector(seed, model_name)
    Constructed --> Fitted: fit(X_train, y_train)
    Fitted --> Scored: predict_score(X_test)
    Scored --> Released: clear session / garbage collection
    Released --> [*]
```

Built-in detectors are registered in `RunPipeline.model_dict`. A user detector can instead be passed through `RunPipeline.run(clf=...)`. `RunPipeline.model_fit()` constructs the adapter, measures fitting and inference time, requests test scores, evaluates them, then releases the detector.

## Evaluation lifecycle

`Utils.metric()` receives binary `y_test` and a one-dimensional anomaly-score vector. It calculates:

- ROC-AUC with `sklearn.metrics.roc_auc_score`;
- PR-AUC as average precision with `sklearn.metrics.average_precision_score`.

Anomaly label `1` is the positive class. Larger detector scores must indicate greater anomaly likelihood.

## Result generation

`RunPipeline.run()` maintains four DataFrames indexed by experiment parameters:

- ROC-AUC;
- PR-AUC;
- fitting time;
- inference time.

After each model execution, the DataFrames are written to `adbench/result/` as `AUCROC_<suffix>.csv`, `AUCPR_<suffix>.csv`, `Time(fit)_<suffix>.csv`, and `Time(inference)_<suffix>.csv`. The method also returns detailed results in memory.

## Experiment construction, seeding, and failures

`RunPipeline.__init__()` selects one supervision category and builds its model registry. `RunPipeline.run()` forms the experiment grid from datasets, label availability, corruption settings, and seeds; unsupervised execution retains only the zero-labeled-anomaly setting.

`DataGenerator.generator()` calls `Utils.set_seed()` before data operations. The split relies on that global NumPy seed rather than passing an explicit `random_state` to `train_test_split`. Detector adapters may seed again before fitting, and each detector should still pass the provided seed to estimator-specific random-state parameters.

The pipeline catches detector initialization and execution failures. A failed fit produces missing metrics and `None` timing values rather than changing the result schema. After successful evaluation, the pipeline clears the Keras session, releases the detector, and invokes garbage collection. Constructor, preprocessing, metric calculation, cleanup, and CSV writing remain outside the fit and inference timing boundaries.

## Architectural rationale

The structure separates experiment policy, dataset transformations, detector behavior, and evaluation. This allows the same dataset and metric rules to be applied across unsupervised, semi-supervised, and supervised methods. The adapter boundary also lets heterogeneous third-party estimators expose a consistent scoring API. Some shared modules currently couple lightweight and deep dependencies; this is documented in `dependency_audit.md` rather than treated as intended detector behavior.
