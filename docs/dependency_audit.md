# Dependency Audit

## Dependency classes

| Class | Meaning | IForest example |
|---|---|---|
| Intrinsic | Needed by the detector algorithm or its direct numerical backend. | PyOD, NumPy, SciPy, scikit-learn. |
| Benchmark framework | Needed for shared data, orchestration, metrics, progress, or result storage. | pandas, tqdm, scikit-learn preprocessing and metrics. |
| Framework overhead | Imported or executed because shared modules are coupled to unrelated functionality. | TensorFlow, Torch, Keras, Copulas, DAGMM, unrelated PyOD detector modules. |
| Deep-model only | Provides neural networks, optimizers, data loaders, or deep architectures. | TensorFlow/Keras, Torch/Torchvision, rtdl, delu. |
| Optional feature | Supports workflows not used in an ordinary IForest experiment. | Copulas for dependency anomalies; download and plotting packages. |

`requirements.txt` pins only `pyod==1.0.0`. Other declared packages are unpinned. Several directly imported packages, including NumPy, pandas, SciPy, matplotlib, requests, Torchvision, NetworkX, Seaborn, and Click, are not declared directly.

## Import graph

```mermaid
flowchart TD
    R["adbench.run"] --> K["keras.backend"]
    R --> DG["DataGenerator"]
    R --> U["Utils"]
    DG --> U
    DG --> SK["scikit-learn"]
    DG --> CP["Copulas"]
    U --> TF["TensorFlow"]
    U --> T["Torch"]
    U --> MET["scikit-learn metrics"]
    R -->|unsupervised constructor| P["PYOD adapter"]
    R -->|unsupervised constructor| DA["DAGMM"]
    P --> IF["PyOD IForest"]
    P --> OTHER["All other PyOD detectors"]
    OTHER --> DEEP["Deep-model stacks"]
    DA --> T
```

## Dependency graph

```mermaid
flowchart LR
    subgraph Required["Required for IForest benchmark semantics"]
        N["NumPy"]
        S["SciPy"]
        SK["scikit-learn"]
        P["PyOD"]
        PD["pandas"]
        TQ["tqdm"]
    end

    subgraph Overhead["Required by current framework design"]
        TF["TensorFlow"]
        TO["Torch"]
        K["Keras"]
        C["Copulas"]
        D["DAGMM import chain"]
        O["Unselected PyOD modules"]
    end

    subgraph Deep["Deep models only"]
        TV["Torchvision"]
        R["rtdl / delu"]
        NN["VAE / AutoEncoder / GAAL / DeepSVDD"]
    end
```

TensorFlow and Torch are not intrinsic IForest dependencies, but `Utils.set_seed()` currently calls both frameworks. Keras is not intrinsic, but `RunPipeline.model_fit()` unconditionally calls `K.clear_session()`.

## Module dependency tree

```text
adbench.run
├── DataGenerator
│   ├── scikit-learn split/scaling/mixture
│   ├── Copulas
│   └── Utils
│       ├── TensorFlow and Torch seeding
│       ├── scikit-learn metrics
│       ├── download dependencies
│       └── plotting/statistical dependencies
├── Keras backend
└── unsupervised model registry
    ├── PYOD
    │   ├── IForest
    │   ├── other shallow detectors
    │   ├── deep detectors
    │   └── XGBOD
    └── DAGMM
        └── Torch training and evaluation modules
```

## Current eager imports

- Importing `adbench.run` imports Keras, `DataGenerator`, and `Utils`.
- Importing `DataGenerator` imports Copulas even when synthetic dependency anomalies are disabled.
- Importing `Utils` imports TensorFlow, Torch, download packages, plotting packages, and SciPy statistics.
- Constructing an unsupervised pipeline imports both `PYOD` and DAGMM.
- Importing `PYOD` imports every supported PyOD detector, including deep and XGBoost-backed models.

## Potential lazy-import opportunities

These are audit findings, not implemented changes:

- Resolve only the selected PyOD detector class.
- Import DAGMM only when DAGMM is selected.
- Load Keras only for models that require session cleanup.
- Load Copulas only for dependency-anomaly generation.
- Load download, plotting, and statistical packages inside their corresponding utility methods.
- Seed TensorFlow and Torch only when the selected detector uses those frameworks, subject to an explicit reproducibility policy.

The last item is not a mechanical import change: it changes global framework-seeding side effects and therefore needs targeted regression tests.
