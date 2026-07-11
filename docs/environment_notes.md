# Environment Notes

## Reproducibility boundary

The verified environment is an external virtual environment at `C:\Users\Admin\.venvs\adbench-smoke-py310`, using Python 3.10.11. The complete lock is recorded in [SETUP.md](../SETUP.md) and embedded in `scripts/setup.ps1`.

All locked distributions were present when the Isolation Forest smoke test was recorded. The list includes transitive dependencies so a future resolver cannot silently select different versions; it does not add packages beyond that verified environment.

## Intentional top-level pins

| Pin | Reason |
|---|---|
| `Python 3.10.11` | This exact interpreter completed environment verification and the smoke test. Repository metadata advertises Python 3.7-3.9, but neither 3.9 nor older interpreters were installed locally. |
| `numpy==1.26.4` | Verified numerical-array version; constrained below NumPy 2 during setup and compatible with TensorFlow 2.15.1. |
| `scipy==1.15.3` | Verified PyOD/scikit-learn numerical backend. |
| `pandas==2.3.3` | Verified result-table and CSV dependency. |
| `scikit-learn==1.0.2` | PyOD 1.0.0 imports private `_joblib_parallel_args`; scikit-learn 1.7.2 and 1.1.3 do not provide the expected compatibility path. Version 1.0.2 imports successfully. |
| `pyod==1.0.0` | The only version pinned by the repository and the verified Isolation Forest adapter version. |
| `torch==2.2.2` | `adbench.myutils` imports Torch eagerly. Torch 2.13.0 intermittently failed to initialize `c10.dll` with `WinError 1114`; 2.2.2 imported reliably. |
| `tensorflow-cpu==2.15.1` | Supplies the eager `tensorflow` import without requiring a GPU and was the verified working TensorFlow distribution. On Windows it depends on `tensorflow-intel==2.15.1`. |
| `tensorflow-intel==2.15.1` | Windows implementation selected by `tensorflow-cpu==2.15.1`; pinned to prevent a mismatched TensorFlow namespace. |
| `keras==2.15.0` | Matches TensorFlow 2.15.1 and the older Keras API generation expected by ADBench/PyOD. |
| `copulas==0.14.1` | `DataGenerator` imports Copulas eagerly even when synthetic dependency anomalies are disabled. |
| `combo==0.1.3` | `adbench.baseline.PyOD` eagerly imports PyOD's combination module, which fails without Combo even for Isolation Forest. |
| `tqdm==4.68.4` | `RunPipeline` imports it at module import time. |
| `fsspec==2026.6.0`, `requests==2.34.2`, `wget==3.2` | `adbench.myutils` imports these download dependencies eagerly. |

Matplotlib, Statsmodels, Numba, TensorBoard, Plotly, authentication libraries, and other entries in the lock are transitive dependencies of the verified top-level set. Their exact versions are pinned solely to reproduce the tested environment.

## Compatibility issues encountered

### Empty global environment

The initial global Python 3.10.11 installation contained only pip and setuptools. Level 1 imports failed for NumPy, SciPy, pandas, and scikit-learn. An external virtual environment was created to avoid contaminating the global interpreter or repository.

### Python metadata mismatch

`setup.py` lists Python 3.7, 3.8, and 3.9 classifiers but has no `python_requires`. Python 3.9 was unavailable locally. Python 3.10.11 is therefore a verified compatibility result, not a version formally declared by the package metadata.

### Eager deep-framework imports

Isolation Forest does not intrinsically require TensorFlow, Keras, or Torch. The unchanged ADBench path requires them because:

- `adbench.myutils` imports TensorFlow and Torch and seeds both frameworks;
- `adbench.run` imports the Keras backend and clears its session;
- `adbench.baseline.PyOD` imports shallow and deep PyOD model modules eagerly.

These packages remain in the reproducible environment because the smoke test exercised the unmodified public pipeline.

### TensorFlow 2.21 partial installation

Attempts to install TensorFlow 2.21.0 failed with Windows `WinError 32` file locks and left an inconsistent partial installation. It also required Keras 3.12+, `ml-dtypes>=0.5.1`, and Protobuf 6.31+, conflicting with the compatible TensorFlow/Keras 2.15 stack.

The stray TensorFlow 2.21 distribution was removed. `tensorflow-cpu==2.15.1` and `tensorflow-intel==2.15.1` were installed, with `keras==2.15.0`, `ml-dtypes==0.3.2`, and `protobuf==4.25.9`. The TensorFlow Intel namespace was force-reinstalled after cleanup, and `pip check` passed.

### scikit-learn private API

PyOD 1.0.0 imports `_joblib_parallel_args` from `sklearn.utils.fixes`. With scikit-learn 1.7.2 and then 1.1.3, importing `pyod.models.iforest` failed. Pinning scikit-learn 1.0.2 restored the expected private symbol. This coupling is fragile because it relies on a non-public scikit-learn API.

### Torch DLL initialization

Torch 2.13.0 installed but intermittently failed while loading `torch/lib/c10.dll` with `WinError 1114`. Replacing it with Torch 2.2.2 resolved the verified import path. The lock must not be upgraded casually on this Windows host.

### Optional PyOD combination dependency

After core imports passed, `adbench.baseline.PyOD` stopped at `ModuleNotFoundError: combo`. This was framework overhead from eager model imports, not an Isolation Forest algorithm requirement. Installing and pinning `combo==0.1.3` allowed the verified adapter path to proceed.

## Verification outcome

- Python: 3.10.11.
- `pip check`: no broken requirements.
- Ordered ADBench imports: verified in the smoke-test environment.
- PyOD Isolation Forest: three successful deterministic pipeline runs.
- ROC-AUC and PR-AUC: 1.0 for seeds 1, 2, and 3.
- CSV results: created under `adbench/result/`.

Do not infer that untested deep, semi-supervised, supervised, or optional baselines are verified by this environment.

