# Implementation Status

This is the living project-progress record. Future progress updates should modify only this file unless a task explicitly authorizes other changes.

| Work item | Status |
|---|---|
| Repository inspection | Completed |
| Dependency audit | Completed |
| Knowledge base | Completed |
| Documentation cleanup | Completed |
| Environment verification | Completed - Python 3.10.11 |
| Isolation Forest smoke test | Completed |
| NFST algorithm design | Completed |
| NFST standalone implementation | Completed |
| NFST ADBench registration | Completed |
| NFST pipeline smoke test | Completed |
| IForest comparison smoke test | Completed |
| NFST Step 9 one real-dataset benchmark | Completed with explicit low-eigen variant; strict-null attempt failed |

## Environment verification

- Python: 3.10.11
- Status: Completed

## Isolation Forest smoke test

| Item | Verified result |
|---|---|
| Status | Completed |
| Baseline | PyOD Isolation Forest |
| Dataset | Deterministic synthetic dataset |
| Samples | 950 normal, 50 anomaly |
| Features | 20 numerical features |
| Seeds | 1, 2, 3 |
| Result directory | `adbench/result/` |

| Seed | ROC-AUC | PR-AUC | Fit time (s) |
|---:|---:|---:|---:|
| 1 | 1.0 | 1.0 | 0.2461173534 |
| 2 | 1.0 | 1.0 | 0.2212970257 |
| 3 | 1.0 | 1.0 | 0.2699847221 |

Result CSV files were created under `adbench/result/`. Blank rows for labeled-anomaly ratios above zero are expected in unsupervised mode.

## Documentation cleanup

- 2026-07-16 - Documentation cleanup completed: environment compatibility findings were consolidated into `dependency_audit.md`, generic ADBench execution details were consolidated into `architecture.md`, and redundant `environment_notes.md` and `workflow.md` were removed.
- 2026-07-22 - Official detector name changed from KLIM to NFST. Package, focused tests, and detector-specific documents now use NFST; the mathematical source retains its original filename `references/KLIM_Group.pdf`.

## NFST implementation

- Step 5.1 Detector skeleton: Completed
- Step 5.2 Anchor mapping: Completed
- Step 6 Scatter construction: Completed
- Step 7 NFST solver: Completed
- Step 8 Subclass base points: Completed
- Step 9 Anomaly scoring: Completed
- Step 10 End-to-end NFSTModel: Completed
- Step 11 Thin ADBench adapter: Completed
- ADBench registration: Completed
- NFST pipeline smoke test: Completed
- IForest comparison smoke test: Completed
- Step 9 one real-dataset benchmark: Completed with low-eigen variant

### Final standalone verification

- Command: `& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' -m unittest discover -s tests -p 'test_nfst*.py' -v`
- Result: 57 tests passed, 0 failed.
- Breakdown: anchor mapping 15; scatter 14; solver 8; scoring 7; interface/adapter 8; end-to-end 5.
- Solver policy: range basis of `S_T`, reduced symmetric eigensystem, scale-aware near-zero selection, and optional cap only after null selection. No generalized eigenproblem, ridge `gamma`, or fixed `null_dim`.
- Empty subclasses: excluded from base points and nearest-centroid scoring.
- Failed refit policy: all learned state is cleared; `is_fitted_` remains false.

### Remaining integration risks

- Exact dense `W` and `L` require `O(n^2)` memory; the implementation rejects estimates above `max_graph_bytes` and provides no approximate fallback.
- The strict source solver can produce zero numerical nullity for some datasets or hyperparameters; this raises a clear fit error rather than selecting a non-null direction.
- Bandwidth, `alpha`, feature scaling, and numerical tolerances remain dataset-sensitive and require benchmark-level evaluation.
- NFST is registered only in the unsupervised `RunPipeline` registry. No dataset, metric, timing boundary, CSV schema, requirement, or setup behavior was changed.

## NFST ADBench integration verification

- Date: 2026-07-23
- Python: 3.10.11
- Environment: `C:\Users\Admin\.venvs\adbench-smoke-py310`
- Dataset: deterministic seed 42; 950 normal, 50 diffuse hypersphere anomalies, 20 numeric features.
- Pipeline split per seed: `X_train=(700,20)`, `X_test=(300,20)`, `y_train=(700,)`, `y_test=(300,)`.
- Training policy: contaminated one-class / unsupervised; NFST fits every row in `X_train` and ignores masked `y_train` values.
- Smoke parameters: `n_anchors=8`, `sigma=20000.0`, `alpha=0.5`, `batch_size=256`, `rank_tolerance=1e-10`, `null_tolerance=1e-8`, `max_components=None`, `max_graph_bytes=67108864`, `score_batch_size=256`, and detector `random_state=seed`.
- Direct adapter gate: passed; 300 finite, non-negative float scores; anchors, sigma, projection, and base points unchanged by prediction.

| Model | Seed | ROC-AUC | PR-AUC | Fit (s) | Inference (s) |
|---|---:|---:|---:|---:|---:|
| NFST | 1 | 0.8123976608 | 0.5466933493 | 0.2097208500 | 0.0010042191 |
| NFST | 2 | 0.8800000000 | 0.6160916145 | 0.2773249149 | 0.0010011196 |
| NFST | 3 | 0.8566081871 | 0.3822675713 | 0.2110249996 | 0.0000000000 |
| IForest | 1 | 1.0000000000 | 1.0000000000 | 0.1452193260 | 0.0236060619 |
| IForest | 2 | 1.0000000000 | 1.0000000000 | 0.1255972385 | 0.0216741562 |
| IForest | 3 | 1.0000000000 | 1.0000000000 | 0.1248519421 | 0.0219235420 |

| Model | Mean ROC-AUC | Mean PR-AUC | Mean fit (s) | Mean inference (s) | Failed runs | Non-finite scores |
|---|---:|---:|---:|---:|---:|---:|
| NFST | 0.8496686160 | 0.5150175117 | 0.2326902548 | 0.0006684462 | 0 | 0 |
| IForest | 1.0000000000 | 1.0000000000 | 0.1318895022 | 0.0224012534 | 0 | 0 |

Commands:

```powershell
& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' -m unittest discover -s tests -p 'test_nfst*.py' -v
& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' -m unittest tests.test_nfst_integration -v
& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' scripts/smoke_test_nfst.py
```

- Focused integration tests: 9 passed, 0 failed.
- Combined focused NFST discovery after integration: 66 passed, 0 failed (57 standalone plus 9 integration).
- Smoke script: completed with exit code zero; three NFST and three IForest executions, zero failed runs.

Generated result files use these suffixes under `adbench/result/`:

- `AUCROC_NFST_smoke_type(None)_noise(None)_unsupervise.csv`
- `AUCPR_NFST_smoke_type(None)_noise(None)_unsupervise.csv`
- `Time(fit)_NFST_smoke_type(None)_noise(None)_unsupervise.csv`
- `Time(inference)_NFST_smoke_type(None)_noise(None)_unsupervise.csv`
- the corresponding four `IForest_comparison_smoke_type(None)_noise(None)_unsupervise.csv` files.

Blank rows for labeled-anomaly ratios above zero are expected in unsupervised mode. The CSV schema is unchanged.

Known risks: dense `W` and `L` require `O(n^2)` memory; numerical nullity may be zero for other data or parameters; sigma and anchor count affect membership geometry; unsupervised training can contain contaminated anomalies; this synthetic smoke comparison is not evidence of real-data superiority; and the existing `time.time()` boundary can round sub-millisecond inference to `0.0` without indicating a failed execution.

## Step 9 - First real-dataset benchmark

- Date: 2026-07-23
- Initial strict-null status: **Failed**
- Dataset: `adbench/datasets/Classical/6_cardio.npz`
- Dataset statistics: 1,831 samples, 21 numeric features, 176 anomalies, anomaly ratio 9.6122%.
- Per-seed pipeline shapes: `X_train=(1281,21)`, `X_test=(550,21)`.
- Training policy: contaminated one-class / unsupervised; all `X_train` rows were supplied to NFST.
- Parameters: `n_anchors=8`, `sigma=None`, `alpha=0.5`, `batch_size=256`, `rank_tolerance=1e-10`, `null_tolerance=1e-8`, `max_components=None`, `max_graph_bytes=67108864`, `score_batch_size=256`, `random_state=seed`.
- Memory estimate: one dense float64 matrix 13,127,688 bytes (12.52 MiB); `W+L` 26,255,376 bytes (25.04 MiB); conservative practical peak 52,510,752 bytes (50.08 MiB), below the 64 MiB limit.
- Split fingerprints confirmed that NFST and IForest received identical training/test arrays for each seed.

### NFST result

| Seed | Result | Failure | ROC-AUC | PR-AUC | Scores |
|---:|---|---|---:|---:|---|
| 1 | Failed | Reduced NFST problem has numerical nullity zero | NaN | NaN | Not produced |
| 2 | Failed | Reduced NFST problem has numerical nullity zero | NaN | NaN | Not produced |
| 3 | Failed | Reduced NFST problem has numerical nullity zero | NaN | NaN | Not produced |

NFST stopped at the source-defined projection solver before `projection_`, base points, or anomaly scores existed. Therefore rank, reduced dimension, projection shape, subclass count, score range, and score means are unavailable. The failure was preserved; no formula, sigma, alpha, tolerance, or projection rule was changed after observing it.

### IForest comparison

| Seed | ROC-AUC | PR-AUC | Fit (s) | Inference (s) | Failed | Non-finite scores |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9324247371 | 0.6167293650 | 0.1683776379 | 0.0281460285 | No | 0 |
| 2 | 0.9437758627 | 0.6761082525 | 0.1960465908 | 0.0274419785 | No | 0 |
| 3 | 0.9195171026 | 0.5056465977 | 0.1586325169 | 0.0268974304 | No | 0 |

| Metric | Mean | Population standard deviation |
|---|---:|---:|
| ROC-AUC | 0.9319059008 | 0.0099103903 |
| PR-AUC | 0.5994947384 | 0.0706496907 |
| Fit time (s) | 0.1743522485 | 0.0158477172 |
| Inference time (s) | 0.0274951458 | 0.0005111225 |

Commands:

```powershell
& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' -m unittest discover -s tests -p 'test_nfst*.py' -v
& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' scripts/smoke_test_nfst.py
& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' -m unittest tests.test_nfst_real_benchmark -v
& 'C:\Users\Admin\.venvs\adbench-smoke-py310\Scripts\python.exe' scripts/benchmark_nfst_one_dataset.py
```

The benchmark command intentionally exited with code 1 after writing the comparison and all eight standard result files. NFST CSV rows contain expected missing values because all three fits failed; IForest has three finite executed rows in every CSV type. Generated files remain ignored under `adbench/result/` and the schema is unchanged.

- Focused benchmark-script tests: 4 passed, 0 failed.
- Final focused NFST discovery before low-eigen recovery: 70 passed, 0 failed.

Next recommended experiment: first review the interaction between the source null-space condition and `alpha=0.5` using training-only spectral diagnostics. Any later sensitivity experiment for sigma, alpha, anchor count, or null tolerance must be predeclared and must not use test metrics for selection.

Limitations: one dataset is insufficient for a scientific conclusion; only a limited training-diagnostics feasibility check was performed and no test-metric tuning was used; dense `W` and `L` limit scalability; results depend on anchor count, sigma, alpha, component count, and null tolerance; and unsupervised training may contain anomalies.

## Step 9b - Low-eigen practical recovery

- Status: **Completed**
- Decision: preserve `selection_mode="null"` as the source-strict default and add an explicit opt-in `selection_mode="smallest"` variant.
- Training-only feasibility result: `alpha=0` could produce null spaces only with very small sigma, but training scores collapsed to approximately zero, so that route was rejected before examining test metrics.
- Locked recovery configuration: `n_anchors=8`, `sigma=None`, `alpha=0.5`, `selection_mode="smallest"`, `max_components=2`, default rank/null tolerances, 64 MiB graph guard, and seed 1/2/3.
- Interpretation: this is a practical low-eigen NFST variant, not the exact-null selection rule from the PDF.

| Seed | ROC-AUC | PR-AUC | Fit (s) | Inference (s) |
|---:|---:|---:|---:|---:|
| 1 | 0.7406324741 | 0.2331787209 | 0.5549092293 | 0.0010228157 |
| 2 | 0.8484112220 | 0.3548326932 | 0.4952874184 | 0.0010094643 |
| 3 | 0.7655745796 | 0.5080632536 | 0.5173432827 | 0.0010006428 |

| Model | Mean ROC-AUC | ROC-AUC std | Mean PR-AUC | PR-AUC std | Mean fit (s) | Mean inference (s) | Failed runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| NFST low-eigen | 0.7848727586 | 0.0460679166 | 0.3653582225 | 0.1124676743 | 0.5225133101 | 0.0010109742 | 0 |
| IForest | 0.9319059008 | 0.0099103903 | 0.5994947384 | 0.0706496907 | 0.1877570152 | 0.0280300776 | 0 |

NFST diagnostics for every seed: `X_train=(1281,21)`, `X_test=(550,21)`, `Z_train=(1281,8)`, eight anchors, `rank(S_T)=7`, reduced dimension 7, exact numerical nullity 0, selected low-eigen projection `(8,2)`, eight valid subclasses, and all 550 scores finite.

| Seed | Selected eigenvalues | Score min | Score max | Normal mean | Anomaly mean |
|---:|---|---:|---:|---:|---:|
| 1 | 1347.1511, 1465.7983 | 0.0000056702 | 0.1309946249 | 0.0071667245 | 0.0214610364 |
| 2 | 1235.9833, 1422.7394 | 0.0000005871 | 0.0777043538 | 0.0050946084 | 0.0209254255 |
| 3 | 889.0113, 975.6902 | 0.0000015531 | 0.0929449676 | 0.0032678497 | 0.0233057851 |

The benchmark completed with three finite NFST runs and all four NFST CSV types containing three finite executed rows. IForest performed better on both ranking metrics on this dataset. This one result does not establish general superiority.

- Final regression after adding low-eigen mode: 72 passed, 0 failed.
