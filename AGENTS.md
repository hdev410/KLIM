# Agent Instructions

## Project purpose

ADBench is a benchmark framework for comparing anomaly detectors under consistent dataset preparation, supervision, corruption, evaluation, and result-saving policies. Future work may add KLIM, but no agent should implement it without an explicit task.

## Read before editing

Read the relevant documents first:

1. `docs/architecture.md`
2. `docs/detector_contract.md`
3. `docs/dependency_audit.md`
4. `docs/workflow.md`
5. `docs/implementation_status.md`

Prefer these documents over scanning the entire repository again. Verify source only when a detail is absent, ambiguous, or potentially stale. Keep documentation synchronized with any authorized structural or behavioral change.

## Important modules

| Path | Role |
|---|---|
| `adbench/run.py` | Experiment orchestration, model registry, timing, evaluation, and CSV saving. |
| `adbench/datasets/data_generator.py` | Dataset lifecycle, split, scaling, corruption, and label availability. |
| `adbench/myutils.py` | Shared seeding, metrics, sampling, downloading, plotting, and analysis. |
| `adbench/baseline/PyOD.py` | PyOD detector adapter. |
| `adbench/baseline/Customized/` | Reference external-detector integration. |

## Files that should almost never be modified

Treat these as benchmark-critical:

- `adbench/run.py`, especially `RunPipeline`;
- `adbench/datasets/data_generator.py`, especially `DataGenerator.generator()`;
- `adbench/myutils.py`, especially `Utils.metric()`;
- existing dataset files;
- existing baseline implementations and saved model artifacts.

Modify them only when the task explicitly requires a benchmark-wide behavior change, the impact is documented, and regression verification is planned. Do not change public APIs without a concrete justification.

## Detector interface

A benchmark-facing detector normally provides:

```python
__init__(seed: int, model_name: str | None = None)
fit(X_train, y_train) -> self
predict_score(X) -> one-dimensional scores
```

`X` is a two-dimensional numeric feature matrix. Labels are one-dimensional with normal `0` and anomaly `1`. Scores must have one value per input row and larger values must mean more anomalous. Unsupervised detectors must not treat masked `y_train` as ground truth. Honor the provided seed for detector-specific randomness.

## Coding rules

- Preserve benchmark behavior unless the task explicitly authorizes a change.
- Avoid unrelated cleanup and unnecessary refactoring.
- Keep changes small, reviewable, and limited to the requested detector or feature.
- Preserve existing public constructor, `fit()`, `predict_score()`, and pipeline signatures.
- Do not change dataset split, scaling, label masking, metric definitions, timing boundaries, or result schemas incidentally.
- Do not change dependencies or import policy without an explicit dependency task.
- Do not edit or regenerate bundled datasets or saved model files.
- Add focused tests or smoke checks proportional to the behavioral risk.
- Never claim an environment or benchmark works without executing the relevant verification.

## Style

- Use English identifiers, type hints for new public or nontrivial code, and concise docstrings.
- Comments should explain intent, assumptions, or non-obvious constraints rather than restate code.
- Use Vietnamese comments only when requested or when maintaining a nearby Vietnamese documentation convention; keep identifiers in English.
- Match the style of the surrounding module and avoid formatting unrelated lines.

## Documentation and progress

- Record stable architectural knowledge in the appropriate file under `docs/`.
- Avoid duplicating the same explanation across documents; link to the authoritative document.
- For status-only updates, modify only `docs/implementation_status.md`.
- Do not mark environment verification, smoke tests, implementation, or benchmarks complete without direct evidence.
