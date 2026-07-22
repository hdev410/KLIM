"""Run deterministic NFST and IForest ADBench pipeline smoke checks."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from adbench.baseline.NFST.run import NFST
from adbench.run import RunPipeline


NFST_SMOKE_PARAMS: dict[str, Any] = {
    "n_anchors": 8,
    "sigma": 20000.0,
    "alpha": 0.5,
    "batch_size": 256,
    "rank_tolerance": 1e-10,
    "null_tolerance": 1e-8,
    "max_components": None,
    "max_graph_bytes": 64 * 1024 * 1024,
    "score_batch_size": 256,
}
RESULT_DIR = REPOSITORY_ROOT / "adbench" / "result"


def make_synthetic_dataset() -> dict[str, np.ndarray]:
    """Create separated normal and diffuse anomaly samples with seed 42."""
    rng = np.random.RandomState(42)
    normal = rng.normal(0.0, 1.0, size=(950, 20))
    anomaly_directions = rng.normal(size=(50, 20))
    anomaly_directions /= np.linalg.norm(anomaly_directions, axis=1, keepdims=True)
    anomaly_radii = rng.uniform(15.0, 25.0, size=(50, 1))
    anomalies = anomaly_directions * anomaly_radii
    return {
        "X": np.vstack([normal, anomalies]).astype(np.float64),
        "y": np.concatenate(
            [np.zeros(950, dtype=np.int64), np.ones(50, dtype=np.int64)]
        ),
    }


def direct_adapter_check(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    """Validate the adapter before allowing a full RunPipeline execution."""
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["X"],
        dataset["y"],
        test_size=0.3,
        random_state=1,
        stratify=dataset["y"],
    )
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    masked_y_train = np.zeros_like(y_train)

    detector = NFST(seed=1, model_name="NFST", **NFST_SMOKE_PARAMS)
    detector.fit(X_train, masked_y_train)
    state_before = {
        "anchors": detector.model.anchors_.copy(),
        "sigma": detector.model.sigma_,
        "projection": detector.model.projection_.copy(),
        "base_points": detector.model.base_points_.copy(),
    }
    scores = detector.predict_score(X_test)

    if scores.shape != (X_test.shape[0],):
        raise AssertionError("NFST adapter returned an invalid score shape.")
    if not np.issubdtype(scores.dtype, np.floating):
        raise AssertionError("NFST adapter scores must use a floating dtype.")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise AssertionError("NFST adapter scores must be finite and non-negative.")
    unchanged = (
        np.array_equal(state_before["anchors"], detector.model.anchors_)
        and state_before["sigma"] == detector.model.sigma_
        and np.array_equal(state_before["projection"], detector.model.projection_)
        and np.array_equal(state_before["base_points"], detector.model.base_points_)
    )
    if not unchanged:
        raise AssertionError("NFST prediction mutated fitted training state.")
    normal_mean = float(np.mean(scores[y_test == 0]))
    anomaly_mean = float(np.mean(scores[y_test == 1]))
    if not anomaly_mean > normal_mean:
        raise AssertionError("Controlled anomalies must have a higher mean score.")

    return {
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": list(masked_y_train.shape),
        "y_test_shape": list(y_test.shape),
        "score_shape": list(scores.shape),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "normal_score_mean": normal_mean,
        "anomaly_score_mean": anomaly_mean,
        "state_unchanged": unchanged,
    }


class _CaptureContext:
    y_test: np.ndarray | None = None


class _CapturingPipeline(RunPipeline):
    def model_fit(self):
        _CaptureContext.y_test = np.asarray(self.data["y_test"]).copy()
        return super().model_fit()


class SmokeNFST(NFST):
    records: list[dict[str, Any]] = []

    def __init__(self, seed: int, model_name: str | None = None) -> None:
        super().__init__(seed=seed, model_name=model_name, **NFST_SMOKE_PARAMS)

    def fit(self, X_train: Any, y_train: Any) -> SmokeNFST:
        self._train_shapes = (np.shape(X_train), np.shape(y_train))
        super().fit(X_train, y_train)
        return self

    def predict_score(self, X_test: Any) -> np.ndarray:
        scores = super().predict_score(X_test)
        y_test = np.asarray(_CaptureContext.y_test)
        model = self.model
        self.__class__.records.append(
            {
                "seed": int(self.seed),
                "X_train_shape": list(self._train_shapes[0]),
                "y_train_shape": list(self._train_shapes[1]),
                "X_test_shape": list(np.shape(X_test)),
                "y_test_shape": list(y_test.shape),
                "n_anchors": int(model.anchors_.shape[0]),
                "Z_train_shape": list(model.Z_train_.shape),
                "total_scatter_rank": int(model.range_basis_.shape[1]),
                "reduced_space_dimension": int(model.eigenvalues_.shape[0]),
                "numerical_nullity": int(
                    model.solver_diagnostics_["numerical_nullity"]
                ),
                "projection_shape": list(model.projection_.shape),
                "valid_subclass_count": int(model.valid_cluster_ids_.shape[0]),
                "score_shape": list(scores.shape),
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
                "finite_scores": bool(np.all(np.isfinite(scores))),
                "nonfinite_score_count": int(np.count_nonzero(~np.isfinite(scores))),
                "normal_score_mean": float(np.mean(scores[y_test == 0])),
                "anomaly_score_mean": float(np.mean(scores[y_test == 1])),
            }
        )
        return scores


class SmokeIForest:
    records: list[dict[str, Any]] = []

    def __init__(self, seed: int, model_name: str | None = None) -> None:
        from adbench.baseline.PyOD import PYOD

        self.seed = seed
        self.detector = PYOD(seed=seed, model_name=model_name)

    def fit(self, X_train: Any, y_train: Any) -> SmokeIForest:
        self.detector.fit(X_train=X_train, y_train=y_train)
        return self

    def predict_score(self, X_test: Any) -> np.ndarray:
        scores = np.asarray(self.detector.predict_score(X_test), dtype=np.float64)
        y_test = np.asarray(_CaptureContext.y_test)
        self.__class__.records.append(
            {
                "seed": int(self.seed),
                "score_shape": list(scores.shape),
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
                "finite_scores": bool(np.all(np.isfinite(scores))),
                "nonfinite_score_count": int(np.count_nonzero(~np.isfinite(scores))),
                "normal_score_mean": float(np.mean(scores[y_test == 0])),
                "anomaly_score_mean": float(np.mean(scores[y_test == 1])),
            }
        )
        return scores


def _result_paths(pipeline: RunPipeline) -> dict[str, Path]:
    return {
        "AUCROC": RESULT_DIR / f"AUCROC_{pipeline.suffix}.csv",
        "AUCPR": RESULT_DIR / f"AUCPR_{pipeline.suffix}.csv",
        "Time(fit)": RESULT_DIR / f"Time(fit)_{pipeline.suffix}.csv",
        "Time(inference)": RESULT_DIR / f"Time(inference)_{pipeline.suffix}.csv",
    }


def _verify_result_files(paths: dict[str, Path], model_name: str) -> None:
    for result_type, path in paths.items():
        if not path.is_file():
            raise AssertionError(f"Missing {result_type} result file: {path}")
        frame = pd.read_csv(path)
        if model_name not in frame.columns:
            raise AssertionError(f"Missing {model_name} column in {path.name}.")
        executed = pd.to_numeric(frame[model_name], errors="coerce").dropna()
        if len(executed) != 3 or not np.all(np.isfinite(executed.to_numpy())):
            raise AssertionError(f"Expected three finite executed rows in {path.name}.")


def run_model_smoke(
    dataset: dict[str, np.ndarray],
    model_name: str,
    adapter: type,
    suffix: str,
) -> dict[str, Any]:
    adapter.records.clear()
    pipeline = _CapturingPipeline(
        suffix=suffix,
        mode="rla",
        parallel="unsupervise",
        generate_duplicates=False,
        n_samples_threshold=1000,
    )
    pipeline.model_dict = {model_name: adapter}
    results = pipeline.run(dataset=dataset)
    if len(results) != 3:
        raise AssertionError(f"Expected three {model_name} executions, got {len(results)}.")

    metrics = []
    failed_runs = 0
    for params, executed_model, result, fit_time, inference_time in results:
        if executed_model != model_name:
            raise AssertionError(f"Unexpected baseline executed: {executed_model}.")
        row = {
            "seed": int(params[-1]),
            "aucroc": float(result["aucroc"]),
            "aucpr": float(result["aucpr"]),
            "fit_time": None if fit_time is None else float(fit_time),
            "inference_time": (
                None if inference_time is None else float(inference_time)
            ),
        }
        if (
            row["fit_time"] is None
            or row["inference_time"] is None
            or not np.isfinite(row["aucroc"])
            or not np.isfinite(row["aucpr"])
        ):
            failed_runs += 1
        metrics.append(row)
    if failed_runs:
        raise AssertionError(f"{model_name} had {failed_runs} failed pipeline runs.")

    paths = _result_paths(pipeline)
    _verify_result_files(paths, model_name)
    nonfinite_scores = sum(r["nonfinite_score_count"] for r in adapter.records)
    if nonfinite_scores:
        raise AssertionError(f"{model_name} produced non-finite scores.")
    if any(r["anomaly_score_mean"] <= r["normal_score_mean"] for r in adapter.records):
        raise AssertionError(
            f"{model_name} did not rank controlled anomalies higher for every seed."
        )

    return {
        "model": model_name,
        "metrics_per_seed": metrics,
        "mean_aucroc": float(np.mean([row["aucroc"] for row in metrics])),
        "mean_aucpr": float(np.mean([row["aucpr"] for row in metrics])),
        "mean_fit_time": float(np.mean([row["fit_time"] for row in metrics])),
        "mean_inference_time": float(
            np.mean([row["inference_time"] for row in metrics])
        ),
        "failed_runs": failed_runs,
        "nonfinite_score_count": nonfinite_scores,
        "records": list(adapter.records),
        "result_files": {name: str(path) for name, path in paths.items()},
    }


def run_smoke_suite() -> dict[str, Any]:
    dataset = make_synthetic_dataset()
    direct = direct_adapter_check(dataset)
    nfst = run_model_smoke(dataset, "NFST", SmokeNFST, "NFST_smoke")
    iforest = run_model_smoke(
        dataset,
        "IForest",
        SmokeIForest,
        "IForest_comparison_smoke",
    )
    return {
        "python_dataset_seed": 42,
        "raw_dataset_shape": list(dataset["X"].shape),
        "class_counts": {
            "normal": int(np.count_nonzero(dataset["y"] == 0)),
            "anomaly": int(np.count_nonzero(dataset["y"] == 1)),
        },
        "nfst_parameters": NFST_SMOKE_PARAMS,
        "direct_adapter_check": direct,
        "comparison": [nfst, iforest],
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke_suite(), indent=2, sort_keys=True))
