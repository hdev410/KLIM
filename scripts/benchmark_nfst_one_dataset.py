"""Run NFST and IForest on one local real ADBench dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from adbench.baseline.NFST.run import NFST
from adbench.run import RunPipeline


DATASET_PATH = (
    REPOSITORY_ROOT / "adbench" / "datasets" / "Classical" / "6_cardio.npz"
)
RESULT_DIR = REPOSITORY_ROOT / "adbench" / "result"
GRAPH_LIMIT_BYTES = 64 * 1024 * 1024
NFST_PARAMETERS: dict[str, Any] = {
    "n_anchors": 8,
    "sigma": None,
    "alpha": 0.5,
    "batch_size": 256,
    "rank_tolerance": 1e-10,
    "null_tolerance": 1e-8,
    "max_components": 2,
    "selection_mode": "smallest",
    "max_graph_bytes": GRAPH_LIMIT_BYTES,
    "score_batch_size": 256,
}


def load_selected_dataset() -> dict[str, np.ndarray]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Selected dataset is missing: {DATASET_PATH}")
    with np.load(DATASET_PATH, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("The selected dataset has invalid X/y shapes.")
    if not np.all(np.isfinite(X)) or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("The selected dataset must be finite with binary 0/1 labels.")
    return {"X": X, "y": y}


def selected_dataset_metadata(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    y = dataset["y"]
    anomalies = int(np.count_nonzero(y == 1))
    return {
        "name": DATASET_PATH.stem,
        "path": str(DATASET_PATH),
        "samples": int(y.shape[0]),
        "features": int(dataset["X"].shape[1]),
        "anomalies": anomalies,
        "anomaly_ratio": float(anomalies / y.shape[0]),
    }


def estimate_dense_graph_memory(n_train: int) -> dict[str, int]:
    if isinstance(n_train, bool) or not isinstance(n_train, (int, np.integer)):
        raise TypeError("n_train must be an integer.")
    if n_train < 1:
        raise ValueError("n_train must be positive.")
    one_matrix = int(n_train) * int(n_train) * np.dtype(np.float64).itemsize
    return {
        "one_matrix_bytes": one_matrix,
        "W_plus_L_bytes": 2 * one_matrix,
        "practical_peak_bytes": 4 * one_matrix,
        "configured_limit_bytes": GRAPH_LIMIT_BYTES,
    }


def _fingerprint(array: Any) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


class _SplitContext:
    seed: int | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    split_fingerprint: dict[str, str] = {}


class _CapturingPipeline(RunPipeline):
    def model_fit(self):
        _SplitContext.seed = int(self.seed)
        _SplitContext.X_test = np.asarray(self.data["X_test"])
        _SplitContext.y_test = np.asarray(self.data["y_test"])
        _SplitContext.split_fingerprint = {
            "X_train": _fingerprint(self.data["X_train"]),
            "y_train": _fingerprint(self.data["y_train"]),
            "X_test": _fingerprint(self.data["X_test"]),
            "y_test": _fingerprint(self.data["y_test"]),
        }
        return super().model_fit()


class RealDatasetNFST(NFST):
    records: list[dict[str, Any]] = []

    def __init__(self, seed: int, model_name: str | None = None) -> None:
        super().__init__(seed=seed, model_name=model_name, **NFST_PARAMETERS)

    def fit(self, X_train: Any, y_train: Any) -> RealDatasetNFST:
        n_train = int(np.shape(X_train)[0])
        memory = estimate_dense_graph_memory(n_train)
        record = {
            "seed": int(self.seed),
            "X_train_shape": list(np.shape(X_train)),
            "X_test_shape": list(np.shape(_SplitContext.X_test)),
            "split_fingerprint": dict(_SplitContext.split_fingerprint),
            "dense_graph_memory": memory,
            "failed": False,
            "failure": None,
        }
        if memory["practical_peak_bytes"] > GRAPH_LIMIT_BYTES:
            record["failed"] = True
            record["failure"] = (
                "Estimated practical dense-graph peak exceeds the configured limit: "
                f"{memory['practical_peak_bytes']} > {GRAPH_LIMIT_BYTES} bytes."
            )
            self.__class__.records.append(record)
            raise MemoryError(record["failure"])
        self._benchmark_record = record
        try:
            super().fit(X_train, y_train)
        except Exception as error:
            record["failed"] = True
            record["failure"] = f"{type(error).__name__}: {error}"
            self.__class__.records.append(record)
            raise
        return self

    def predict_score(self, X_test: Any) -> np.ndarray:
        record = self._benchmark_record
        try:
            scores = super().predict_score(X_test)
            if not np.all(np.isfinite(scores)):
                raise ValueError("NFST produced NaN or infinity scores.")
            y_test = np.asarray(_SplitContext.y_test)
            model = self.model
            record.update(
                {
                    "Z_train_shape": list(model.Z_train_.shape),
                    "n_anchors": int(model.anchors_.shape[0]),
                    "total_scatter_rank": int(model.range_basis_.shape[1]),
                    "reduced_dimension": int(model.eigenvalues_.shape[0]),
                    "nullity": int(model.solver_diagnostics_["numerical_nullity"]),
                    "projection_shape": list(model.projection_.shape),
                    "valid_subclass_count": int(model.valid_cluster_ids_.shape[0]),
                    "score_shape": list(scores.shape),
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                    "normal_score_mean": float(np.mean(scores[y_test == 0])),
                    "anomaly_score_mean": float(np.mean(scores[y_test == 1])),
                    "nonfinite_score_count": 0,
                }
            )
            self.__class__.records.append(record)
            return scores
        except Exception as error:
            record["failed"] = True
            record["failure"] = f"{type(error).__name__}: {error}"
            self.__class__.records.append(record)
            raise


class RealDatasetIForest:
    records: list[dict[str, Any]] = []

    def __init__(self, seed: int, model_name: str | None = None) -> None:
        from adbench.baseline.PyOD import PYOD

        self.seed = int(seed)
        self.detector = PYOD(seed=seed, model_name=model_name)

    def fit(self, X_train: Any, y_train: Any) -> RealDatasetIForest:
        self._benchmark_record = {
            "seed": self.seed,
            "X_train_shape": list(np.shape(X_train)),
            "X_test_shape": list(np.shape(_SplitContext.X_test)),
            "split_fingerprint": dict(_SplitContext.split_fingerprint),
            "failed": False,
            "failure": None,
        }
        try:
            self.detector.fit(X_train=X_train, y_train=y_train)
        except Exception as error:
            self._benchmark_record["failed"] = True
            self._benchmark_record["failure"] = f"{type(error).__name__}: {error}"
            self.__class__.records.append(self._benchmark_record)
            raise
        return self

    def predict_score(self, X_test: Any) -> np.ndarray:
        record = self._benchmark_record
        try:
            scores = np.asarray(self.detector.predict_score(X_test), dtype=np.float64)
            if not np.all(np.isfinite(scores)):
                raise ValueError("IForest produced NaN or infinity scores.")
            y_test = np.asarray(_SplitContext.y_test)
            record.update(
                {
                    "score_shape": list(scores.shape),
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                    "normal_score_mean": float(np.mean(scores[y_test == 0])),
                    "anomaly_score_mean": float(np.mean(scores[y_test == 1])),
                    "nonfinite_score_count": 0,
                }
            )
            self.__class__.records.append(record)
            return scores
        except Exception as error:
            record["failed"] = True
            record["failure"] = f"{type(error).__name__}: {error}"
            self.__class__.records.append(record)
            raise


def _result_paths(pipeline: RunPipeline) -> dict[str, Path]:
    return {
        result_type: RESULT_DIR / f"{result_type}_{pipeline.suffix}.csv"
        for result_type in ("AUCROC", "AUCPR", "Time(fit)", "Time(inference)")
    }


def _inspect_result_files(paths: dict[str, Path], model_name: str) -> dict[str, Any]:
    inspection: dict[str, Any] = {}
    for result_type, path in paths.items():
        if not path.is_file():
            raise AssertionError(f"Missing {result_type} result file: {path}")
        frame = pd.read_csv(path)
        if model_name not in frame.columns:
            raise AssertionError(f"Missing {model_name} column in {path.name}.")
        values = pd.to_numeric(frame[model_name], errors="coerce")
        inspection[result_type] = {
            "path": str(path),
            "finite_executed_values": int(np.count_nonzero(np.isfinite(values))),
            "column": model_name,
        }
    return inspection


def _metric_rows(results: list[list[Any]], model_name: str) -> list[dict[str, Any]]:
    rows = []
    for params, executed_model, metrics, fit_time, inference_time in results:
        if executed_model != model_name:
            raise AssertionError(f"Unexpected baseline executed: {executed_model}")
        rows.append(
            {
                "model": model_name,
                "seed": int(params[-1]),
                "aucroc": float(metrics["aucroc"]),
                "aucpr": float(metrics["aucpr"]),
                "fit_time": None if fit_time is None else float(fit_time),
                "inference_time": (
                    None if inference_time is None else float(inference_time)
                ),
                "failed": bool(
                    fit_time is None
                    or inference_time is None
                    or not np.isfinite(metrics["aucroc"])
                    or not np.isfinite(metrics["aucpr"])
                ),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row["failed"]]
    if not valid:
        return {
            "mean_aucroc": None,
            "std_aucroc": None,
            "mean_aucpr": None,
            "std_aucpr": None,
            "mean_fit_time": None,
            "mean_inference_time": None,
            "failed_runs": len(rows),
        }
    return {
        "mean_aucroc": float(np.mean([row["aucroc"] for row in valid])),
        "std_aucroc": float(np.std([row["aucroc"] for row in valid])),
        "mean_aucpr": float(np.mean([row["aucpr"] for row in valid])),
        "std_aucpr": float(np.std([row["aucpr"] for row in valid])),
        "mean_fit_time": float(np.mean([row["fit_time"] for row in valid])),
        "mean_inference_time": float(
            np.mean([row["inference_time"] for row in valid])
        ),
        "failed_runs": len(rows) - len(valid),
    }


def _run_one_model(
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
        n_samples_threshold=1,
    )
    pipeline.model_dict = {model_name: adapter}
    results = pipeline.run(dataset=dataset)
    if len(results) != 3:
        raise AssertionError(f"Expected three {model_name} runs, got {len(results)}.")
    rows = _metric_rows(results, model_name)
    paths = _result_paths(pipeline)
    return {
        "model": model_name,
        "metrics_per_seed": rows,
        "aggregate": _aggregate(rows),
        "diagnostics": list(adapter.records),
        "result_files": _inspect_result_files(paths, model_name),
    }


def _verify_same_splits(nfst: dict[str, Any], iforest: dict[str, Any]) -> bool:
    nfst_by_seed = {record["seed"]: record for record in nfst["diagnostics"]}
    iforest_by_seed = {record["seed"]: record for record in iforest["diagnostics"]}
    if set(nfst_by_seed) != {1, 2, 3} or set(iforest_by_seed) != {1, 2, 3}:
        return False
    return all(
        nfst_by_seed[seed]["split_fingerprint"]
        == iforest_by_seed[seed]["split_fingerprint"]
        for seed in (1, 2, 3)
    )


def run_benchmark() -> dict[str, Any]:
    dataset = load_selected_dataset()
    metadata = selected_dataset_metadata(dataset)
    expected_train = int(np.floor(0.7 * metadata["samples"]))
    memory = estimate_dense_graph_memory(expected_train)
    if memory["practical_peak_bytes"] > GRAPH_LIMIT_BYTES:
        raise MemoryError(
            "Selected dataset exceeds the conservative dense-graph memory limit: "
            f"{memory['practical_peak_bytes']} > {GRAPH_LIMIT_BYTES} bytes."
        )

    nfst = _run_one_model(dataset, "NFST", RealDatasetNFST, "NFST_cardio_real")
    iforest = _run_one_model(
        dataset,
        "IForest",
        RealDatasetIForest,
        "IForest_cardio_real",
    )
    same_splits = _verify_same_splits(nfst, iforest)
    if not same_splits:
        raise AssertionError("NFST and IForest did not receive identical splits.")

    nfst_success = (
        nfst["aggregate"]["failed_runs"] == 0
        and all(
            item["finite_executed_values"] == 3
            for item in nfst["result_files"].values()
        )
        and all(
            record.get("nonfinite_score_count", 1) == 0
            for record in nfst["diagnostics"]
        )
    )
    return {
        "dataset": metadata,
        "expected_train_samples": expected_train,
        "memory_estimate": memory,
        "nfst_parameters": NFST_PARAMETERS,
        "same_split_fingerprints": same_splits,
        "models": [nfst, iforest],
        "nfst_success": nfst_success,
    }


def _print_report(report: dict[str, Any]) -> None:
    rows = []
    for model in report["models"]:
        for row in model["metrics_per_seed"]:
            rows.append(row)
    print("\nPer-seed comparison")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nAggregates")
    print(
        pd.DataFrame(
            [
                {"model": model["model"], **model["aggregate"]}
                for model in report["models"]
            ]
        ).to_string(index=False)
    )
    print("\nNFST diagnostics")
    print(json.dumps(report["models"][0]["diagnostics"], indent=2))
    print("\nResult files")
    print(
        json.dumps(
            {
                model["model"]: model["result_files"]
                for model in report["models"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    benchmark_report = run_benchmark()
    _print_report(benchmark_report)
    if not benchmark_report["nfst_success"]:
        print(
            "\nNFST real-dataset benchmark FAILED: inspect the preserved per-seed "
            "failure diagnostics above.",
            file=sys.stderr,
        )
        raise SystemExit(1)
