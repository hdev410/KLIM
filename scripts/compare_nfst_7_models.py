"""Compare NFST with seven unsupervised baselines on the Cardio dataset."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from benchmark_nfst_one_dataset import (
    DATASET_PATH,
    GRAPH_LIMIT_BYTES,
    RealDatasetNFST,
    _SplitContext,
    _run_one_model,
    estimate_dense_graph_memory,
    load_selected_dataset,
    selected_dataset_metadata,
)


BASELINE_NAMES = ("IForest", "OCSVM", "COPOD", "ECOD", "HBOS", "KNN", "LOF")


class RealDatasetPyOD:
    """Capture diagnostics around the shared ADBench PyOD adapter."""

    records: list[dict[str, Any]] = []

    def __init__(self, seed: int, model_name: str | None = None) -> None:
        from adbench.baseline.PyOD import PYOD

        if model_name not in BASELINE_NAMES:
            raise ValueError(f"Unsupported comparison model: {model_name!r}.")
        self.seed = int(seed)
        self.model_name = model_name
        self.detector = PYOD(seed=self.seed, model_name=self.model_name)

    def fit(self, X_train: Any, y_train: Any) -> RealDatasetPyOD:
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
            scores = np.asarray(
                self.detector.predict_score(X_test),
                dtype=np.float64,
            )
            if scores.ndim != 1 or scores.shape[0] != np.shape(X_test)[0]:
                raise ValueError("Detector returned an invalid score shape.")
            if not np.all(np.isfinite(scores)):
                raise ValueError("Detector produced NaN or infinity scores.")
            record["nonfinite_score_count"] = 0
            self.__class__.records.append(record)
            return scores
        except Exception as error:
            record["failed"] = True
            record["failure"] = f"{type(error).__name__}: {error}"
            self.__class__.records.append(record)
            raise


def _records_by_seed(model_result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(record["seed"]): record for record in model_result["diagnostics"]}


def _verify_identical_splits(model_results: list[dict[str, Any]]) -> None:
    expected_seeds = {1, 2, 3}
    reference = _records_by_seed(model_results[0])
    if set(reference) != expected_seeds:
        raise AssertionError("NFST did not produce diagnostics for all three seeds.")

    for model_result in model_results[1:]:
        candidate = _records_by_seed(model_result)
        if set(candidate) != expected_seeds:
            raise AssertionError(
                f"{model_result['model']} did not run all three seeds."
            )
        for seed in expected_seeds:
            if (
                candidate[seed]["split_fingerprint"]
                != reference[seed]["split_fingerprint"]
            ):
                raise AssertionError(
                    f"{model_result['model']} received a different split "
                    f"for seed {seed}."
                )


def _run_quietly(
    dataset: dict[str, np.ndarray],
    model_name: str,
    adapter: type,
) -> dict[str, Any]:
    print(f"Running {model_name:<7} (seeds 1, 2, 3)...", flush=True)
    captured_stdout = StringIO()
    captured_stderr = StringIO()
    try:
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            return _run_one_model(
                dataset,
                model_name,
                adapter,
                f"{model_name}_cardio_8_model_comparison",
            )
    except Exception:
        print(captured_stdout.getvalue(), end="")
        print(captured_stderr.getvalue(), end="")
        raise


def run_comparison() -> dict[str, Any]:
    dataset = load_selected_dataset()
    metadata = selected_dataset_metadata(dataset)
    expected_train = int(np.floor(0.7 * metadata["samples"]))
    memory = estimate_dense_graph_memory(expected_train)
    if memory["practical_peak_bytes"] > GRAPH_LIMIT_BYTES:
        raise MemoryError(
            "Selected dataset exceeds the NFST dense-graph memory limit: "
            f"{memory['practical_peak_bytes']} > {GRAPH_LIMIT_BYTES} bytes."
        )

    model_results = [_run_quietly(dataset, "NFST", RealDatasetNFST)]
    for model_name in BASELINE_NAMES:
        model_results.append(
            _run_quietly(dataset, model_name, RealDatasetPyOD)
        )

    _verify_identical_splits(model_results)
    return {
        "dataset": metadata,
        "models": model_results,
        "same_splits": True,
    }


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "FAILED"
    return f"{mean:.4f} +/- {std:.4f}"


def print_comparison(report: dict[str, Any]) -> None:
    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_result in report["models"]:
        for row in model_result["metrics_per_seed"]:
            per_seed_rows.append(
                {
                    "Model": row["model"],
                    "Seed": row["seed"],
                    "AUC-PR": row["aucpr"],
                    "AUC-ROC": row["aucroc"],
                    "Fit (s)": row["fit_time"],
                    "Inference (s)": row["inference_time"],
                    "Status": "FAILED" if row["failed"] else "OK",
                }
            )

        aggregate = model_result["aggregate"]
        completed_runs = 3 - int(aggregate["failed_runs"])
        summary_rows.append(
            {
                "Model": model_result["model"],
                "AUC-PR (mean +/- std)": _format_mean_std(
                    aggregate["mean_aucpr"],
                    aggregate["std_aucpr"],
                ),
                "AUC-ROC (mean +/- std)": _format_mean_std(
                    aggregate["mean_aucroc"],
                    aggregate["std_aucroc"],
                ),
                "Mean fit (s)": aggregate["mean_fit_time"],
                "Mean inference (s)": aggregate["mean_inference_time"],
                "Runs": f"{completed_runs}/3",
                "Status": "OK" if aggregate["failed_runs"] == 0 else "FAILED",
                "_sort_aucpr": (
                    -np.inf
                    if aggregate["mean_aucpr"] is None
                    else aggregate["mean_aucpr"]
                ),
            }
        )

    per_seed = pd.DataFrame(per_seed_rows)
    for column in ("AUC-PR", "AUC-ROC", "Fit (s)", "Inference (s)"):
        per_seed[column] = per_seed[column].map(
            lambda value: "—" if value is None else f"{value:.6f}"
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("_sort_aucpr", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "Rank", np.arange(1, len(summary) + 1))
    summary = summary.drop(columns="_sort_aucpr")
    for column in ("Mean fit (s)", "Mean inference (s)"):
        summary[column] = summary[column].map(
            lambda value: "—" if value is None else f"{value:.6f}"
        )

    metadata = report["dataset"]
    print("\n" + "=" * 104)
    print("NFST VS 7 UNSUPERVISED MODELS")
    print("=" * 104)
    print(
        f"Dataset: {DATASET_PATH.name} | Samples: {metadata['samples']} | "
        f"Features: {metadata['features']} | Anomalies: {metadata['anomalies']} "
        f"({metadata['anomaly_ratio']:.2%})"
    )
    print("Seeds: 1, 2, 3 | Primary ranking metric: AUC-PR | Identical splits: YES")

    print("\nPER-SEED RESULTS")
    print(per_seed.to_string(index=False))

    print("\nFINAL RANKING (higher AUC-PR is better)")
    print(summary.to_string(index=False))
    print("=" * 104)


if __name__ == "__main__":
    comparison_report = run_comparison()
    print_comparison(comparison_report)
