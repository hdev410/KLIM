"""Compare NFST, TMPNFST, and seven PyOD models on multiple datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from adbench.baseline.TMPNFST.run import TMPNFST
from scripts.benchmark_nfst_one_dataset import (
    GRAPH_LIMIT_BYTES,
    RealDatasetNFST,
    _SplitContext,
    _run_one_model,
    estimate_dense_graph_memory,
)
from scripts.compare_nfst_7_models import (
    BASELINE_NAMES,
    RealDatasetPyOD,
    _verify_identical_splits,
)
from scripts.ranking_xlsx import write_ranking_workbook


CLASSICAL_DATASET_DIR = REPOSITORY_ROOT / "adbench" / "datasets" / "Classical"
DEFAULT_DATASET_NAMES = (
    "15_Hepatitis.npz",
    "18_Ionosphere.npz",
    "40_vowels.npz",
    "6_cardio.npz",
    # Additional datasets maximize diversity in sample count, dimensionality,
    # and anomaly ratio while respecting the original NFST graph-memory guard.
    "17_InternetAds.npz",
    "12_fault.npz",
    "14_glass.npz",
    "47_yeast.npz",
    "43_WDBC.npz",
)
TMPNFST_PARAMETERS: dict[str, Any] = {
    "n_neighbors": 15,
    "n_components": 3,
    "sigma": None,
    "n_init": 10,
    "eigen_tolerance": 1e-6,
    "eigen_maxiter": 50_000,
    "rank_tolerance": 1e-10,
    "null_tolerance": 1e-8,
    "max_components": 2,
    "selection_mode": "smallest",
    "score_batch_size": 256,
}


class RealDatasetTMPNFST(TMPNFST):
    """Capture TMPNFST diagnostics under the shared benchmark harness."""

    records: list[dict[str, Any]] = []

    def __init__(self, seed: int, model_name: str | None = None) -> None:
        super().__init__(seed=seed, model_name=model_name, **TMPNFST_PARAMETERS)

    def fit(self, X_train: Any, y_train: Any) -> RealDatasetTMPNFST:
        self._benchmark_record = {
            "seed": int(self.seed),
            "X_train_shape": list(np.shape(X_train)),
            "X_test_shape": list(np.shape(_SplitContext.X_test)),
            "split_fingerprint": dict(_SplitContext.split_fingerprint),
            "failed": False,
            "failure": None,
        }
        try:
            super().fit(X_train, y_train)
        except Exception as error:
            self._benchmark_record["failed"] = True
            self._benchmark_record["failure"] = f"{type(error).__name__}: {error}"
            self.__class__.records.append(self._benchmark_record)
            raise
        return self

    def predict_score(self, X_test: Any) -> np.ndarray:
        record = self._benchmark_record
        try:
            scores = np.asarray(super().predict_score(X_test), dtype=np.float64)
            if scores.ndim != 1 or scores.shape[0] != np.shape(X_test)[0]:
                raise ValueError("TMPNFST returned an invalid score shape.")
            if not np.all(np.isfinite(scores)):
                raise ValueError("TMPNFST produced NaN or infinity scores.")
            model = self.model
            record.update(
                {
                    "component_sizes": model.diagnostics_["partition"][
                        "component_sizes"
                    ],
                    "graph_nnz": model.diagnostics_["partition"]["adjacency_nnz"],
                    "graph_sigma": model.graph_sigma_,
                    "lifetime_max": int(np.max(model.lifetimes_)),
                    "projection_shape": list(model.projection_.shape),
                    "score_shape": list(scores.shape),
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
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


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"Invalid X/y shapes in {path.name}.")
    if not np.all(np.isfinite(X)) or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError(f"{path.name} must contain finite X and binary 0/1 y.")
    return {"X": X, "y": y}


def dataset_metadata(path: Path, dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    y = dataset["y"]
    anomalies = int(np.count_nonzero(y == 1))
    return {
        "name": path.stem,
        "path": str(path),
        "samples": int(y.shape[0]),
        "features": int(dataset["X"].shape[1]),
        "anomalies": anomalies,
        "anomaly_ratio": float(anomalies / y.shape[0]),
    }


def is_compatible_with_original_nfst(dataset: dict[str, np.ndarray]) -> bool:
    expected_train = int(np.floor(0.7 * dataset["X"].shape[0]))
    estimate = estimate_dense_graph_memory(expected_train)
    return estimate["practical_peak_bytes"] <= GRAPH_LIMIT_BYTES


def resolve_dataset_paths(
    names: Sequence[str] | None = None,
    *,
    all_compatible: bool = False,
) -> list[Path]:
    if all_compatible:
        candidates = sorted(CLASSICAL_DATASET_DIR.glob("*.npz"))
        selected = [
            path
            for path in candidates
            if is_compatible_with_original_nfst(load_dataset(path))
        ]
    else:
        selected_names = DEFAULT_DATASET_NAMES if not names else tuple(names)
        selected = [CLASSICAL_DATASET_DIR / name for name in selected_names]
    if len(selected) < 4:
        raise ValueError("Select at least four compatible datasets for this comparison.")
    return selected


def _run_dataset(path: Path) -> dict[str, Any]:
    dataset = load_dataset(path)
    if not is_compatible_with_original_nfst(dataset):
        raise MemoryError(
            f"{path.name} exceeds the dense-graph limit of the original NFST."
        )
    model_results = [
        _run_one_model(dataset, "NFST", RealDatasetNFST, f"NFST_{path.stem}_9_models"),
        _run_one_model(
            dataset,
            "TMPNFST",
            RealDatasetTMPNFST,
            f"TMPNFST_{path.stem}_9_models",
        ),
    ]
    for model_name in BASELINE_NAMES:
        model_results.append(
            _run_one_model(
                dataset,
                model_name,
                RealDatasetPyOD,
                f"{model_name}_{path.stem}_9_models",
            )
        )
    _verify_identical_splits(model_results)
    return {
        "dataset": dataset_metadata(path, dataset),
        "models": model_results,
        "same_splits": True,
    }


def run_comparison(paths: Sequence[Path]) -> dict[str, Any]:
    return {
        "datasets": [_run_dataset(path) for path in paths],
        "models": ["NFST", "TMPNFST", *BASELINE_NAMES],
        "seeds": [1, 2, 3],
        "tmpnfst_parameters": dict(TMPNFST_PARAMETERS),
    }


def summary_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dataset_report in report["datasets"]:
        dataset_name = dataset_report["dataset"]["name"]
        for result in dataset_report["models"]:
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": result["model"],
                    "Mean AUC-PR": result["aggregate"]["mean_aucpr"],
                    "Std AUC-PR": result["aggregate"]["std_aucpr"],
                    "Mean AUC-ROC": result["aggregate"]["mean_aucroc"],
                    "Std AUC-ROC": result["aggregate"]["std_aucroc"],
                    "Mean fit (s)": result["aggregate"]["mean_fit_time"],
                    "Mean inference (s)": result["aggregate"][
                        "mean_inference_time"
                    ],
                    "Failed runs": result["aggregate"]["failed_runs"],
                }
            )
    return pd.DataFrame(rows)


def ranking_frame(dataset_report: dict[str, Any]) -> pd.DataFrame:
    """Rank models by mean AUC-PR, then mean AUC-ROC, descending."""
    rows = []
    for result in dataset_report["models"]:
        aggregate = result["aggregate"]
        failed_runs = int(aggregate["failed_runs"])
        rows.append(
            {
                "Model": result["model"],
                "Mean AUC-PR": aggregate["mean_aucpr"],
                "Std AUC-PR": aggregate["std_aucpr"],
                "Mean AUC-ROC": aggregate["mean_aucroc"],
                "Std AUC-ROC": aggregate["std_aucroc"],
                "Mean Fit (s)": aggregate["mean_fit_time"],
                "Mean Inference (s)": aggregate["mean_inference_time"],
                "Successful Runs": 3 - failed_runs,
                "Status": "OK" if failed_runs == 0 else f"FAILED ({failed_runs}/3)",
            }
        )
    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            ["Mean AUC-PR", "Mean AUC-ROC"],
            ascending=[False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    return ranking


def _excel_rows(ranking: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in ranking.to_dict(orient="records"):
        rows.append(
            {
                key: (None if pd.isna(value) else value)
                for key, value in record.items()
            }
        )
    return rows


def print_rankings(report: dict[str, Any]) -> None:
    for dataset_report in report["datasets"]:
        metadata = dataset_report["dataset"]
        ranking = ranking_frame(dataset_report)
        print("\n" + "=" * 142)
        print(
            f"RANKING - {metadata['name']} | samples={metadata['samples']} | "
            f"features={metadata['features']} | anomalies={metadata['anomalies']} "
            f"({metadata['anomaly_ratio']:.2%})"
        )
        print("Primary: Mean AUC-PR | Tie-break: Mean AUC-ROC | Higher is better")
        print("-" * 142)
        print(
            ranking.to_string(
                index=False,
                na_rep="FAILED",
                formatters={
                    column: (lambda value: f"{value:.6f}")
                    for column in (
                        "Mean AUC-PR",
                        "Std AUC-PR",
                        "Mean AUC-ROC",
                        "Std AUC-ROC",
                        "Mean Fit (s)",
                        "Mean Inference (s)",
                    )
                },
            )
        )
        winner = ranking.iloc[0]
        if pd.notna(winner["Mean AUC-PR"]):
            print(
                f"\nWINNER: #{int(winner['Rank'])} {winner['Model']} | "
                f"Mean AUC-PR={winner['Mean AUC-PR']:.6f} | "
                f"Mean AUC-ROC={winner['Mean AUC-ROC']:.6f}"
            )
        else:
            print("\nWINNER: unavailable - every model run failed")
        print("=" * 142)


def save_report(
    report: dict[str, Any], output_dir: Path
) -> tuple[Path, Path, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "nfst_tmpnfst_7_models_summary.csv"
    diagnostics_path = output_dir / "nfst_tmpnfst_7_models_diagnostics.json"
    summary_frame(report).to_csv(summary_path, index=False)
    diagnostics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    workbook_paths = []
    for dataset_report in report["datasets"]:
        metadata = dataset_report["dataset"]
        ranking = ranking_frame(dataset_report)
        workbook_paths.append(
            write_ranking_workbook(
                output_dir / f"{metadata['name']}_model_ranking.xlsx",
                metadata["name"],
                metadata,
                _excel_rows(ranking),
            )
        )
    return summary_path, diagnostics_path, workbook_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Classical .npz filenames; at least four. Defaults to a curated set.",
    )
    parser.add_argument(
        "--all-compatible",
        action="store_true",
        help="Run every local Classical dataset that fits original NFST's graph limit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "adbench" / "result" / "nfst_multi_dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    dataset_paths = resolve_dataset_paths(
        arguments.datasets, all_compatible=arguments.all_compatible
    )
    comparison = run_comparison(dataset_paths)
    print_rankings(comparison)
    csv_path, json_path, workbook_paths = save_report(
        comparison, arguments.output_dir
    )
    print(f"\nSummary: {csv_path}")
    print(f"Diagnostics: {json_path}")
    print("Excel ranking files:")
    for workbook_path in workbook_paths:
        print(f"  - {workbook_path}")
