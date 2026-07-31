"""Non-execution and isolated output checks for the multi-dataset comparison."""

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from scripts.compare_nfst_tmpnfst_7_models import (
    BASELINE_NAMES,
    DEFAULT_DATASET_NAMES,
    TMPNFST_PARAMETERS,
    ranking_frame,
    resolve_dataset_paths,
    save_report,
)


class NFSTMultiDatasetComparisonTests(unittest.TestCase):
    def test_default_comparison_has_nine_existing_datasets(self) -> None:
        paths = resolve_dataset_paths()
        self.assertEqual(len(paths), 9)
        self.assertEqual(tuple(path.name for path in paths), DEFAULT_DATASET_NAMES)
        self.assertTrue(all(path.is_file() for path in paths))

    def test_five_added_datasets_cover_diverse_shapes(self) -> None:
        self.assertEqual(
            DEFAULT_DATASET_NAMES[4:],
            (
                "17_InternetAds.npz",
                "12_fault.npz",
                "14_glass.npz",
                "47_yeast.npz",
                "43_WDBC.npz",
            ),
        )

    def test_comparison_contains_seven_original_baselines(self) -> None:
        self.assertEqual(
            BASELINE_NAMES,
            ("IForest", "OCSVM", "COPOD", "ECOD", "HBOS", "KNN", "LOF"),
        )

    def test_tmpnfst_benchmark_parameters_are_explicit(self) -> None:
        self.assertEqual(TMPNFST_PARAMETERS["n_neighbors"], 15)
        self.assertEqual(TMPNFST_PARAMETERS["n_components"], 3)
        self.assertEqual(TMPNFST_PARAMETERS["eigen_tolerance"], 1e-6)
        self.assertEqual(TMPNFST_PARAMETERS["eigen_maxiter"], 50_000)
        self.assertEqual(TMPNFST_PARAMETERS["selection_mode"], "smallest")
        self.assertEqual(TMPNFST_PARAMETERS["max_components"], 2)

    def test_fewer_than_four_requested_datasets_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least four"):
            resolve_dataset_paths(["6_cardio.npz"])

    @staticmethod
    def model_result(
        name: str,
        aucpr: float | None,
        aucroc: float | None,
        failed_runs: int = 0,
    ) -> dict:
        return {
            "model": name,
            "aggregate": {
                "mean_aucpr": aucpr,
                "std_aucpr": None if aucpr is None else 0.01,
                "mean_aucroc": aucroc,
                "std_aucroc": None if aucroc is None else 0.02,
                "mean_fit_time": None if aucpr is None else 0.1,
                "mean_inference_time": None if aucpr is None else 0.01,
                "failed_runs": failed_runs,
            },
            "metrics_per_seed": [],
            "diagnostics": [],
            "result_files": {},
        }

    def test_ranking_uses_aucpr_then_aucroc_and_places_failures_last(self) -> None:
        dataset_report = {
            "models": [
                self.model_result("B", 0.80, 0.91),
                self.model_result("A", 0.90, 0.85),
                self.model_result("C", 0.80, 0.95),
                self.model_result("FAILED", None, None, failed_runs=3),
            ]
        }
        ranking = ranking_frame(dataset_report)
        self.assertEqual(ranking["Model"].tolist(), ["A", "C", "B", "FAILED"])
        self.assertEqual(ranking["Rank"].tolist(), [1, 2, 3, 4])
        self.assertTrue(np.isnan(ranking.iloc[-1]["Mean AUC-PR"]))

    def test_save_report_writes_one_valid_xlsx_per_dataset(self) -> None:
        dataset_reports = []
        for name in ("dataset_one", "dataset_two"):
            dataset_reports.append(
                {
                    "dataset": {
                        "name": name,
                        "path": f"{name}.npz",
                        "samples": 100,
                        "features": 5,
                        "anomalies": 10,
                        "anomaly_ratio": 0.1,
                    },
                    "models": [
                        self.model_result("NFST", 0.70, 0.80),
                        self.model_result("TMPNFST", 0.85, 0.90),
                    ],
                    "same_splits": True,
                }
            )
        report = {
            "datasets": dataset_reports,
            "models": ["NFST", "TMPNFST"],
            "seeds": [1, 2, 3],
            "tmpnfst_parameters": {},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, _, workbook_paths = save_report(report, Path(temporary_directory))
            self.assertEqual(len(workbook_paths), 2)
            for workbook_path in workbook_paths:
                self.assertTrue(workbook_path.is_file())
                with ZipFile(workbook_path) as archive:
                    self.assertIn("xl/worksheets/sheet1.xml", archive.namelist())
                    sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
                    self.assertIn("TMPNFST", sheet)
                    self.assertIn("Mean AUC-PR", sheet)


if __name__ == "__main__":
    unittest.main()
