"""Focused ADBench registry and pipeline checks for NFST."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

import numpy as np

from adbench.baseline.NFST.run import NFST
from adbench.baseline.TMPNFST.run import TMPNFST
from adbench.run import RunPipeline
from scripts.smoke_test_nfst import (
    SmokeNFST,
    direct_adapter_check,
    make_synthetic_dataset,
    run_model_smoke,
)


EXPECTED_EXISTING_UNSUPERVISED = {
    "IForest",
    "OCSVM",
    "CBLOF",
    "COF",
    "COPOD",
    "ECOD",
    "HBOS",
    "KNN",
    "LODA",
    "LOF",
    "PCA",
    "SOD",
    "DeepSVDD",
    "DAGMM",
}


class NFSTIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = make_synthetic_dataset()
        cls.direct = direct_adapter_check(cls.dataset)
        cls.pipeline_result = run_model_smoke(
            cls.dataset,
            "NFST",
            SmokeNFST,
            "NFST_integration_test",
        )

    def test_nfst_is_registered_only_as_an_unsupervised_detector(self) -> None:
        pipeline = RunPipeline(
            suffix="registry_test",
            parallel="unsupervise",
            generate_duplicates=False,
            n_samples_threshold=1000,
        )
        self.assertIs(pipeline.model_dict["NFST"], NFST)
        self.assertIs(pipeline.model_dict["TMPNFST"], TMPNFST)

    def test_existing_unsupervised_registry_entries_are_unchanged(self) -> None:
        pipeline = RunPipeline(
            suffix="registry_entries_test",
            parallel="unsupervise",
            generate_duplicates=False,
            n_samples_threshold=1000,
        )
        self.assertEqual(
            set(pipeline.model_dict) - {"NFST", "TMPNFST"},
            EXPECTED_EXISTING_UNSUPERVISED,
        )
        self.assertEqual(pipeline.model_dict["IForest"].__name__, "PYOD")

    def test_registry_constructs_nfst_with_contract_keywords(self) -> None:
        pipeline = RunPipeline(
            suffix="registry_constructor_test",
            parallel="unsupervise",
            generate_duplicates=False,
            n_samples_threshold=1000,
        )
        detector = pipeline.model_dict["NFST"](seed=7, model_name="NFST")
        self.assertEqual(detector.seed, 7)
        self.assertEqual(detector.model_name, "NFST")

    def test_direct_adapter_contract_passes(self) -> None:
        self.assertEqual(self.direct["score_shape"], self.direct["X_test_shape"][:1])
        self.assertTrue(self.direct["state_unchanged"])
        self.assertGreaterEqual(self.direct["score_min"], 0.0)
        self.assertGreater(self.direct["anomaly_score_mean"], self.direct["normal_score_mean"])

    def test_three_seed_pipeline_has_finite_metrics_and_no_failures(self) -> None:
        result = self.pipeline_result
        self.assertEqual(result["failed_runs"], 0)
        self.assertEqual(result["nonfinite_score_count"], 0)
        self.assertEqual(len(result["metrics_per_seed"]), 3)
        for row in result["metrics_per_seed"]:
            self.assertTrue(np.isfinite(row["aucroc"]))
            self.assertTrue(np.isfinite(row["aucpr"]))
            self.assertIsNotNone(row["fit_time"])
            self.assertIsNotNone(row["inference_time"])

    def test_pipeline_scores_have_expected_shape_orientation_and_values(self) -> None:
        for record in self.pipeline_result["records"]:
            self.assertEqual(record["score_shape"], record["y_test_shape"])
            self.assertTrue(record["finite_scores"])
            self.assertGreaterEqual(record["score_min"], 0.0)
            self.assertGreater(
                record["anomaly_score_mean"], record["normal_score_mean"]
            )

    def test_pipeline_writes_all_four_standard_csv_types(self) -> None:
        self.assertEqual(
            set(self.pipeline_result["result_files"]),
            {"AUCROC", "AUCPR", "Time(fit)", "Time(inference)"},
        )

    def test_smoke_registry_executes_no_unrelated_baseline(self) -> None:
        self.assertEqual(self.pipeline_result["model"], "NFST")
        self.assertEqual(len(self.pipeline_result["records"]), 3)

    def test_nfst_package_imports_no_forbidden_framework(self) -> None:
        code = (
            "import json,sys; import adbench.baseline.NFST; "
            "print(json.dumps([n for n in "
            "('tensorflow','torch','keras','pyod') if n in sys.modules]))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout.strip()), [])


if __name__ == "__main__":
    unittest.main()
