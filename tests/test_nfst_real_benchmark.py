"""Focused non-execution checks for the one-dataset NFST benchmark."""

import unittest

from scripts.benchmark_nfst_one_dataset import (
    DATASET_PATH,
    GRAPH_LIMIT_BYTES,
    NFST_PARAMETERS,
    estimate_dense_graph_memory,
    load_selected_dataset,
    selected_dataset_metadata,
)


class NFSTRealBenchmarkScriptTests(unittest.TestCase):
    def test_selected_real_dataset_exists_and_has_expected_metadata(self) -> None:
        self.assertTrue(DATASET_PATH.is_file())
        metadata = selected_dataset_metadata(load_selected_dataset())
        self.assertEqual(metadata["name"], "6_cardio")
        self.assertEqual(metadata["samples"], 1831)
        self.assertEqual(metadata["features"], 21)
        self.assertEqual(metadata["anomalies"], 176)

    def test_memory_estimation_is_deterministic_and_below_limit(self) -> None:
        first = estimate_dense_graph_memory(1281)
        second = estimate_dense_graph_memory(1281)
        self.assertEqual(first, second)
        self.assertEqual(first["one_matrix_bytes"], 13_127_688)
        self.assertEqual(first["W_plus_L_bytes"], 26_255_376)
        self.assertEqual(first["practical_peak_bytes"], 52_510_752)
        self.assertLess(first["practical_peak_bytes"], GRAPH_LIMIT_BYTES)

    def test_memory_estimation_rejects_invalid_sizes(self) -> None:
        for invalid in (0, -1):
            with self.assertRaises(ValueError):
                estimate_dense_graph_memory(invalid)
        with self.assertRaises(TypeError):
            estimate_dense_graph_memory(1.5)

    def test_nfst_real_benchmark_parameters_are_frozen(self) -> None:
        self.assertEqual(NFST_PARAMETERS["n_anchors"], 8)
        self.assertIsNone(NFST_PARAMETERS["sigma"])
        self.assertEqual(NFST_PARAMETERS["alpha"], 0.5)
        self.assertEqual(NFST_PARAMETERS["null_tolerance"], 1e-8)
        self.assertEqual(NFST_PARAMETERS["max_components"], 2)
        self.assertEqual(NFST_PARAMETERS["selection_mode"], "smallest")
        self.assertEqual(NFST_PARAMETERS["max_graph_bytes"], GRAPH_LIMIT_BYTES)


if __name__ == "__main__":
    unittest.main()
