"""Focused tests for the topological manifold-partitioned NFST baseline."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import Mock

import numpy as np

from adbench.baseline.TMPNFST import TMPNFST, TMPNFSTModel
from adbench.baseline.TMPNFST.partition import (
    calculate_graph_lifetimes,
    spectral_partition,
)
from adbench.baseline.TMPNFST.scatter import construct_topological_scatter


class TMPNFSTPartitionTests(unittest.TestCase):
    @staticmethod
    def fixture() -> np.ndarray:
        rng = np.random.RandomState(14)
        return np.vstack(
            [
                rng.normal([-2.0, 0.0], 0.15, size=(30, 2)),
                rng.normal([2.0, 0.0], 0.15, size=(30, 2)),
            ]
        )

    def test_sparse_spectral_partition_has_expected_shapes(self) -> None:
        result = spectral_partition(
            self.fixture(), n_neighbors=6, n_components=2, random_state=4
        )
        self.assertEqual(result.embedding.shape, (60, 2))
        self.assertEqual(result.labels.shape, (60,))
        self.assertEqual(result.lifetimes.shape, (60,))
        self.assertEqual(set(np.unique(result.labels)), {0, 1})
        self.assertGreater(result.adjacency.nnz, 0)
        self.assertEqual((result.adjacency - result.adjacency.T).nnz, 0)
        self.assertTrue(np.all(np.isfinite(result.eigenvalues)))
        self.assertGreater(result.sigma, 0.0)

    def test_partition_is_reproducible_for_same_seed(self) -> None:
        first = spectral_partition(
            self.fixture(), n_neighbors=5, n_components=2, random_state=8
        )
        second = spectral_partition(
            self.fixture(), n_neighbors=5, n_components=2, random_state=8
        )
        self.assertTrue(np.array_equal(first.labels, second.labels))
        self.assertTrue(np.array_equal(first.lifetimes, second.lifetimes))

    def test_lifetime_peeling_assigns_positive_rounds(self) -> None:
        result = spectral_partition(
            self.fixture(), n_neighbors=4, n_components=2, random_state=2
        )
        lifetimes = calculate_graph_lifetimes(result.adjacency)
        self.assertTrue(np.all(lifetimes >= 1))
        self.assertGreaterEqual(np.max(lifetimes), np.min(lifetimes))

    def test_invalid_partition_parameters_raise(self) -> None:
        X = self.fixture()
        for kwargs in (
            {"n_neighbors": 0},
            {"n_components": 0},
            {"n_components": X.shape[0]},
            {"sigma": 0.0},
            {"eigen_tolerance": 0.0},
            {"eigen_maxiter": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    spectral_partition(X, **kwargs)


class TMPNFSTScatterTests(unittest.TestCase):
    def test_scatter_decomposes_total_variance(self) -> None:
        X = np.array(
            [[-2.0, 0.0], [-1.0, 0.2], [1.0, -0.2], [2.0, 0.0]],
            dtype=np.float64,
        )
        result = construct_topological_scatter(X, np.array([0, 0, 1, 1]))
        centered = X - X.mean(axis=0)
        expected_total = centered.T @ centered / X.shape[0]
        self.assertTrue(np.allclose(result.total_scatter, expected_total))
        self.assertTrue(
            np.allclose(
                result.within_scatter + result.between_scatter,
                result.total_scatter,
            )
        )
        self.assertLess(result.diagnostics["scatter_decomposition_max_abs"], 1e-12)


class TMPNFSTModelTests(unittest.TestCase):
    @staticmethod
    def fixture() -> np.ndarray:
        rng = np.random.RandomState(22)
        return np.vstack(
            [
                rng.normal([-2.0, 0.0, 0.0], 0.2, size=(40, 3)),
                rng.normal([2.0, 0.0, 0.0], 0.2, size=(40, 3)),
            ]
        )

    def test_end_to_end_scores_are_finite_and_prediction_does_not_refit(self) -> None:
        X = self.fixture()
        model = TMPNFSTModel(
            n_neighbors=6,
            n_components=2,
            random_state=5,
            max_components=1,
            selection_mode="smallest",
            score_batch_size=7,
        ).fit(X)
        projection_before = model.projection_.copy()
        labels_before = model.labels_.copy()
        scores = model.decision_function(X[:11])
        self.assertEqual(scores.shape, (11,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.array_equal(model.projection_, projection_before))
        self.assertTrue(np.array_equal(model.labels_, labels_before))
        self.assertFalse(model.diagnostics_["lifetime_used_for_partitioning"])

    def test_failed_refit_clears_fitted_state(self) -> None:
        model = TMPNFSTModel(n_components=2, random_state=3).fit(self.fixture())
        with self.assertRaises(ValueError):
            model.fit(np.array([[0.0, 0.0, 0.0]]))
        self.assertFalse(model.is_fitted_)
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            model.decision_function(np.zeros((1, 3)))

    def test_adapter_matches_adbench_contract_and_ignores_labels(self) -> None:
        detector = TMPNFST(seed=9, model_name="TMPNFST", n_components=2)
        detector.model.fit = Mock(return_value=detector.model)
        detector.model.decision_function = Mock(return_value=np.array([0.5]))
        X_train = np.zeros((3, 2))
        X_test = np.ones((1, 2))
        self.assertIs(detector.fit(X_train, np.array([0, 0, 1])), detector)
        self.assertTrue(np.array_equal(detector.predict_score(X_test), [0.5]))
        detector.model.fit.assert_called_once_with(X_train)
        detector.model.decision_function.assert_called_once_with(X_test)
        self.assertEqual(
            list(inspect.signature(TMPNFST.fit).parameters),
            ["self", "X_train", "y_train"],
        )


if __name__ == "__main__":
    unittest.main()
