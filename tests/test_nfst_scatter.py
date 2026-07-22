"""Focused tests for NFST Step 6 scatter construction."""

import unittest

import numpy as np

from adbench.baseline.NFST.model import NFSTModel
from adbench.baseline.NFST.scatter import (
    assign_subclasses,
    construct_graph_laplacian,
    construct_scatter_matrices,
    validate_memberships,
)


class ScatterConstructionTests(unittest.TestCase):
    @staticmethod
    def fixture() -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)

    @staticmethod
    def expected_fixture() -> dict[str, np.ndarray]:
        return {
            "W": np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 0.5]]),
            "D": np.diag([1.5, 1.5, 1.5]),
            "L": np.array([[0.5, 0.0, -0.5], [0.0, 0.5, -0.5], [-0.5, -0.5, 1.0]]),
            "global": np.array([[0.5, -0.5], [-0.5, 0.5]]),
            "local": np.array([[0.25, -0.25], [-0.25, 0.25]]),
            "regularized": np.array([[0.3125, -0.3125], [-0.3125, 0.3125]]),
        }

    def test_subclass_assignments_and_first_maximum_tie_breaking(self) -> None:
        assignments = assign_subclasses(self.fixture())

        self.assertEqual(assignments.shape, (3,))
        self.assertTrue(np.issubdtype(assignments.dtype, np.integer))
        self.assertTrue(np.all((0 <= assignments) & (assignments < 2)))
        self.assertTrue(np.array_equal(assignments, np.array([0, 1, 0])))

    def test_empty_subclasses_are_diagnosed(self) -> None:
        Z = np.array([[0.8, 0.2, 0.0], [0.7, 0.3, 0.0]])
        result = construct_scatter_matrices(Z, alpha=0.5)

        self.assertEqual(result.diagnostics["empty_subclass_ids"], [1, 2])

    def test_reference_graph_matches_explicit_values(self) -> None:
        expected = self.expected_fixture()
        graph = construct_graph_laplacian(self.fixture())

        self.assertTrue(np.allclose(graph.similarity_graph, expected["W"]))
        self.assertTrue(np.allclose(graph.degree, expected["W"].sum(axis=1)))
        self.assertTrue(np.allclose(np.diag(graph.degree), expected["D"]))
        self.assertTrue(np.allclose(graph.laplacian, expected["L"]))
        self.assertTrue(np.allclose(graph.laplacian.sum(axis=1), 0.0))
        self.assertTrue(np.allclose(graph.similarity_graph, graph.similarity_graph.T))
        self.assertTrue(np.allclose(graph.laplacian, graph.laplacian.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(graph.laplacian).min(), -1e-10)
        self.assertTrue(graph.diagnostics["self_similarity_preserved"])

    def test_reference_scatter_matches_explicit_values(self) -> None:
        Z = self.fixture()
        expected = self.expected_fixture()
        result = construct_scatter_matrices(Z, alpha=0.25)

        self.assertTrue(np.allclose(result.global_scatter, expected["global"]))
        self.assertTrue(np.allclose(result.local_scatter, expected["local"]))
        self.assertTrue(np.allclose(result.total_scatter, expected["global"]))
        self.assertTrue(
            np.allclose(result.regularized_scatter, expected["regularized"])
        )

    def test_centered_global_equals_explicit_H_expression(self) -> None:
        Z = self.fixture()
        n_samples = Z.shape[0]
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        result = construct_scatter_matrices(Z, alpha=0.5)

        self.assertTrue(np.allclose(result.global_scatter, Z.T @ H @ Z))

    def test_local_scatter_equals_direct_laplacian_expression(self) -> None:
        Z = self.fixture()
        graph = construct_graph_laplacian(Z)
        result = construct_scatter_matrices(Z, alpha=0.5)

        self.assertTrue(np.allclose(result.local_scatter, Z.T @ graph.laplacian @ Z))

    def test_alpha_boundaries_select_corresponding_scatter(self) -> None:
        Z = self.fixture()
        global_only = construct_scatter_matrices(Z, alpha=1.0)
        local_only = construct_scatter_matrices(Z, alpha=0.0)

        self.assertTrue(
            np.array_equal(global_only.regularized_scatter, global_only.global_scatter)
        )
        self.assertTrue(
            np.array_equal(local_only.regularized_scatter, local_only.local_scatter)
        )

    def test_invalid_alpha_raises_value_error(self) -> None:
        for alpha in (-0.1, 1.1, np.nan, np.inf, "invalid"):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    construct_scatter_matrices(self.fixture(), alpha=alpha)

    def test_scatter_shapes_finiteness_symmetry_and_psd(self) -> None:
        result = construct_scatter_matrices(self.fixture(), alpha=0.4)
        matrices = (
            result.global_scatter,
            result.local_scatter,
            result.total_scatter,
            result.regularized_scatter,
        )

        for matrix in matrices:
            with self.subTest(matrix=matrix):
                self.assertEqual(matrix.shape, (2, 2))
                self.assertTrue(np.all(np.isfinite(matrix)))
                self.assertTrue(np.allclose(matrix, matrix.T))
                self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-10)

    def test_invalid_membership_shapes_and_values_raise_value_error(self) -> None:
        invalid_cases = (
            ([0.5, 0.5], "two-dimensional"),
            (np.empty((0, 2)), "sample row"),
            (np.empty((2, 0)), "anchor column"),
            (np.array([[np.nan, 0.0]]), "finite"),
            (np.array([[np.inf, 0.0]]), "finite"),
            (np.array([[1.1, -0.1]]), "non-negative"),
            (np.array([[0.2, 0.2]]), "sum approximately"),
        )
        for Z, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_memberships(Z)

    def test_dense_graph_memory_guard_is_explicit(self) -> None:
        Z = np.ones((10, 1), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, r"O\(n\^2\).*max_graph_bytes"):
            construct_graph_laplacian(Z, max_graph_bytes=100)

    def test_input_is_not_mutated(self) -> None:
        Z = self.fixture()
        before = Z.copy()
        construct_scatter_matrices(Z, alpha=0.5)
        self.assertTrue(np.array_equal(Z, before))

    def test_model_integration_preserves_anchor_state_and_stops_before_solver(
        self,
    ) -> None:
        X = np.array(
            [[0.0, 0.0], [0.2, 0.1], [4.9, 5.0], [5.1, 4.8]],
            dtype=np.float64,
        )
        model = NFSTModel(n_anchors=2, random_state=4)
        model.fit_anchor_mapping(X)
        anchors_before = model.anchors_.copy()
        sigma_before = model.sigma_
        Z_before = model.Z_train_.copy()

        model.construct_scatter_from_anchor_mapping()

        self.assertTrue(np.array_equal(model.anchors_, anchors_before))
        self.assertEqual(model.sigma_, sigma_before)
        self.assertTrue(np.array_equal(model.Z_train_, Z_before))
        self.assertTrue(model.scatter_fitted_)
        self.assertFalse(model.is_fitted_)
        self.assertIsNone(model.projection_)
        self.assertEqual(model.assignments_.shape, (4,))
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            model.decision_function(X)

    def test_model_pipeline_executes_anchor_mapping_then_scatter(self) -> None:
        X = np.array(
            [[0.0, 0.0], [0.1, 0.2], [4.8, 5.0], [5.0, 4.9]],
            dtype=np.float64,
        )
        before = X.copy()
        model = NFSTModel(n_anchors=2, random_state=7)
        model.fit_scatter_construction(X)

        self.assertTrue(model.anchor_mapping_fitted_)
        self.assertTrue(model.scatter_fitted_)
        self.assertFalse(model.is_fitted_)
        self.assertEqual(model.global_scatter_.shape, (2, 2))
        self.assertEqual(model.regularized_scatter_.shape, (2, 2))
        self.assertTrue(np.array_equal(X, before))


if __name__ == "__main__":
    unittest.main()
