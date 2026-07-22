"""Focused tests for the two-stage NFST solver."""

import inspect
import unittest

import numpy as np

import adbench.baseline.NFST.solver as solver_module
from adbench.baseline.NFST.solver import solve_nfst_projection


def projector(matrix: np.ndarray) -> np.ndarray:
    return matrix @ matrix.T


class NFSTSolverTests(unittest.TestCase):
    def test_reference_projection_and_diagnostics(self) -> None:
        total = np.diag([3.0, 2.0, 0.0])
        regularized = np.diag([0.0, 1.0, 5.0])
        result = solve_nfst_projection(total, regularized)

        self.assertEqual(result.projection.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(result.projection)))
        self.assertEqual(result.rank, 2)
        self.assertEqual(result.nullity, 1)
        self.assertTrue(
            np.allclose(result.range_basis.T @ result.range_basis, np.eye(2))
        )
        self.assertTrue(np.all(np.diff(result.eigenvalues) >= 0.0))
        self.assertTrue(
            np.all(np.abs(result.selected_eigenvalues)
                   <= result.diagnostics["null_threshold"])
        )
        self.assertLessEqual(
            result.diagnostics["projected_null_residual_max"], 1e-10
        )
        self.assertLessEqual(result.diagnostics["range_residual"], 1e-10)
        self.assertTrue(
            np.allclose(projector(result.projection), np.diag([1.0, 0.0, 0.0]))
        )

    def test_rank_zero_total_scatter_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank zero"):
            solve_nfst_projection(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_zero_nullity_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "nullity zero"):
            solve_nfst_projection(np.eye(2), np.diag([1.0, 2.0]))

    def test_smallest_mode_selects_lowest_non_null_directions(self) -> None:
        result = solve_nfst_projection(
            np.eye(3),
            np.diag([1.0, 2.0, 3.0]),
            selection_mode="smallest",
            max_components=2,
        )
        self.assertEqual(result.nullity, 0)
        self.assertEqual(result.projection.shape, (3, 2))
        self.assertTrue(np.allclose(result.selected_eigenvalues, [1.0, 2.0]))
        self.assertEqual(result.diagnostics["selection_mode"], "smallest")

    def test_smallest_mode_requires_an_explicit_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_components"):
            solve_nfst_projection(
                np.eye(2), np.eye(2), selection_mode="smallest"
            )

    def test_max_components_caps_only_valid_null_directions(self) -> None:
        result = solve_nfst_projection(
            np.diag([3.0, 2.0, 1.0]),
            np.diag([0.0, 0.0, 1.0]),
            max_components=1,
        )
        self.assertEqual(result.nullity, 2)
        self.assertEqual(result.projection.shape, (3, 1))
        self.assertEqual(result.selected_eigenvalues.shape, (1,))
        self.assertLessEqual(abs(result.selected_eigenvalues[0]), 1e-8)

    def test_singular_and_nearly_singular_total_scatter_follow_tolerance(self) -> None:
        total = np.diag([1.0, 1e-12, 0.0])
        regularized = np.diag([0.0, 4.0, 2.0])
        coarse = solve_nfst_projection(
            total, regularized, rank_tolerance=1e-10
        )
        self.assertEqual(coarse.rank, 1)
        fine = solve_nfst_projection(
            total, regularized, rank_tolerance=1e-14
        )
        self.assertEqual(fine.rank, 2)
        self.assertEqual(fine.nullity, 1)

    def test_repeated_runs_produce_equivalent_subspaces(self) -> None:
        total = np.diag([4.0, 3.0, 2.0, 0.0])
        regularized = np.diag([0.0, 0.0, 1.0, 7.0])
        first = solve_nfst_projection(total, regularized)
        second = solve_nfst_projection(total, regularized)
        self.assertTrue(
            np.allclose(projector(first.projection), projector(second.projection))
        )

    def test_subspace_comparison_is_invariant_to_eigenvector_sign(self) -> None:
        result = solve_nfst_projection(
            np.diag([2.0, 1.0, 0.0]), np.diag([0.0, 1.0, 3.0])
        )
        self.assertTrue(
            np.allclose(projector(result.projection), projector(-result.projection))
        )

    def test_solver_source_contains_no_explicit_inverse(self) -> None:
        source = inspect.getsource(solver_module)
        self.assertNotIn("np.linalg.inv", source)
        self.assertNotIn("scipy.linalg.inv", source)


if __name__ == "__main__":
    unittest.main()
