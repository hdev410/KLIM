"""Focused tests for NFST base points and anomaly scoring."""

import unittest

import numpy as np

from adbench.baseline.NFST.model import NFSTModel
from adbench.baseline.NFST.scoring import (
    build_subclass_base_points,
    score_anchor_memberships,
    score_projected_samples,
)


class NFSTScoringTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = np.array(
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0],
             [0.0, 1.0, 0.0], [0.2, 0.8, 0.0]],
            dtype=np.float64,
        )
        projection = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        assignments = np.array([0, 0, 1, 1], dtype=np.int64)
        return Z, projection, assignments

    def test_projected_training_base_points_empty_clusters_and_counts(self) -> None:
        Z, projection, assignments = self.fixture()
        result = build_subclass_base_points(Z, projection, assignments)

        self.assertEqual(result.projected_training.shape, (4, 2))
        self.assertEqual(result.base_points.shape, (2, 2))
        self.assertTrue(np.array_equal(result.valid_cluster_ids, [0, 1]))
        self.assertTrue(np.array_equal(result.empty_cluster_ids, [2]))
        self.assertTrue(np.array_equal(result.subclass_counts, [2, 2, 0]))
        self.assertTrue(
            np.allclose(result.base_points, [[0.9, 0.1], [0.1, 0.9]])
        )

    def test_point_at_base_point_has_zero_score(self) -> None:
        Z, projection, assignments = self.fixture()
        result = build_subclass_base_points(Z, projection, assignments)
        scores = score_projected_samples(result.base_points, result.base_points)
        self.assertTrue(np.allclose(scores, 0.0))

    def test_score_shape_finiteness_nonnegativity_and_ordering(self) -> None:
        base_points = np.array([[0.0, 0.0], [1.0, 1.0]])
        samples = np.array([[0.1, 0.1], [4.0, 4.0]])
        scores = score_projected_samples(samples, base_points)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(scores.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertGreater(scores[1], scores[0])

    def test_batched_and_unbatched_scoring_agree(self) -> None:
        rng = np.random.RandomState(4)
        samples = rng.normal(size=(11, 3))
        base_points = rng.normal(size=(4, 3))
        unbatched = score_projected_samples(samples, base_points)
        batched = score_projected_samples(samples, base_points, batch_size=3)
        self.assertTrue(np.allclose(unbatched, batched))

    def test_anchor_membership_scoring_projects_before_distance(self) -> None:
        Z, projection, assignments = self.fixture()
        result = build_subclass_base_points(Z, projection, assignments)
        scores = score_anchor_memberships(
            Z, projection, result.base_points, batch_size=2
        )
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_prediction_before_fit_and_feature_mismatch_raise(self) -> None:
        model = NFSTModel(n_anchors=2)
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            model.decision_function(np.ones((1, 2)))

        X = np.array([[-1.0, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.0, 0.0]])
        model = NFSTModel(n_anchors=2, sigma=1e-4, alpha=0.0, random_state=2)
        model.fit(X)
        with self.assertRaisesRegex(ValueError, "must have 2 features"):
            model.decision_function(np.ones((1, 3)))

    def test_prediction_does_not_mutate_training_state(self) -> None:
        X = np.array([[-1.0, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.0, 0.0]])
        model = NFSTModel(n_anchors=2, sigma=1e-4, alpha=0.0, random_state=3)
        model.fit(X)
        anchors = model.anchors_.copy()
        sigma = model.sigma_
        projection = model.projection_.copy()
        base_points = model.base_points_.copy()

        model.decision_function(np.array([[0.0, 3.0], [0.0, -3.0]]))

        self.assertTrue(np.array_equal(model.anchors_, anchors))
        self.assertEqual(model.sigma_, sigma)
        self.assertTrue(np.array_equal(model.projection_, projection))
        self.assertTrue(np.array_equal(model.base_points_, base_points))


if __name__ == "__main__":
    unittest.main()
