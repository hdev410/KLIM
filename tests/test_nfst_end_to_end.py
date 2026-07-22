"""Standalone synthetic validation for the complete NFST detector."""

import json
import subprocess
import sys
import unittest

import numpy as np

from adbench.baseline.NFST.model import NFSTModel


def perpendicular_bisector_anomalies(model: NFSTModel, distance: float = 8.0) -> np.ndarray:
    anchors = model.anchors_
    midpoint = anchors.mean(axis=0)
    delta = anchors[1] - anchors[0]
    perpendicular = np.array([-delta[1], delta[0]], dtype=np.float64)
    perpendicular /= np.linalg.norm(perpendicular)
    return np.vstack(
        [midpoint + distance * perpendicular, midpoint - distance * perpendicular]
    )


class NFSTEndToEndTests(unittest.TestCase):
    @staticmethod
    def make_model(seed: int = 5) -> NFSTModel:
        return NFSTModel(
            n_anchors=2,
            sigma=1e-6,
            alpha=0.0,
            random_state=seed,
            null_tolerance=1e-7,
            score_batch_size=7,
        )

    def test_one_gaussian_mode_separates_controlled_distant_anomalies(self) -> None:
        rng = np.random.RandomState(12)
        X_normal = rng.normal(0.0, 0.35, size=(80, 2))
        model = self.make_model().fit(X_normal)
        anomalies = perpendicular_bisector_anomalies(model)

        normal_scores = model.decision_function(X_normal)
        anomaly_scores = model.decision_function(anomalies)

        self.assertEqual(normal_scores.shape, (80,))
        self.assertEqual(anomaly_scores.shape, (2,))
        self.assertTrue(np.all(np.isfinite(normal_scores)))
        self.assertTrue(np.all(np.isfinite(anomaly_scores)))
        self.assertGreater(anomaly_scores.mean(), normal_scores.mean())

    def test_two_separated_normal_modes_separate_distant_anomalies(self) -> None:
        rng = np.random.RandomState(21)
        X_normal = np.vstack(
            [
                rng.normal([-2.0, 0.0], 0.2, size=(40, 2)),
                rng.normal([2.0, 0.0], 0.2, size=(40, 2)),
            ]
        )
        model = self.make_model(seed=8).fit(X_normal)
        anomalies = perpendicular_bisector_anomalies(model, distance=10.0)

        normal_scores = model.decision_function(X_normal)
        anomaly_scores = model.decision_function(anomalies)

        self.assertTrue(np.all(np.isfinite(normal_scores)))
        self.assertTrue(np.all(np.isfinite(anomaly_scores)))
        self.assertGreater(anomaly_scores.min(), normal_scores.max())

    def test_repeated_fit_with_same_seed_is_reproducible(self) -> None:
        rng = np.random.RandomState(31)
        X_normal = np.vstack(
            [rng.normal([-1.5, 0.0], 0.15, size=(30, 2)),
             rng.normal([1.5, 0.0], 0.15, size=(30, 2))]
        )
        first = self.make_model(seed=9).fit(X_normal)
        second = self.make_model(seed=9).fit(X_normal)
        X_test = perpendicular_bisector_anomalies(first)

        self.assertTrue(np.array_equal(first.anchors_, second.anchors_))
        self.assertTrue(np.array_equal(first.Z_train_, second.Z_train_))
        self.assertTrue(
            np.allclose(
                first.projection_ @ first.projection_.T,
                second.projection_ @ second.projection_.T,
            )
        )
        self.assertTrue(
            np.allclose(
                first.decision_function(X_test), second.decision_function(X_test)
            )
        )

    def test_failed_refit_does_not_leave_model_ready(self) -> None:
        X_normal = np.array([[-1.0, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.0, 0.0]])
        model = self.make_model().fit(X_normal)
        self.assertTrue(model.is_fitted_)
        with self.assertRaises(ValueError):
            model.fit(np.array([[0.0, 0.0]]))
        self.assertFalse(model.is_fitted_)
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            model.decision_function(np.array([[0.0, 0.0]]))

    def test_import_does_not_load_forbidden_heavy_frameworks(self) -> None:
        code = (
            "import json,sys; import adbench.baseline.NFST; "
            "print(json.dumps([n for n in ('tensorflow','torch','keras','pyod') "
            "if n in sys.modules]))"
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
