"""Import and interface tests for the NFST detector."""

import inspect
import unittest
from unittest.mock import Mock

import numpy as np

from adbench.baseline.NFST import NFST, NFSTModel
from adbench.baseline.NFST import __all__ as nfst_exports


class NFSTModelInterfaceTests(unittest.TestCase):
    def test_public_classes_import(self) -> None:
        self.assertEqual(set(nfst_exports), {"NFST", "NFSTModel"})
        self.assertTrue(inspect.isclass(NFSTModel))
        self.assertTrue(inspect.isclass(NFST))

    def test_constructor_stores_final_parameters_and_initializes_state(self) -> None:
        model = NFSTModel(
            n_anchors=8, sigma=2.5, alpha=0.25, random_state=7,
            batch_size=64, tol=1e-6, rank_tolerance=1e-9,
            null_tolerance=1e-7, max_components=2,
            selection_mode="smallest", score_batch_size=16,
        )
        self.assertEqual(model.n_anchors, 8)
        self.assertEqual(model.sigma, 2.5)
        self.assertEqual(model.alpha, 0.25)
        self.assertEqual(model.random_state, 7)
        self.assertEqual(model.batch_size, 64)
        self.assertEqual(model.tol, 1e-6)
        self.assertEqual(model.rank_tolerance, 1e-9)
        self.assertEqual(model.null_tolerance, 1e-7)
        self.assertEqual(model.max_components, 2)
        self.assertEqual(model.selection_mode, "smallest")
        self.assertEqual(model.score_batch_size, 16)
        self.assertFalse(model.is_fitted_)
        self.assertIsNone(model.anchors_)
        self.assertIsNone(model.projection_)
        self.assertIsNone(model.base_points_)
        self.assertEqual(model.diagnostics_, {})

    def test_invalid_constructor_parameters_raise_clear_errors(self) -> None:
        invalid_cases = (
            ({"n_anchors": 0}, ValueError), ({"n_anchors": True}, TypeError),
            ({"sigma": 0.0}, ValueError), ({"sigma": float("nan")}, ValueError),
            ({"alpha": 1.1}, ValueError), ({"random_state": -1}, ValueError),
            ({"batch_size": 0}, ValueError), ({"tol": 0.0}, ValueError),
            ({"rank_tolerance": 0.0}, ValueError),
            ({"null_tolerance": 0.0}, ValueError),
            ({"max_components": 0}, ValueError),
            ({"selection_mode": "invalid"}, ValueError),
            ({"score_batch_size": 0}, ValueError),
        )
        for kwargs, error_type in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(error_type, next(iter(kwargs))):
                    NFSTModel(**kwargs)

    def test_fitted_state_guard_rejects_unfitted_model(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            NFSTModel()._check_is_fitted()


class NFSTAdapterInterfaceTests(unittest.TestCase):
    def test_adapter_stores_compatibility_parameters(self) -> None:
        detector = NFST(seed=11, model_name="NFST", n_anchors=4, alpha=0.75)
        self.assertEqual(detector.seed, 11)
        self.assertEqual(detector.model_name, "NFST")
        self.assertIsInstance(detector.model, NFSTModel)
        self.assertEqual(detector.model.random_state, 11)

    def test_adapter_constructor_rejects_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "random_state"):
            NFST(seed=-1)
        with self.assertRaisesRegex(TypeError, "model_name"):
            NFST(seed=1, model_name=123)

    def test_adapter_is_thin_and_ignores_labels_for_training_policy(self) -> None:
        detector = NFST(seed=3, n_anchors=2)
        detector.model.fit = Mock(return_value=detector.model)
        detector.model.decision_function = Mock(
            return_value=np.array([0.25], dtype=np.float64)
        )
        X_train = np.array([[0.0], [1.0]])
        X_test = np.array([[2.0]])
        self.assertIs(detector.fit(X_train=X_train, y_train=np.array([0, 1])), detector)
        scores = detector.predict_score(X_test=X_test)
        detector.model.fit.assert_called_once_with(X_train)
        detector.model.decision_function.assert_called_once_with(X_test)
        self.assertTrue(np.array_equal(scores, np.array([0.25])))

    def test_adapter_method_parameter_names_match_contract(self) -> None:
        self.assertEqual(
            list(inspect.signature(NFST.__init__).parameters)[:3],
            ["self", "seed", "model_name"],
        )
        self.assertEqual(
            list(inspect.signature(NFST.fit).parameters),
            ["self", "X_train", "y_train"],
        )
        self.assertEqual(
            list(inspect.signature(NFST.predict_score).parameters),
            ["self", "X_test"],
        )


if __name__ == "__main__":
    unittest.main()
