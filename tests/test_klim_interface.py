"""Import and interface tests for the KLIM detector skeleton."""

import inspect
import unittest

from adbench.baseline.KLIM import KLIM, KLIMModel
from adbench.baseline.KLIM import __all__ as klim_exports


class KLIMModelInterfaceTests(unittest.TestCase):
    def test_public_classes_import(self) -> None:
        self.assertEqual(set(klim_exports), {"KLIM", "KLIMModel"})
        self.assertTrue(inspect.isclass(KLIMModel))
        self.assertTrue(inspect.isclass(KLIM))

    def test_constructor_stores_parameters_and_initializes_state(self) -> None:
        model = KLIMModel(
            n_anchors=8,
            sigma=2.5,
            alpha=0.25,
            null_dim=2,
            gamma=0.1,
            random_state=7,
            batch_size=64,
            tol=1e-6,
        )

        self.assertEqual(model.n_anchors, 8)
        self.assertEqual(model.sigma, 2.5)
        self.assertEqual(model.alpha, 0.25)
        self.assertEqual(model.null_dim, 2)
        self.assertEqual(model.gamma, 0.1)
        self.assertEqual(model.random_state, 7)
        self.assertEqual(model.batch_size, 64)
        self.assertEqual(model.tol, 1e-6)
        self.assertFalse(model.is_fitted_)
        self.assertIsNone(model.anchors_)
        self.assertIsNone(model.projection_)
        self.assertIsNone(model.base_points_)
        self.assertEqual(model.diagnostics_, {})

    def test_invalid_constructor_parameters_raise_clear_errors(self) -> None:
        invalid_cases = (
            ({"n_anchors": 0}, ValueError),
            ({"n_anchors": True}, TypeError),
            ({"sigma": 0.0}, ValueError),
            ({"sigma": float("nan")}, ValueError),
            ({"alpha": 1.1}, ValueError),
            ({"null_dim": 0}, ValueError),
            ({"gamma": -0.1}, ValueError),
            ({"random_state": -1}, ValueError),
            ({"batch_size": 0}, ValueError),
            ({"tol": 0.0}, ValueError),
        )

        for kwargs, error_type in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(error_type, next(iter(kwargs))):
                    KLIMModel(**kwargs)

    def test_unimplemented_model_methods_are_explicit(self) -> None:
        model = KLIMModel()

        with self.assertRaisesRegex(NotImplementedError, "training is not implemented"):
            model.fit([[0.0]])
        with self.assertRaisesRegex(NotImplementedError, "scoring is not implemented"):
            model.decision_function([[0.0]])

    def test_fitted_state_guard_rejects_unfitted_model(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            KLIMModel()._check_is_fitted()


class KLIMAdapterInterfaceTests(unittest.TestCase):
    def test_adapter_stores_compatibility_parameters(self) -> None:
        detector = KLIM(seed=11, model_name="KLIM", n_anchors=4, alpha=0.75)

        self.assertEqual(detector.seed, 11)
        self.assertEqual(detector.model_name, "KLIM")
        self.assertIsInstance(detector.model, KLIMModel)
        self.assertEqual(detector.model.random_state, 11)
        self.assertEqual(detector.model.n_anchors, 4)
        self.assertEqual(detector.model.alpha, 0.75)

    def test_adapter_constructor_rejects_invalid_compatibility_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "random_state"):
            KLIM(seed=-1)
        with self.assertRaisesRegex(TypeError, "model_name"):
            KLIM(seed=1, model_name=123)

    def test_adapter_methods_delegate_to_unimplemented_model(self) -> None:
        detector = KLIM(seed=3)

        with self.assertRaises(NotImplementedError):
            detector.fit(X_train=[[0.0]], y_train=[0])
        with self.assertRaises(NotImplementedError):
            detector.predict_score(X_test=[[0.0]])

    def test_adapter_method_parameter_names_match_contract(self) -> None:
        self.assertEqual(
            list(inspect.signature(KLIM.__init__).parameters)[:3],
            ["self", "seed", "model_name"],
        )
        self.assertEqual(
            list(inspect.signature(KLIM.fit).parameters),
            ["self", "X_train", "y_train"],
        )
        self.assertEqual(
            list(inspect.signature(KLIM.predict_score).parameters),
            ["self", "X_test"],
        )


if __name__ == "__main__":
    unittest.main()
