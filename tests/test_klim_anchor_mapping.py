"""Focused tests for KLIM Stage 5.2 anchor mapping."""

import unittest

import numpy as np

from adbench.baseline.KLIM.anchor_mapping import AnchorMapping
from adbench.baseline.KLIM.model import KLIMModel


class AnchorMappingTests(unittest.TestCase):
    @staticmethod
    def fixture() -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.2, 0.1],
                [4.8, 5.0],
                [5.0, 4.9],
                [5.2, 5.1],
            ],
            dtype=np.float64,
        )

    def test_anchor_and_membership_shapes_and_invariants(self) -> None:
        X = self.fixture()
        mapping = AnchorMapping(n_anchors=2, random_state=7, batch_size=4)
        Z = mapping.fit_transform_anchor_space(X)

        self.assertEqual(mapping.anchors_.shape, (2, 2))
        self.assertEqual(mapping.anchors_.dtype, np.float64)
        self.assertEqual(Z.shape, (6, 2))
        self.assertTrue(np.allclose(Z.sum(axis=1), 1.0))
        self.assertTrue(np.all(np.isfinite(Z)))
        self.assertTrue(np.all(Z >= 0.0))

    def test_same_seed_and_data_are_deterministic(self) -> None:
        X = self.fixture()
        first = AnchorMapping(2, random_state=3).fit_anchor_mapping(X)
        second = AnchorMapping(2, random_state=3).fit_anchor_mapping(X)

        self.assertTrue(np.array_equal(first.anchors_, second.anchors_))
        self.assertTrue(
            np.array_equal(
                first.transform_anchor_space(X), second.transform_anchor_space(X)
            )
        )

    def test_transform_does_not_change_fitted_state(self) -> None:
        mapping = AnchorMapping(2, random_state=5).fit_anchor_mapping(self.fixture())
        anchors_before = mapping.anchors_.copy()
        sigma_before = mapping.sigma_
        diagnostics_before = dict(mapping.anchor_diagnostics_)

        mapping.transform_anchor_space(np.array([[0.5, 0.5], [5.5, 5.5]]))

        self.assertTrue(np.array_equal(mapping.anchors_, anchors_before))
        self.assertEqual(mapping.sigma_, sigma_before)
        self.assertEqual(mapping.anchor_diagnostics_, diagnostics_before)

    def test_explicit_sigma_is_preserved_and_zero_distance_similarity_is_one(self) -> None:
        mapping = AnchorMapping(2, sigma=2.75, random_state=2).fit_anchor_mapping(
            self.fixture()
        )
        similarity = mapping.compute_similarity(mapping.anchors_[:1])
        displaced = mapping.anchors_[:1].copy()
        displaced[0, 0] += 1.0
        displaced_similarity = mapping.compute_similarity(displaced)

        self.assertEqual(mapping.sigma_, 2.75)
        self.assertEqual(similarity[0, 0], 1.0)
        self.assertAlmostEqual(
            displaced_similarity[0, 0], np.exp(-1.0 / mapping.sigma_)
        )

    def test_automatic_sigma_is_positive_and_finite(self) -> None:
        mapping = AnchorMapping(2, sigma=None, random_state=4).fit_anchor_mapping(
            self.fixture()
        )

        self.assertGreater(mapping.sigma_, 0.0)
        self.assertTrue(np.isfinite(mapping.sigma_))
        self.assertEqual(
            mapping.anchor_diagnostics_["sigma_source"],
            "median_positive_nearest_distance",
        )

    def test_duplicate_samples_are_deterministic(self) -> None:
        X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
        first = AnchorMapping(3, random_state=9).fit_anchor_mapping(X)
        second = AnchorMapping(3, random_state=9).fit_anchor_mapping(X)

        self.assertTrue(np.array_equal(first.anchors_, second.anchors_))
        self.assertGreaterEqual(first.anchor_diagnostics_["duplicate_anchor_count"], 1)

    def test_identical_samples_use_positive_epsilon_fallback(self) -> None:
        X = np.full((4, 3), 6.0)
        mapping = AnchorMapping(2, random_state=1).fit_anchor_mapping(X)
        Z = mapping.transform_anchor_space(X)

        self.assertEqual(mapping.sigma_, np.finfo(np.float64).eps)
        self.assertTrue(mapping.anchor_diagnostics_["sigma_fallback_used"])
        self.assertEqual(
            mapping.anchor_diagnostics_["sigma_source"],
            "machine_epsilon_fallback",
        )
        self.assertTrue(np.allclose(Z.sum(axis=1), 1.0))

    def test_extreme_distances_remain_finite(self) -> None:
        X = np.array([[-1e150, -1e150], [1e150, 1e150], [0.0, 0.0]])
        mapping = AnchorMapping(2, sigma=1e-12, random_state=6).fit_anchor_mapping(X)
        Z = mapping.transform_anchor_space(
            np.array([[5e149, -5e149], [-9e149, 9e149]])
        )

        self.assertTrue(np.all(np.isfinite(Z)))
        self.assertTrue(np.allclose(Z.sum(axis=1), 1.0))

    def test_invalid_sigma_raises_value_error(self) -> None:
        for sigma in (0.0, -1.0, np.inf, np.nan, "invalid"):
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(ValueError, "sigma"):
                    AnchorMapping(2, sigma=sigma)

    def test_too_many_anchors_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_anchors.*n_samples"):
            AnchorMapping(3).fit_anchor_mapping(np.ones((2, 2)))

    def test_non_2d_and_non_numeric_inputs_raise_value_error(self) -> None:
        for X in ([1.0, 2.0], 1.0, [["a", "b"]]):
            with self.subTest(X=X):
                with self.assertRaisesRegex(ValueError, "two-dimensional numeric"):
                    AnchorMapping(1).fit_anchor_mapping(X)

    def test_empty_rows_and_features_raise_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one row"):
            AnchorMapping(1).fit_anchor_mapping(np.empty((0, 2)))
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            AnchorMapping(1).fit_anchor_mapping(np.empty((2, 0)))

    def test_nan_and_infinity_raise_value_error(self) -> None:
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                X = np.array([[0.0, invalid], [1.0, 1.0]])
                with self.assertRaisesRegex(ValueError, "finite"):
                    AnchorMapping(1).fit_anchor_mapping(X)

    def test_feature_count_mismatch_raises_value_error(self) -> None:
        mapping = AnchorMapping(2, random_state=2).fit_anchor_mapping(self.fixture())
        with self.assertRaisesRegex(ValueError, "must have 2 features"):
            mapping.transform_anchor_space(np.ones((2, 3)))

    def test_model_integrates_only_anchor_mapping(self) -> None:
        model = KLIMModel(n_anchors=2, random_state=8)
        model.fit_anchor_mapping(self.fixture())

        self.assertTrue(model.anchor_mapping_fitted_)
        self.assertFalse(model.is_fitted_)
        self.assertEqual(model.anchors_.shape, (2, 2))
        self.assertEqual(model.Z_train_.shape, (6, 2))
        with self.assertRaisesRegex(NotImplementedError, "scoring is not implemented"):
            model.decision_function(self.fixture())


if __name__ == "__main__":
    unittest.main()
