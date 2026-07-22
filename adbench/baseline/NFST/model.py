"""Standalone Anchor-based Kernelized One-Class LPP-NFST detector."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .anchor_mapping import AnchorMapping
from .scatter import DEFAULT_MAX_GRAPH_BYTES, ScatterResult, construct_scatter_matrices
from .scoring import BasePointResult, build_subclass_base_points, score_anchor_memberships
from .solver import SolverResult, solve_nfst_projection


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return int(value)


def _validate_finite_positive_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return normalized


class NFSTModel:
    """Fit the complete standalone NFST detector from one-class training data."""

    def __init__(
        self,
        n_anchors: int = 32,
        sigma: float | None = None,
        alpha: float = 0.5,
        random_state: int | None = None,
        batch_size: int = 1024,
        tol: float = 1e-8,
        rank_tolerance: float = 1e-10,
        null_tolerance: float = 1e-8,
        max_components: int | None = None,
        selection_mode: Literal["null", "smallest"] = "null",
        max_graph_bytes: int | None = DEFAULT_MAX_GRAPH_BYTES,
        score_batch_size: int | None = None,
    ) -> None:
        self.n_anchors = _validate_positive_integer("n_anchors", n_anchors)
        if sigma is None:
            self.sigma = None
        else:
            self.sigma = _validate_finite_positive_real("sigma", sigma)
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a real number.")
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be finite and between zero and one, inclusive.")
        if random_state is None:
            self.random_state = None
        elif isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise TypeError("random_state must be an integer or None.")
        elif random_state < 0:
            raise ValueError("random_state must be non-negative.")
        else:
            self.random_state = int(random_state)
        self.batch_size = _validate_positive_integer("batch_size", batch_size)
        self.tol = _validate_finite_positive_real("tol", tol)
        self.rank_tolerance = _validate_finite_positive_real(
            "rank_tolerance", rank_tolerance
        )
        self.null_tolerance = _validate_finite_positive_real(
            "null_tolerance", null_tolerance
        )
        if max_components is None:
            self.max_components = None
        else:
            self.max_components = _validate_positive_integer(
                "max_components", max_components
            )
        if selection_mode not in {"null", "smallest"}:
            raise ValueError("selection_mode must be 'null' or 'smallest'.")
        if selection_mode == "smallest" and self.max_components is None:
            raise ValueError(
                "max_components is required when selection_mode='smallest'."
            )
        self.selection_mode = selection_mode
        if max_graph_bytes is None:
            self.max_graph_bytes = None
        else:
            self.max_graph_bytes = _validate_positive_integer(
                "max_graph_bytes", max_graph_bytes
            )
        if score_batch_size is None:
            self.score_batch_size = None
        else:
            self.score_batch_size = _validate_positive_integer(
                "score_batch_size", score_batch_size
            )
        self._reset_fitted_state()

    def _reset_fitted_state(self) -> None:
        self.scaler_: Any | None = None
        self.anchors_: NDArray[np.float64] | None = None
        self.sigma_: float | None = None
        self.n_features_in_: int | None = None
        self.Z_train_: NDArray[np.float64] | None = None
        self.assignments_: NDArray[np.int64] | None = None
        self.global_scatter_: NDArray[np.float64] | None = None
        self.local_scatter_: NDArray[np.float64] | None = None
        self.total_scatter_: NDArray[np.float64] | None = None
        self.regularized_scatter_: NDArray[np.float64] | None = None
        self.range_basis_: NDArray[np.float64] | None = None
        self.projection_: NDArray[np.float64] | None = None
        self.eigenvalues_: NDArray[np.float64] | None = None
        self.selected_eigenvalues_: NDArray[np.float64] | None = None
        self.base_points_: NDArray[np.float64] | None = None
        self.valid_cluster_ids_: NDArray[np.int64] | None = None
        self.empty_cluster_ids_: NDArray[np.int64] | None = None
        self.subclass_counts_: NDArray[np.int64] | None = None
        self.diagnostics_: dict[str, Any] = {}
        self.anchor_diagnostics_: dict[str, Any] = {}
        self.scatter_diagnostics_: dict[str, Any] = {}
        self.solver_diagnostics_: dict[str, Any] = {}
        self.scoring_diagnostics_: dict[str, Any] = {}
        self.anchor_mapping_fitted_ = False
        self.scatter_fitted_ = False
        self.solver_fitted_ = False
        self.base_points_fitted_ = False
        self._anchor_mapping: AnchorMapping | None = None
        self.is_fitted_ = False

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("NFSTModel is not fitted.")

    def _new_anchor_mapping(self) -> AnchorMapping:
        return AnchorMapping(
            n_anchors=self.n_anchors,
            sigma=self.sigma,
            random_state=self.random_state,
            batch_size=self.batch_size,
        )

    def fit(self, X_normal: Any) -> NFSTModel:
        """Fit every standalone NFST stage and commit state atomically."""
        self._reset_fitted_state()
        mapping = self._new_anchor_mapping()
        Z_train = mapping.fit_transform_anchor_space(X_normal)
        scatter: ScatterResult = construct_scatter_matrices(
            Z_train,
            self.alpha,
            max_graph_bytes=self.max_graph_bytes,
            tolerance=self.tol,
        )
        solver: SolverResult = solve_nfst_projection(
            scatter.total_scatter,
            scatter.regularized_scatter,
            rank_tolerance=self.rank_tolerance,
            null_tolerance=self.null_tolerance,
            max_components=self.max_components,
            selection_mode=self.selection_mode,
        )
        base_points: BasePointResult = build_subclass_base_points(
            Z_train, solver.projection, scatter.assignments
        )

        self._anchor_mapping = mapping
        self.anchors_ = mapping.anchors_
        self.sigma_ = mapping.sigma_
        self.n_features_in_ = mapping.n_features_in_
        self.Z_train_ = Z_train
        self.assignments_ = scatter.assignments
        self.global_scatter_ = scatter.global_scatter
        self.local_scatter_ = scatter.local_scatter
        self.total_scatter_ = scatter.total_scatter
        self.regularized_scatter_ = scatter.regularized_scatter
        self.range_basis_ = solver.range_basis
        self.projection_ = solver.projection
        self.eigenvalues_ = solver.eigenvalues
        self.selected_eigenvalues_ = solver.selected_eigenvalues
        self.base_points_ = base_points.base_points
        self.valid_cluster_ids_ = base_points.valid_cluster_ids
        self.empty_cluster_ids_ = base_points.empty_cluster_ids
        self.subclass_counts_ = base_points.subclass_counts
        self.anchor_diagnostics_ = dict(mapping.anchor_diagnostics_)
        self.scatter_diagnostics_ = dict(scatter.diagnostics)
        self.solver_diagnostics_ = dict(solver.diagnostics)
        self.scoring_diagnostics_ = dict(base_points.diagnostics)
        self.diagnostics_ = {
            "anchor_mapping": self.anchor_diagnostics_,
            "scatter": self.scatter_diagnostics_,
            "solver": self.solver_diagnostics_,
            "scoring": self.scoring_diagnostics_,
        }
        self.anchor_mapping_fitted_ = True
        self.scatter_fitted_ = True
        self.solver_fitted_ = True
        self.base_points_fitted_ = True
        self.is_fitted_ = True
        return self

    def fit_anchor_mapping(self, X_normal: Any) -> NFSTModel:
        self._reset_fitted_state()
        mapping = self._new_anchor_mapping()
        Z_train = mapping.fit_transform_anchor_space(X_normal)
        self._anchor_mapping = mapping
        self.anchors_ = mapping.anchors_
        self.sigma_ = mapping.sigma_
        self.n_features_in_ = mapping.n_features_in_
        self.anchor_diagnostics_ = dict(mapping.anchor_diagnostics_)
        self.Z_train_ = Z_train
        self.anchor_mapping_fitted_ = True
        return self

    def transform_anchor_space(self, X: Any) -> NDArray[np.float64]:
        if not self.anchor_mapping_fitted_ or self._anchor_mapping is None:
            raise ValueError("Anchor mapping is not fitted.")
        return self._anchor_mapping.transform_anchor_space(X)

    def construct_scatter_from_anchor_mapping(self) -> NFSTModel:
        if not self.anchor_mapping_fitted_ or self.Z_train_ is None:
            raise ValueError("Anchor mapping must be fitted before scatter construction.")
        scatter = construct_scatter_matrices(
            self.Z_train_,
            self.alpha,
            max_graph_bytes=self.max_graph_bytes,
            tolerance=self.tol,
        )
        self.assignments_ = scatter.assignments
        self.global_scatter_ = scatter.global_scatter
        self.local_scatter_ = scatter.local_scatter
        self.total_scatter_ = scatter.total_scatter
        self.regularized_scatter_ = scatter.regularized_scatter
        self.scatter_diagnostics_ = dict(scatter.diagnostics)
        self.scatter_fitted_ = True
        return self

    def fit_scatter_construction(self, X_normal: Any) -> NFSTModel:
        self.fit_anchor_mapping(X_normal)
        return self.construct_scatter_from_anchor_mapping()

    def decision_function(self, X_test: Any) -> NDArray[np.float64]:
        """Score test samples without refitting any training component."""
        self._check_is_fitted()
        assert self._anchor_mapping is not None
        assert self.projection_ is not None
        assert self.base_points_ is not None
        # Suy luận chỉ tái sử dụng trạng thái đã fit, không fit lại.
        Z_test = self._anchor_mapping.transform_anchor_space(X_test)
        return score_anchor_memberships(
            Z_test,
            self.projection_,
            self.base_points_,
            batch_size=self.score_batch_size,
        )
