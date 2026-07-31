"""Topological manifold-partitioned one-class NFST detector."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from adbench.baseline.NFST.solver import SolverResult, solve_nfst_projection

from .partition import PartitionResult, spectral_partition, validate_features
from .scatter import TopologicalScatterResult, construct_topological_scatter
from .scoring import build_projected_centroids, score_samples


def _positive_integer(name: str, value: int) -> int:
    """Optional validation: kiểm tra tham số là số nguyên dương."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _positive_real(name: str, value: float) -> float:
    """Optional validation: kiểm tra tham số là số thực dương hữu hạn."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


class TMPNFSTModel:
    """Điều phối spectral pseudo-class, NFST projection và anomaly scoring."""

    def __init__(
        self,
        n_neighbors: int = 10,
        n_components: int = 3,
        sigma: float | None = None,
        random_state: int | None = None,
        n_init: int = 10,
        eigen_tolerance: float = 1e-8,
        eigen_maxiter: int | None = None,
        rank_tolerance: float = 1e-10,
        null_tolerance: float = 1e-8,
        max_components: int | None = None,
        selection_mode: Literal["null", "smallest"] = "smallest",
        score_batch_size: int | None = None,
    ) -> None:
        """Lưu và validate cấu hình; chưa xử lý dữ liệu tại bước khởi tạo."""
        self.n_neighbors = _positive_integer("n_neighbors", n_neighbors)
        self.n_components = _positive_integer("n_components", n_components)
        self.sigma = None if sigma is None else _positive_real("sigma", sigma)
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise TypeError("random_state must be an integer or None.")
        if random_state is not None and random_state < 0:
            raise ValueError("random_state must be non-negative.")
        self.random_state = None if random_state is None else int(random_state)
        self.n_init = _positive_integer("n_init", n_init)
        self.eigen_tolerance = _positive_real("eigen_tolerance", eigen_tolerance)
        self.eigen_maxiter = (
            None
            if eigen_maxiter is None
            else _positive_integer("eigen_maxiter", eigen_maxiter)
        )
        self.rank_tolerance = _positive_real("rank_tolerance", rank_tolerance)
        self.null_tolerance = _positive_real("null_tolerance", null_tolerance)
        self.max_components = (
            None
            if max_components is None
            else _positive_integer("max_components", max_components)
        )
        if selection_mode not in {"null", "smallest"}:
            raise ValueError("selection_mode must be 'null' or 'smallest'.")
        self.selection_mode = selection_mode
        self.score_batch_size = (
            None
            if score_batch_size is None
            else _positive_integer("score_batch_size", score_batch_size)
        )
        self._reset_fitted_state()

    def _reset_fitted_state(self) -> None:
        """Xóa toàn bộ graph, labels, projection và centroid đã học."""
        self.n_features_in_: int | None = None
        self.labels_: NDArray[np.int64] | None = None
        self.lifetimes_: NDArray[np.int64] | None = None
        self.embedding_: NDArray[np.float64] | None = None
        self.graph_eigenvalues_: NDArray[np.float64] | None = None
        self.graph_sigma_: float | None = None
        self.within_scatter_: NDArray[np.float64] | None = None
        self.between_scatter_: NDArray[np.float64] | None = None
        self.total_scatter_: NDArray[np.float64] | None = None
        self.projection_: NDArray[np.float64] | None = None
        self.centroids_: NDArray[np.float64] | None = None
        self.projected_training_: NDArray[np.float64] | None = None
        self.diagnostics_: dict[str, Any] = {}
        self.is_fitted_ = False

    def _check_is_fitted(self) -> None:
        """Optional validation: chặn predict khi model chưa fit hoàn chỉnh."""
        if not self.is_fitted_:
            raise RuntimeError("TMPNFSTModel is not fitted.")

    def fit(self, X_normal: Any) -> TMPNFSTModel:
        """Nhận X train (n,d), học TMPNFST và trả self đã fit.

        Luồng: spectral partition -> scatter -> NFST solver -> centroid.
        Chất lượng chịu ảnh hưởng chính bởi graph/component parameters; khả năng
        pass còn phụ thuộc eigensolver và tolerance của NFST solver.
        """
        self._reset_fitted_state()
        features = validate_features(X_normal, minimum_samples=3)

        # Phase 1: xây graph và tạo pseudo-label hoàn toàn từ X, không dùng y.
        partition: PartitionResult = spectral_partition(
            features,
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            sigma=self.sigma,
            random_state=self.random_state,
            n_init=self.n_init,
            eigen_tolerance=self.eigen_tolerance,
            eigen_maxiter=self.eigen_maxiter,
        )
        # Phase 2: dùng pseudo-label như class tạm để tính Sw, Sb và St.
        scatter: TopologicalScatterResult = construct_topological_scatter(
            features, partition.labels
        )
        # Phase 3: smallest mode cần số chiều rõ ràng; mặc định dùng c-1.
        projection_cap = self.max_components
        if self.selection_mode == "smallest" and projection_cap is None:
            projection_cap = max(1, self.n_components - 1)
        # Solver dùng St làm total scatter và Sw làm within scatter.
        solver: SolverResult = solve_nfst_projection(
            scatter.total_scatter,
            scatter.within_scatter,
            rank_tolerance=self.rank_tolerance,
            null_tolerance=self.null_tolerance,
            max_components=projection_cap,
            selection_mode=self.selection_mode,
        )
        # Phase 4: chiếu train và tạo một centroid cho mỗi pseudo-class.
        projected, centroids = build_projected_centroids(
            features, solver.projection, partition.labels
        )

        # Chỉ commit sau khi tất cả phase hoàn tất để tránh half-fitted state.
        self.n_features_in_ = features.shape[1]
        self.labels_ = partition.labels
        self.lifetimes_ = partition.lifetimes
        self.embedding_ = partition.embedding
        self.graph_eigenvalues_ = partition.eigenvalues
        self.graph_sigma_ = partition.sigma
        self.within_scatter_ = scatter.within_scatter
        self.between_scatter_ = scatter.between_scatter
        self.total_scatter_ = scatter.total_scatter
        self.projection_ = solver.projection
        self.centroids_ = centroids
        self.projected_training_ = projected
        self.diagnostics_ = {
            "partition": dict(partition.diagnostics),
            "scatter": dict(scatter.diagnostics),
            "solver": dict(solver.diagnostics),
            "lifetime_used_for_partitioning": False,
        }
        self.is_fitted_ = True
        return self

    def decision_function(self, X_test: Any) -> NDArray[np.float64]:
        """Nhận X test (m,d), trả nearest-centroid anomaly score (m,)."""
        self._check_is_fitted()
        assert self.n_features_in_ is not None
        assert self.projection_ is not None
        assert self.centroids_ is not None
        features = validate_features(X_test, expected_features=self.n_features_in_)
        return score_samples(
            features,
            self.projection_,
            self.centroids_,
            batch_size=self.score_batch_size,
        )
