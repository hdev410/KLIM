"""Anchor generation and normalized Gaussian memberships for NFST."""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning

FloatArray = NDArray[np.float64]


def _validate_positive_integer(name: str, value: int) -> int:
    """Optional validation: kiểm tra một tham số là số nguyên dương."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return int(value)


def _validate_sigma(sigma: float | None) -> float | None:
    """Optional validation: chuẩn hóa sigma thành None hoặc số thực dương."""
    if sigma is None:
        return None
    if isinstance(sigma, bool) or not isinstance(sigma, Real):
        raise ValueError("sigma must be None or a finite positive float.")
    resolved = float(sigma)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("sigma must be None or a finite positive float.")
    return resolved


def _validate_input(X: Any, expected_features: int | None = None) -> FloatArray:
    """Optional validation: đổi X thành float64 2D hữu hạn, đúng số feature."""
    try:
        raw = np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a two-dimensional numeric array.") from exc

    if raw.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array.")
    if raw.shape[0] < 1:
        raise ValueError("X must contain at least one row.")
    if raw.shape[1] < 1:
        raise ValueError("X must contain at least one feature.")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("X must be a two-dimensional numeric array.")

    validated = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(validated)):
        raise ValueError("X must contain only finite values.")
    if expected_features is not None and validated.shape[1] != expected_features:
        raise ValueError(
            f"X must have {expected_features} features; got {validated.shape[1]}."
        )
    return validated


def _squared_euclidean_distances(X: FloatArray, anchors: FloatArray) -> FloatArray:
    """Tính khoảng cách Euclid bình phương; input X/anchors, output (n, m)."""
    with np.errstate(over="ignore", invalid="ignore"):
        differences = X[:, np.newaxis, :] - anchors[np.newaxis, :, :]
        distances = np.einsum("nmd,nmd->nm", differences, differences, optimize=True)

    # Giữ khoảng cách hữu hạn khi float64 tràn số.
    max_float = np.finfo(np.float64).max
    distances = np.nan_to_num(distances, nan=max_float, posinf=max_float)
    return np.ascontiguousarray(np.maximum(distances, 0.0), dtype=np.float64)


class AnchorMapping:
    """Học anchor và biến đổi mẫu sang vector membership chuẩn hóa."""

    def __init__(
        self,
        n_anchors: int,
        sigma: float | None = None,
        random_state: int | None = None,
        batch_size: int = 1024,
        n_init: int = 10,
    ) -> None:
        """Lưu cấu hình; n_anchors, sigma và batch_size ảnh hưởng chất lượng/tốc độ."""
        self.n_anchors = _validate_positive_integer("n_anchors", n_anchors)
        self.sigma = _validate_sigma(sigma)
        self.batch_size = _validate_positive_integer("batch_size", batch_size)
        self.n_init = _validate_positive_integer("n_init", n_init)

        if random_state is None:
            self.random_state = 0
        elif isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise ValueError("random_state must be an integer or None.")
        elif random_state < 0:
            raise ValueError("random_state must be non-negative.")
        else:
            self.random_state = int(random_state)

        self.anchors_: FloatArray | None = None
        self.sigma_: float | None = None
        self.n_features_in_: int | None = None
        self.anchor_diagnostics_: dict[str, Any] = {}
        self.anchor_mapping_fitted_ = False

    def _check_is_fitted(self) -> None:
        """Optional validation: chặn transform trước khi anchor được fit."""
        if not self.anchor_mapping_fitted_:
            raise ValueError("Anchor mapping is not fitted.")

    def fit_anchor_mapping(self, X: Any) -> AnchorMapping:
        """Học anchor từ X; trả về self với anchors_ và sigma_ đã cố định.

        MiniBatch K-means tạo ``n_anchors`` tâm. Có thể tuning ``n_anchors``,
        ``sigma``, ``batch_size``, ``n_init`` và ``random_state``.
        """
        X_validated = _validate_input(X)
        n_samples, n_features = X_validated.shape
        if self.n_anchors > n_samples:
            raise ValueError(
                f"n_anchors must not exceed n_samples ({n_samples}); got {self.n_anchors}."
            )

        effective_batch_size = min(self.batch_size, n_samples)
        # Phase 1: gom dữ liệu thành m tâm đại diện bằng MiniBatch K-means.
        estimator = MiniBatchKMeans(
            n_clusters=self.n_anchors,
            random_state=self.random_state,
            n_init=self.n_init,
            batch_size=effective_batch_size,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(X_validated)

        anchors = np.ascontiguousarray(estimator.cluster_centers_, dtype=np.float64)
        distances = _squared_euclidean_distances(X_validated, anchors)

        # Phase 2: dùng sigma truyền vào, hoặc tự ước lượng từ anchor gần nhất.
        fallback_used = False
        if self.sigma is not None:
            sigma_resolved = self.sigma
            sigma_source = "explicit"
        else:
            nearest_distances = np.min(distances, axis=1)
            positive = nearest_distances[
                (nearest_distances > 0.0) & np.isfinite(nearest_distances)
            ]
            if positive.size:
                sigma_resolved = float(np.median(positive))
                sigma_source = "median_positive_nearest_distance"
            else:
                # Dùng epsilon khi mọi khoảng cách gần nhất đều bằng không.
                sigma_resolved = float(np.finfo(np.float64).eps)
                sigma_source = "machine_epsilon_fallback"
                fallback_used = True

        # Output được lưu để inference tái sử dụng, tuyệt đối không fit lại.
        unique_anchor_count = int(np.unique(anchors, axis=0).shape[0])
        self.anchors_ = anchors
        self.sigma_ = sigma_resolved
        self.n_features_in_ = n_features
        self.anchor_diagnostics_ = {
            "duplicate_anchor_count": self.n_anchors - unique_anchor_count,
            "effective_batch_size": effective_batch_size,
            "n_init": self.n_init,
            "random_state": self.random_state,
            "sigma_fallback_used": fallback_used,
            "sigma_source": sigma_source,
            "unique_anchor_count": unique_anchor_count,
        }
        self.anchor_mapping_fitted_ = True
        return self

    def compute_similarity(self, X: Any) -> FloatArray:
        """Nhận X và trả ma trận Gaussian similarity (n, n_anchors), chưa normalize."""
        self._check_is_fitted()
        assert self.anchors_ is not None
        assert self.sigma_ is not None
        assert self.n_features_in_ is not None

        X_validated = _validate_input(X, expected_features=self.n_features_in_)
        distances = _squared_euclidean_distances(X_validated, self.anchors_)
        # Thành viên Gaussian dùng đúng mẫu exp(-distance_squared / sigma).
        with np.errstate(over="ignore", under="ignore"):
            similarities = np.exp(-distances / self.sigma_)
        return np.ascontiguousarray(similarities, dtype=np.float64)

    def transform_anchor_space(self, X: Any) -> FloatArray:
        """Nhận X và trả membership Z (n, n_anchors), mỗi hàng có tổng bằng 1.

        ``sigma`` điều khiển membership mềm hay sắc; anchor và sigma đã fit được
        giữ nguyên để train/test dùng cùng một phép biến đổi.
        """
        self._check_is_fitted()
        assert self.anchors_ is not None
        assert self.sigma_ is not None
        assert self.n_features_in_ is not None

        # Chỉ tái sử dụng anchor và sigma đã học, không fit lại.
        X_validated = _validate_input(X, expected_features=self.n_features_in_)
        distances = _squared_euclidean_distances(X_validated, self.anchors_)
        row_minimum = np.min(distances, axis=1, keepdims=True)

        # Trừ logit lớn nhất để chuẩn hóa ổn định nhưng giữ nguyên tỷ lệ.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            logits = -(distances - row_minimum) / self.sigma_
            weights = np.exp(logits)
        row_sum = np.sum(weights, axis=1, keepdims=True)
        memberships = weights / row_sum

        if not np.all(np.isfinite(memberships)):
            raise ValueError("Normalized anchor memberships must be finite.")
        return np.ascontiguousarray(memberships, dtype=np.float64)

    def fit_transform_anchor_space(self, X: Any) -> FloatArray:
        """Fit anchor trên X rồi trả ngay membership Z của chính X."""
        self.fit_anchor_mapping(X)
        return self.transform_anchor_space(X)
