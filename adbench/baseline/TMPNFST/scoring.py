"""Projected component centroids and nearest-centroid anomaly scoring."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .partition import validate_features


FloatArray = NDArray[np.float64]


def build_projected_centroids(
    X: Any,
    projection: Any,
    labels: Any,
) -> tuple[FloatArray, FloatArray]:
    """Nhận X/projection/labels, trả projected train và centroid mỗi nhóm."""
    features = validate_features(X)
    matrix = np.asarray(projection, dtype=np.float64)
    class_labels = np.asarray(labels)
    if matrix.ndim != 2 or matrix.shape[0] != features.shape[1] or matrix.shape[1] < 1:
        raise ValueError("projection must have shape (n_features, p), with p >= 1.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("projection must contain only finite values.")
    if class_labels.ndim != 1 or class_labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have one value per training sample.")
    # Chiếu train vào NFST subspace rồi lấy mean theo từng pseudo-class.
    projected = np.ascontiguousarray(features @ matrix)
    centroids = np.vstack(
        [projected[class_labels == label].mean(axis=0) for label in np.unique(class_labels)]
    )
    return projected, np.ascontiguousarray(centroids, dtype=np.float64)


def score_samples(
    X: Any,
    projection: Any,
    centroids: Any,
    *,
    batch_size: int | None = None,
) -> FloatArray:
    """Nhận X/projection/centroids và trả nearest-centroid score cho mỗi mẫu.

    ``batch_size`` có thể giảm RAM khi test lớn nhưng không thay đổi thuật toán.
    Score là khoảng cách Euclid bình phương; giá trị lớn bất thường hơn.
    """
    features = validate_features(X)
    matrix = np.asarray(projection, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != features.shape[1]:
        raise ValueError("projection has an incompatible shape.")
    if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] != matrix.shape[1]:
        raise ValueError("centroids have an incompatible shape.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(centers)):
        raise ValueError("projection and centroids must be finite.")
    if batch_size is None:
        effective_batch_size = features.shape[0]
    elif isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be a positive integer or None.")
    elif batch_size < 1:
        raise ValueError("batch_size must be a positive integer or None.")
    else:
        effective_batch_size = min(int(batch_size), features.shape[0])

    # Dùng công thức khai triển khoảng cách để không tạo tensor 3 chiều lớn.
    scores = np.empty(features.shape[0], dtype=np.float64)
    center_norms = np.einsum("ij,ij->i", centers, centers)
    # Xử lý từng batch và lấy khoảng cách nhỏ nhất tới mọi centroid.
    for start in range(0, features.shape[0], effective_batch_size):
        stop = min(start + effective_batch_size, features.shape[0])
        projected = features[start:stop] @ matrix
        projected_norms = np.einsum("ij,ij->i", projected, projected)[:, None]
        distances = projected_norms + center_norms[None, :] - 2.0 * projected @ centers.T
        scores[start:stop] = np.min(np.maximum(distances, 0.0), axis=1)
    if not np.all(np.isfinite(scores)):
        raise ValueError("anomaly scores must be finite.")
    return np.ascontiguousarray(scores)
