"""Topological pseudo-class scatter matrices for TMP-NFST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .partition import validate_features


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TopologicalScatterResult:
    """Output scatter: Sw, Sb, St, labels, component means và diagnostics."""
    within_scatter: FloatArray
    between_scatter: FloatArray
    total_scatter: FloatArray
    labels: IntArray
    component_means: FloatArray
    component_ids: IntArray
    diagnostics: dict[str, Any]


def construct_topological_scatter(
    X: Any,
    labels: Any,
) -> TopologicalScatterResult:
    """Nhận X/pseudo-labels và trả within, between, total scatter.

    Mỗi pseudo-class có mean riêng. ``Sw`` đo độ phân tán trong nhóm, ``Sb`` đo
    khoảng cách giữa các mean; chất lượng phụ thuộc trực tiếp spectral labels.
    """
    features = validate_features(X)
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have one integer value per sample.")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("labels must contain integers.")
    class_labels = np.ascontiguousarray(raw_labels, dtype=np.int64)
    component_ids = np.unique(class_labels)
    if component_ids.size < 2:
        raise ValueError("At least two non-empty topological components are required.")

    # Phase 1: khởi tạo mean toàn cục và các scatter trong feature-space gốc.
    n_samples, n_features = features.shape
    global_mean = np.mean(features, axis=0)
    within = np.zeros((n_features, n_features), dtype=np.float64)
    between = np.zeros_like(within)
    means = []
    counts = []
    # Phase 2: cộng covariance trong nhóm và độ lệch mean giữa các nhóm.
    for component_id in component_ids:
        members = features[class_labels == component_id]
        component_mean = np.mean(members, axis=0)
        centered = members - component_mean
        within += centered.T @ centered
        mean_delta = component_mean - global_mean
        between += members.shape[0] * np.outer(mean_delta, mean_delta)
        means.append(component_mean)
        counts.append(int(members.shape[0]))

    # Phase 3: chuẩn hóa theo n và kiểm tra St = Sw + Sb với scatter trực tiếp.
    within /= n_samples
    between /= n_samples
    total = within + between
    direct_total = (features - global_mean).T @ (features - global_mean) / n_samples
    decomposition_error = float(np.max(np.abs(total - direct_total)))
    within = np.ascontiguousarray(0.5 * (within + within.T))
    between = np.ascontiguousarray(0.5 * (between + between.T))
    total = np.ascontiguousarray(0.5 * (total + total.T))
    # Output giữ component mean/count để diagnostics và kiểm thử decomposition.
    return TopologicalScatterResult(
        within_scatter=within,
        between_scatter=between,
        total_scatter=total,
        labels=class_labels,
        component_means=np.ascontiguousarray(means, dtype=np.float64),
        component_ids=np.ascontiguousarray(component_ids, dtype=np.int64),
        diagnostics={
            "component_counts": counts,
            "component_ids": component_ids.tolist(),
            "scatter_decomposition_max_abs": decomposition_error,
        },
    )
