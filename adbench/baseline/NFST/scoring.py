"""Subclass base points and nearest-centroid anomaly scoring for NFST."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .scatter import validate_memberships


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BasePointResult:
    base_points: FloatArray
    valid_cluster_ids: IntArray
    empty_cluster_ids: IntArray
    subclass_counts: IntArray
    projected_training: FloatArray
    diagnostics: dict[str, Any]


def _validate_projection(projection: Any, n_anchors: int) -> FloatArray:
    try:
        raw = np.asarray(projection)
    except (TypeError, ValueError) as exc:
        raise ValueError("projection must be a two-dimensional numeric matrix.") from exc
    if raw.ndim != 2 or raw.shape[0] != n_anchors or raw.shape[1] < 1:
        raise ValueError(
            f"projection must have shape ({n_anchors}, p) with p at least one."
        )
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("projection must be a two-dimensional numeric matrix.")
    validated = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(validated)):
        raise ValueError("projection must contain only finite values.")
    return validated


def _validate_assignments(assignments: Any, n_samples: int, n_anchors: int) -> IntArray:
    raw = np.asarray(assignments)
    if raw.ndim != 1 or raw.shape[0] != n_samples:
        raise ValueError(f"assignments must have shape ({n_samples},).")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("assignments must contain integer cluster IDs.")
    validated = np.ascontiguousarray(raw, dtype=np.int64)
    if np.any(validated < 0) or np.any(validated >= n_anchors):
        raise ValueError("assignments contain an invalid cluster ID.")
    return validated


def _validate_points(points: Any, name: str) -> FloatArray:
    try:
        raw = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a two-dimensional numeric matrix.") from exc
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a two-dimensional numeric matrix.")
    validated = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(validated)):
        raise ValueError(f"{name} must contain only finite values.")
    return validated


def _validate_batch_size(batch_size: int | None, n_samples: int) -> int:
    if batch_size is None:
        return n_samples
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise ValueError("batch_size must be a positive integer or None.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer or None.")
    return min(int(batch_size), n_samples)


def build_subclass_base_points(
    Z_train: Any,
    projection: Any,
    assignments: Any,
) -> BasePointResult:
    """Project training memberships and average each non-empty subclass."""
    memberships = validate_memberships(Z_train)
    projection_matrix = _validate_projection(projection, memberships.shape[1])
    cluster_ids = _validate_assignments(
        assignments, memberships.shape[0], memberships.shape[1]
    )
    projected_training = np.ascontiguousarray(memberships @ projection_matrix)
    if not np.all(np.isfinite(projected_training)):
        raise ValueError("projected training samples must be finite.")

    subclass_counts = np.bincount(
        cluster_ids, minlength=memberships.shape[1]
    ).astype(np.int64, copy=False)
    valid_cluster_ids = np.flatnonzero(subclass_counts > 0).astype(np.int64)
    empty_cluster_ids = np.flatnonzero(subclass_counts == 0).astype(np.int64)
    if valid_cluster_ids.size == 0:
        raise ValueError("At least one non-empty subclass is required.")

    # Bỏ subclass rỗng, không tạo centroid zero giả.
    base_points = np.vstack(
        [projected_training[cluster_ids == cluster_id].mean(axis=0)
         for cluster_id in valid_cluster_ids]
    ).astype(np.float64, copy=False)
    base_points = np.ascontiguousarray(base_points)
    if not np.all(np.isfinite(base_points)):
        raise ValueError("base points must contain only finite values.")

    diagnostics = {
        "empty_cluster_ids": empty_cluster_ids.tolist(),
        "n_valid_base_points": int(valid_cluster_ids.size),
        "subclass_counts": subclass_counts.tolist(),
    }
    return BasePointResult(
        base_points=base_points,
        valid_cluster_ids=np.ascontiguousarray(valid_cluster_ids),
        empty_cluster_ids=np.ascontiguousarray(empty_cluster_ids),
        subclass_counts=np.ascontiguousarray(subclass_counts),
        projected_training=projected_training,
        diagnostics=diagnostics,
    )


def score_projected_samples(
    projected_samples: Any,
    base_points: Any,
    *,
    batch_size: int | None = None,
    roundoff_tolerance: float = 1e-12,
) -> FloatArray:
    """Return nearest squared Euclidean distance for each projected sample."""
    samples = _validate_points(projected_samples, "projected_samples")
    centers = _validate_points(base_points, "base_points")
    if samples.shape[1] != centers.shape[1]:
        raise ValueError("projected_samples and base_points must share a dimension.")
    if isinstance(roundoff_tolerance, bool) or not isinstance(
        roundoff_tolerance, Real
    ):
        raise ValueError("roundoff_tolerance must be a finite positive real number.")
    tolerance = float(roundoff_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("roundoff_tolerance must be a finite positive real number.")

    effective_batch_size = _validate_batch_size(batch_size, samples.shape[0])
    scores = np.empty(samples.shape[0], dtype=np.float64)
    center_norms = np.einsum("qd,qd->q", centers, centers)
    for start in range(0, samples.shape[0], effective_batch_size):
        stop = min(start + effective_batch_size, samples.shape[0])
        batch = samples[start:stop]
        sample_norms = np.einsum("bd,bd->b", batch, batch)[:, np.newaxis]
        distances = sample_norms + center_norms[np.newaxis, :] - 2.0 * batch @ centers.T
        scale = max(1.0, float(np.max(np.abs(distances))))
        minimum = float(np.min(distances))
        if minimum < -tolerance * scale:
            raise ValueError("squared distances contain a significant negative value.")
        # Chặn sai số âm rất nhỏ của công thức khoảng cách bình phương.
        distances = np.maximum(distances, 0.0)
        scores[start:stop] = np.min(distances, axis=1)

    if not np.all(np.isfinite(scores)):
        raise ValueError("anomaly scores must contain only finite values.")
    return np.ascontiguousarray(scores, dtype=np.float64)


def score_anchor_memberships(
    Z_test: Any,
    projection: Any,
    base_points: Any,
    *,
    batch_size: int | None = None,
) -> FloatArray:
    memberships = validate_memberships(Z_test)
    projection_matrix = _validate_projection(projection, memberships.shape[1])
    projected = np.ascontiguousarray(memberships @ projection_matrix)
    return score_projected_samples(projected, base_points, batch_size=batch_size)
