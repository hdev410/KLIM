"""Subclass assignment and source-defined scatter construction for NFST."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DEFAULT_MAX_GRAPH_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class GraphResult:
    """Output graph: W, degree, Laplacian và diagnostics."""
    similarity_graph: FloatArray
    degree: FloatArray
    laplacian: FloatArray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ScatterResult:
    """Output phase scatter: các ma trận, subclass assignment và diagnostics."""
    global_scatter: FloatArray
    local_scatter: FloatArray
    total_scatter: FloatArray
    regularized_scatter: FloatArray
    assignments: IntArray
    degree: FloatArray
    diagnostics: dict[str, Any]


def _validate_tolerance(tolerance: float) -> float:
    """Optional validation: kiểm tra tolerance là số thực dương hữu hạn."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise ValueError("tolerance must be a finite positive real number.")
    validated = float(tolerance)
    if not np.isfinite(validated) or validated <= 0.0:
        raise ValueError("tolerance must be a finite positive real number.")
    return validated


def _validate_alpha(alpha: float) -> float:
    """Optional validation: kiểm tra alpha nằm trong [0, 1]."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite real number in [0, 1].")
    validated = float(alpha)
    if not np.isfinite(validated) or not 0.0 <= validated <= 1.0:
        raise ValueError("alpha must be a finite real number in [0, 1].")
    return validated


def _validate_max_graph_bytes(max_graph_bytes: int | None) -> int | None:
    """Optional validation: kiểm tra memory guard là None hoặc số byte dương."""
    if max_graph_bytes is None:
        return None
    if isinstance(max_graph_bytes, bool) or not isinstance(max_graph_bytes, Integral):
        raise ValueError("max_graph_bytes must be a positive integer or None.")
    if max_graph_bytes < 1:
        raise ValueError("max_graph_bytes must be a positive integer or None.")
    return int(max_graph_bytes)


def validate_memberships(
    Z: Any,
    *,
    row_sum_tolerance: float = 1e-8,
) -> FloatArray:
    """Optional validation: kiểm tra Z 2D, không âm và mỗi hàng có tổng gần 1."""
    tolerance = _validate_tolerance(row_sum_tolerance)
    try:
        raw = np.asarray(Z)
    except (TypeError, ValueError) as exc:
        raise ValueError("Z must be a two-dimensional numeric array.") from exc

    if raw.ndim != 2:
        raise ValueError("Z must be a two-dimensional numeric array.")
    if raw.shape[0] < 1:
        raise ValueError("Z must contain at least one sample row.")
    if raw.shape[1] < 1:
        raise ValueError("Z must contain at least one anchor column.")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("Z must be a two-dimensional numeric array.")

    validated = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(validated)):
        raise ValueError("Z must contain only finite values.")
    if np.any(validated < 0.0):
        raise ValueError("Z memberships must be non-negative.")

    row_sums = np.sum(validated, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=tolerance, atol=tolerance):
        raise ValueError("Every row of Z must sum approximately to one.")
    return validated


def assign_subclasses(Z: Any) -> IntArray:
    """Nhận membership Z, trả subclass ID bằng vị trí membership lớn nhất."""
    memberships = validate_memberships(Z)
    return np.asarray(np.argmax(memberships, axis=1), dtype=np.int64)


def _symmetrize(matrix: FloatArray) -> FloatArray:
    """Optional numerical cleanup: đối xứng hóa sai số float rất nhỏ."""
    # Chỉ đối xứng hóa để loại sai số làm tròn số học.
    return np.ascontiguousarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def _symmetry_error(matrix: FloatArray) -> float:
    """Optional diagnostic: trả sai lệch đối xứng lớn nhất của ma trận."""
    return float(np.max(np.abs(matrix - matrix.T)))


def _validate_symmetric(
    matrix: FloatArray,
    name: str,
    tolerance: float,
) -> float:
    """Optional validation: kiểm tra ma trận đối xứng trong tolerance."""
    error = _symmetry_error(matrix)
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if error > tolerance * scale:
        raise ValueError(f"{name} must be symmetric within numerical tolerance.")
    return error


def _validate_psd(
    matrix: FloatArray,
    name: str,
    tolerance: float,
) -> tuple[float, float]:
    """Optional validation: kiểm tra ma trận positive semidefinite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum = float(eigenvalues[0])
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    lower_bound = -tolerance * scale
    if minimum < lower_bound:
        raise ValueError(f"{name} must be positive semidefinite within tolerance.")
    return minimum, lower_bound


def construct_graph_laplacian(
    Z: Any,
    *,
    max_graph_bytes: int | None = DEFAULT_MAX_GRAPH_BYTES,
    tolerance: float = 1e-10,
) -> GraphResult:
    """Nhận Z và trả dense graph W, degree, Laplacian L cùng diagnostics.

    Thuật toán dùng ``W=ZZ.T`` và ``L=D-W``. ``max_graph_bytes`` chỉ giới hạn
    RAM để tránh crash; ``tolerance`` chỉ điều khiển kiểm tra số học.
    """
    memberships = validate_memberships(Z)
    validated_tolerance = _validate_tolerance(tolerance)
    byte_limit = _validate_max_graph_bytes(max_graph_bytes)
    n_samples = memberships.shape[0]
    estimated_bytes = 2 * n_samples * n_samples * np.dtype(np.float64).itemsize

    # Chặn trước hai ma trận dense W và L có bộ nhớ O(n^2).
    if byte_limit is not None and estimated_bytes > byte_limit:
        raise ValueError(
            "Dense W and L require O(n^2) memory: "
            f"estimated {estimated_bytes} bytes exceeds max_graph_bytes={byte_limit}."
        )

    # Phase 1: độ giống giữa hai mẫu là tích vô hướng membership của chúng.
    similarity_graph = np.ascontiguousarray(memberships @ memberships.T)
    if not np.all(np.isfinite(similarity_graph)):
        raise ValueError("W must contain only finite values.")
    graph_symmetry_error = _validate_symmetric(
        similarity_graph, "W", validated_tolerance
    )

    # Bậc đồ thị là tổng từng hàng của W, gồm cả self-similarity.
    degree = np.ascontiguousarray(np.sum(similarity_graph, axis=1), dtype=np.float64)

    # Tạo L = D - W mà không cần lưu ma trận D riêng.
    # Phase 2: tạo L=D-W trực tiếp để không cần cấp phát thêm ma trận D dense.
    laplacian = -similarity_graph.copy()
    diagonal = np.diag_indices(n_samples)
    laplacian[diagonal] += degree

    if not np.all(np.isfinite(laplacian)):
        raise ValueError("L must contain only finite values.")
    laplacian_symmetry_error = _validate_symmetric(laplacian, "L", validated_tolerance)
    row_sum_error = float(np.max(np.abs(np.sum(laplacian, axis=1))))
    laplacian_scale = max(1.0, float(np.max(np.abs(laplacian))))
    if row_sum_error > validated_tolerance * laplacian_scale * n_samples:
        raise ValueError("Rows of L must sum approximately to zero.")
    minimum_eigenvalue, psd_lower_bound = _validate_psd(
        laplacian, "L", validated_tolerance
    )

    # Output diagnostics ghi memory, symmetry, row-sum và PSD checks.
    diagnostics = {
        "dense_graph_bytes_estimate": estimated_bytes,
        "laplacian_min_eigenvalue": minimum_eigenvalue,
        "laplacian_psd_lower_bound": psd_lower_bound,
        "laplacian_row_sum_max_abs": row_sum_error,
        "laplacian_symmetry_max_abs": laplacian_symmetry_error,
        "max_graph_bytes": byte_limit,
        "self_similarity_preserved": True,
        "similarity_graph_symmetry_max_abs": graph_symmetry_error,
    }
    return GraphResult(similarity_graph, degree, laplacian, diagnostics)


def construct_scatter_matrices(
    Z: Any,
    alpha: float,
    *,
    max_graph_bytes: int | None = DEFAULT_MAX_GRAPH_BYTES,
    tolerance: float = 1e-10,
) -> ScatterResult:
    """Nhận membership Z/alpha và trả scatter matrices cùng subclass ID.

    ``alpha`` cân bằng global scatter và graph-local scatter; nên tuning theo
    validation. ``max_graph_bytes`` và ``tolerance`` chủ yếu ảnh hưởng pass/fail.
    """
    memberships = validate_memberships(Z)
    validated_alpha = _validate_alpha(alpha)
    validated_tolerance = _validate_tolerance(tolerance)
    graph = construct_graph_laplacian(
        memberships,
        max_graph_bytes=max_graph_bytes,
        tolerance=validated_tolerance,
    )

    # Phase 1: mỗi mẫu thuộc subclass có anchor membership lớn nhất.
    assignments = np.asarray(np.argmax(memberships, axis=1), dtype=np.int64)
    empty_subclass_ids = np.setdiff1d(
        np.arange(memberships.shape[1], dtype=np.int64),
        np.unique(assignments),
        assume_unique=True,
    )

    # Centering tương đương Z.T @ H @ Z nhưng không tạo H dense.
    # Phase 2: global/total scatter đo biến thiên quanh mean trong anchor-space.
    centered = memberships - np.mean(memberships, axis=0, keepdims=True)
    global_scatter = _symmetrize(centered.T @ centered)

    # Eq. (16) dùng trực tiếp Z.T @ L @ Z, không tính L^(1/2).
    # Phase 3: local scatter giữ thông tin láng giềng từ graph Laplacian.
    local_scatter = _symmetrize(memberships.T @ graph.laplacian @ memberships)
    total_scatter = global_scatter.copy()
    # Phase 4: alpha=1 thiên global; alpha=0 thiên hoàn toàn về local graph.
    regularized_scatter = _symmetrize(
        validated_alpha * global_scatter + (1.0 - validated_alpha) * local_scatter
    )

    matrices = {
        "global_scatter": global_scatter,
        "local_scatter": local_scatter,
        "total_scatter": total_scatter,
        "regularized_scatter": regularized_scatter,
    }
    matrix_diagnostics: dict[str, Any] = {}
    expected_shape = (memberships.shape[1], memberships.shape[1])
    for name, matrix in matrices.items():
        if matrix.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values.")
        matrix_diagnostics[f"{name}_symmetry_max_abs"] = _validate_symmetric(
            matrix, name, validated_tolerance
        )
        minimum, lower_bound = _validate_psd(matrix, name, validated_tolerance)
        matrix_diagnostics[f"{name}_min_eigenvalue"] = minimum
        matrix_diagnostics[f"{name}_psd_lower_bound"] = lower_bound

    diagnostics = {
        "alpha": validated_alpha,
        "empty_subclass_ids": empty_subclass_ids.tolist(),
        **graph.diagnostics,
        **matrix_diagnostics,
    }
    return ScatterResult(
        global_scatter=global_scatter,
        local_scatter=local_scatter,
        total_scatter=total_scatter,
        regularized_scatter=regularized_scatter,
        assignments=assignments,
        degree=graph.degree.copy(),
        diagnostics=diagnostics,
    )
