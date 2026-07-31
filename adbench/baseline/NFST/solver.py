"""Two-stage range/null-space solver for NFST."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SolverResult:
    """Output của solver: projection, spectrum, range basis và diagnostics."""
    projection: FloatArray
    eigenvalues: FloatArray
    selected_eigenvalues: FloatArray
    range_basis: FloatArray
    rank: int
    nullity: int
    diagnostics: dict[str, Any]


def _validate_tolerance(name: str, value: float) -> float:
    """Optional validation: kiểm tra tolerance là số thực dương hữu hạn."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number.")
    validated = float(value)
    if not np.isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number.")
    return validated


def _validate_max_components(max_components: int | None) -> int | None:
    """Optional validation: kiểm tra số component là None hoặc số nguyên dương."""
    if max_components is None:
        return None
    if isinstance(max_components, bool) or not isinstance(max_components, Integral):
        raise ValueError("max_components must be a positive integer or None.")
    if max_components < 1:
        raise ValueError("max_components must be a positive integer or None.")
    return int(max_components)


def _validate_scatter(matrix: Any, name: str) -> FloatArray:
    """Optional validation: đổi scatter thành ma trận vuông float64 hữu hạn."""
    try:
        raw = np.asarray(matrix)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a square numeric matrix.") from exc
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"{name} must be a non-empty square numeric matrix.")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a square numeric matrix.")
    validated = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(validated)):
        raise ValueError(f"{name} must contain only finite values.")
    return validated


def _symmetrize_roundoff(
    matrix: FloatArray,
    name: str,
    tolerance: float,
) -> tuple[FloatArray, float]:
    """Optional validation: kiểm tra gần đối xứng và loại sai số làm tròn."""
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if asymmetry > tolerance * scale:
        raise ValueError(f"{name} must be symmetric within numerical tolerance.")
    return np.ascontiguousarray(0.5 * (matrix + matrix.T)), asymmetry


def solve_nfst_projection(
    total_scatter: Any,
    regularized_scatter: Any,
    *,
    rank_tolerance: float = 1e-10,
    null_tolerance: float = 1e-8,
    max_components: int | None = None,
    selection_mode: Literal["null", "smallest"] = "null",
) -> SolverResult:
    """Nhận total/regularized scatter và trả projection NFST cùng diagnostics.

    Thuật toán giải range-space rồi eigendecomposition bài toán thu gọn.
    ``rank_tolerance`` và ``null_tolerance`` ảnh hưởng pass/failed số học;
    ``selection_mode`` và ``max_components`` ảnh hưởng số hướng được chọn.
    """
    rank_tol = _validate_tolerance("rank_tolerance", rank_tolerance)
    null_tol = _validate_tolerance("null_tolerance", null_tolerance)
    component_cap = _validate_max_components(max_components)
    if selection_mode not in {"null", "smallest"}:
        raise ValueError("selection_mode must be 'null' or 'smallest'.")
    if selection_mode == "smallest" and component_cap is None:
        raise ValueError("max_components is required when selection_mode='smallest'.")
    total = _validate_scatter(total_scatter, "total_scatter")
    regularized = _validate_scatter(regularized_scatter, "regularized_scatter")
    if total.shape != regularized.shape:
        raise ValueError("total_scatter and regularized_scatter must have the same shape.")

    total, total_asymmetry = _symmetrize_roundoff(total, "total_scatter", rank_tol)
    regularized, regularized_asymmetry = _symmetrize_roundoff(
        regularized, "regularized_scatter", rank_tol
    )

    # Phase 1 - Range space: giữ các hướng có total variance dương đáng kể.
    total_eigenvalues, total_eigenvectors = np.linalg.eigh(total)
    total_scale = max(1.0, float(np.max(np.abs(total_eigenvalues))))
    rank_threshold = rank_tol * total_scale

    # Chọn rank số học của S_T bằng ngưỡng phụ thuộc thang đo.
    range_mask = total_eigenvalues > rank_threshold
    rank = int(np.count_nonzero(range_mask))
    if rank == 0:
        raise ValueError(
            "total_scatter has numerical rank zero; NFST range space is undefined."
        )
    range_basis = np.ascontiguousarray(total_eigenvectors[:, range_mask])

    # Phase 2 - Reduced problem: đưa within/regularized scatter vào range(S_T).
    reduced = range_basis.T @ regularized @ range_basis
    reduced = np.ascontiguousarray(0.5 * (reduced + reduced.T))
    eigenvalues, reduced_eigenvectors = np.linalg.eigh(reduced)
    order = np.argsort(eigenvalues)
    eigenvalues = np.ascontiguousarray(eigenvalues[order], dtype=np.float64)
    reduced_eigenvectors = np.ascontiguousarray(reduced_eigenvectors[:, order])

    reduced_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    null_threshold = null_tol * reduced_scale

    # Chỉ chọn trị riêng gần zero; không bù bằng hướng non-null.
    # Phase 3 - Selection: strict-null chỉ nhận eigenvalue gần 0; smallest là
    # biến thể thực dụng khi dữ liệu thật không có numerical null-space.
    null_indices = np.flatnonzero(np.abs(eigenvalues) <= null_threshold)
    numerical_nullity = int(null_indices.size)
    if selection_mode == "null":
        if numerical_nullity == 0:
            raise ValueError(
                "Reduced NFST problem has numerical nullity zero; "
                "no valid projection exists."
            )
        selected_indices = null_indices
        if component_cap is not None:
            selected_indices = selected_indices[:component_cap]
    else:
        assert component_cap is not None
        selected_indices = np.arange(min(component_cap, rank), dtype=np.int64)

    # Phase 4 - Recover: đổi eigenvector thu gọn về feature/anchor-space gốc.
    selected_vectors = reduced_eigenvectors[:, selected_indices]
    selected_eigenvalues = np.ascontiguousarray(
        eigenvalues[selected_indices], dtype=np.float64
    )
    projection = np.ascontiguousarray(range_basis @ selected_vectors)
    if not np.all(np.isfinite(projection)):
        raise ValueError("NFST projection must contain only finite values.")

    # Diagnostics không đổi kết quả; chúng giúp phân biệt exact-null và low-eigen.
    projected_null_residuals = np.linalg.norm(reduced @ selected_vectors, axis=0)
    range_residual = float(
        np.linalg.norm(projection - range_basis @ (range_basis.T @ projection))
    )
    range_orthogonality_error = float(
        np.linalg.norm(range_basis.T @ range_basis - np.eye(rank))
    )
    projection_orthogonality_error = float(
        np.linalg.norm(
            projection.T @ projection - np.eye(projection.shape[1])
        )
    )
    diagnostics = {
        "full_total_spectrum": total_eigenvalues.tolist(),
        "null_threshold": null_threshold,
        "numerical_nullity": numerical_nullity,
        "projected_null_residual_max": float(np.max(projected_null_residuals)),
        "projection_orthogonality_error": projection_orthogonality_error,
        "range_orthogonality_error": range_orthogonality_error,
        "range_residual": range_residual,
        "rank_threshold": rank_threshold,
        "reduced_spectrum": eigenvalues.tolist(),
        "regularized_scatter_asymmetry": regularized_asymmetry,
        "selection_mode": selection_mode,
        "selected_dimension": int(projection.shape[1]),
        "total_scatter_asymmetry": total_asymmetry,
    }
    return SolverResult(
        projection=projection,
        eigenvalues=eigenvalues,
        selected_eigenvalues=selected_eigenvalues,
        range_basis=range_basis,
        rank=rank,
        nullity=numerical_nullity,
        diagnostics=diagnostics,
    )
