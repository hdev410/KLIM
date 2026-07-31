"""Sparse topological graph construction and spectral pseudo-label discovery."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PartitionResult:
    """Output phân hoạch: graph, embedding, labels, lifetime và diagnostics."""
    adjacency: sparse.csr_matrix
    laplacian: sparse.csr_matrix
    embedding: FloatArray
    labels: IntArray
    lifetimes: IntArray
    eigenvalues: FloatArray
    sigma: float
    diagnostics: dict[str, Any]


def validate_features(
    X: Any,
    *,
    expected_features: int | None = None,
    minimum_samples: int = 1,
) -> FloatArray:
    """Optional validation: đổi X thành float64 2D hữu hạn, đúng shape."""
    try:
        raw = np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a two-dimensional numeric matrix.") from exc
    if raw.ndim != 2 or raw.shape[0] < minimum_samples or raw.shape[1] < 1:
        raise ValueError(
            f"X must have at least {minimum_samples} rows and one feature."
        )
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("X must be a two-dimensional numeric matrix.")
    validated = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(validated)):
        raise ValueError("X must contain only finite values.")
    if expected_features is not None and validated.shape[1] != expected_features:
        raise ValueError(
            f"X must have {expected_features} features; got {validated.shape[1]}."
        )
    return validated


def _positive_integer(name: str, value: int) -> int:
    """Optional validation: kiểm tra một tham số là số nguyên dương."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _positive_real(name: str, value: float) -> float:
    """Optional validation: kiểm tra một tham số là số thực dương hữu hạn."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def calculate_graph_lifetimes(adjacency: sparse.csr_matrix) -> IntArray:
    """Nhận sparse adjacency và trả vòng peeling/lifetime của từng node.

    Thuật toán lặp xóa đồng thời các node có degree nhỏ nhất. Lifetime hiện chỉ
    dùng làm diagnostics vì đặc tả chưa định nghĩa threshold để đổi labels.
    """
    graph = adjacency.tocsr(copy=True)
    graph.data = np.ones_like(graph.data)
    graph.eliminate_zeros()
    n_samples = graph.shape[0]
    alive = np.ones(n_samples, dtype=bool)
    degrees = np.diff(graph.indptr).astype(np.int64, copy=False)
    lifetimes = np.zeros(n_samples, dtype=np.int64)
    iteration = 1

    # Mỗi vòng loại toàn bộ node có degree hiện tại thấp nhất.
    while np.any(alive):
        minimum = int(np.min(degrees[alive]))
        removed = np.flatnonzero(alive & (degrees == minimum))
        lifetimes[removed] = iteration
        alive[removed] = False
        # Chỉ giảm degree của hàng xóm còn sống; mỗi cạnh được xử lý hữu hạn lần.
        for node in removed:
            neighbors = graph.indices[graph.indptr[node] : graph.indptr[node + 1]]
            surviving_neighbors = neighbors[alive[neighbors]]
            degrees[surviving_neighbors] -= 1
        iteration += 1
    return np.ascontiguousarray(lifetimes)


def _repair_isolates(
    mutual: sparse.csr_matrix,
    directed: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, int]:
    """Sửa node cô lập bằng cạnh k-NN mạnh nhất; trả graph và số node đã sửa."""
    degree = np.asarray(mutual.sum(axis=1)).ravel()
    isolates = np.flatnonzero(degree <= 0.0)
    if isolates.size == 0:
        return mutual, 0

    # Mutual k-NN có thể bỏ hết cạnh của một node; thêm lại đúng một cạnh mạnh.
    repaired = mutual.tolil(copy=True)
    for node in isolates:
        start, stop = directed.indptr[node], directed.indptr[node + 1]
        candidates = directed.indices[start:stop]
        weights = directed.data[start:stop]
        if candidates.size == 0:
            raise ValueError("The k-NN graph contains an unrepairable isolated sample.")
        neighbor = int(candidates[int(np.argmax(weights))])
        weight = float(np.max(weights))
        repaired[node, neighbor] = max(float(repaired[node, neighbor]), weight)
        repaired[neighbor, node] = max(float(repaired[neighbor, node]), weight)
    return repaired.tocsr(), int(isolates.size)


def spectral_partition(
    X: Any,
    *,
    n_neighbors: int = 10,
    n_components: int = 3,
    sigma: float | None = None,
    random_state: int | None = None,
    n_init: int = 10,
    eigen_tolerance: float = 1e-8,
    eigen_maxiter: int | None = None,
) -> PartitionResult:
    """Nhận X train và trả pseudo-class bằng spectral clustering.

    Luồng: mutual k-NN Gaussian graph -> normalized Laplacian -> ``c``
    eigenvectors nhỏ nhất -> K-means. Tuning chính: ``n_neighbors``,
    ``n_components`` và ``sigma``; ``eigen_tolerance/eigen_maxiter`` ảnh hưởng
    ARPACK pass/failed; ``n_init/random_state`` ảnh hưởng độ ổn định K-means.
    """
    features = validate_features(X, minimum_samples=3)
    requested_neighbors = _positive_integer("n_neighbors", n_neighbors)
    component_count = _positive_integer("n_components", n_components)
    kmeans_initializations = _positive_integer("n_init", n_init)
    tolerance = _positive_real("eigen_tolerance", eigen_tolerance)
    maximum_iterations = (
        None
        if eigen_maxiter is None
        else _positive_integer("eigen_maxiter", eigen_maxiter)
    )
    if component_count >= features.shape[0]:
        raise ValueError("n_components must be smaller than the number of samples.")
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, Integral)
    ):
        raise TypeError("random_state must be an integer or None.")
    if random_state is not None and random_state < 0:
        raise ValueError("random_state must be non-negative.")
    seed = 0 if random_state is None else int(random_state)
    effective_neighbors = min(requested_neighbors, features.shape[0] - 1)

    # Phase 1 - k-NN: tìm láng giềng trong feature-space, bỏ chính mẫu đó.
    neighbors = NearestNeighbors(n_neighbors=effective_neighbors + 1)
    neighbors.fit(features)
    distances, indices = neighbors.kneighbors(features, return_distance=True)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    squared_distances = np.square(distances, dtype=np.float64)
    positive = squared_distances[
        (squared_distances > 0.0) & np.isfinite(squared_distances)
    ]
    # Phase 2 - Bandwidth: dùng sigma cố định hoặc median khoảng cách k-NN^2.
    if sigma is None:
        resolved_sigma = (
            float(np.median(positive))
            if positive.size
            else float(np.finfo(np.float64).eps)
        )
        sigma_source = "median_positive_knn_squared_distance"
    else:
        resolved_sigma = _positive_real("sigma", sigma)
        sigma_source = "explicit"

    rows = np.repeat(np.arange(features.shape[0]), effective_neighbors)
    columns = indices.reshape(-1)
    weights = np.exp(-squared_distances.reshape(-1) / resolved_sigma)
    # Phase 3 - Graph: trọng số Gaussian; chỉ giữ cạnh xuất hiện ở cả hai hướng.
    directed = sparse.csr_matrix(
        (weights, (rows, columns)), shape=(features.shape[0], features.shape[0])
    )
    directed.setdiag(0.0)
    directed.eliminate_zeros()
    mutual = directed.minimum(directed.T).tocsr()
    adjacency, repaired_isolates = _repair_isolates(mutual, directed)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()

    # Phase 4 - Laplacian: dạng symmetric normalized ổn định cho eigensolver.
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt_degree = np.zeros_like(degrees)
    positive_degree = degrees > 0.0
    inverse_sqrt_degree[positive_degree] = 1.0 / np.sqrt(degrees[positive_degree])
    normalizer = sparse.diags(inverse_sqrt_degree)
    normalized_adjacency = normalizer @ adjacency @ normalizer
    laplacian = (
        sparse.eye(features.shape[0], dtype=np.float64, format="csr")
        - normalized_adjacency
    ).tocsr()

    # Phase 5 - Spectral embedding: dense eigh cho bài toán rất nhỏ, ARPACK
    # ``which=SA`` cho sparse graph lớn. Có thể nới tolerance/tăng maxiter nếu fail.
    if component_count >= features.shape[0] - 1:
        full_values, full_vectors = np.linalg.eigh(laplacian.toarray())
        eigenvalues = full_values[:component_count]
        embedding = full_vectors[:, :component_count]
    else:
        initial = np.random.RandomState(seed).normal(size=features.shape[0])
        eigenvalues, embedding = eigsh(
            laplacian,
            k=component_count,
            # L is symmetric PSD, so smallest algebraic is also smallest magnitude
            # and is more reliable for ARPACK on weakly connected graphs.
            which="SA",
            tol=tolerance,
            v0=initial,
            maxiter=maximum_iterations,
        )
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        embedding = embedding[:, order]
    embedding = np.ascontiguousarray(embedding, dtype=np.float64)
    # Phase 6 - Pseudo-labels: K-means chia mỗi hàng embedding thành c nhóm.
    labels = KMeans(
        n_clusters=component_count,
        random_state=seed,
        n_init=kmeans_initializations,
    ).fit_predict(embedding)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    # Lifetime được lưu để phân tích topology nhưng không tham gia gán label.
    lifetimes = calculate_graph_lifetimes(adjacency)
    component_sizes = np.bincount(labels, minlength=component_count)

    # Output diagnostics giúp phát hiện graph thưa, isolate hoặc cụm mất cân bằng.
    diagnostics = {
        "adjacency_nnz": int(adjacency.nnz),
        "component_sizes": component_sizes.tolist(),
        "effective_neighbors": effective_neighbors,
        "eigen_maxiter": maximum_iterations,
        "eigen_tolerance": tolerance,
        "eigen_which": "SA",
        "eigenvalues": np.asarray(eigenvalues, dtype=np.float64).tolist(),
        "isolates_repaired": repaired_isolates,
        "lifetime_max": int(np.max(lifetimes)),
        "lifetime_min": int(np.min(lifetimes)),
        "laplacian_form": "symmetric_normalized_equivalent_of_random_walk",
        "n_components": component_count,
        "sigma": resolved_sigma,
        "sigma_source": sigma_source,
    }
    return PartitionResult(
        adjacency=adjacency,
        laplacian=laplacian,
        embedding=embedding,
        labels=labels,
        lifetimes=lifetimes,
        eigenvalues=np.ascontiguousarray(eigenvalues, dtype=np.float64),
        sigma=resolved_sigma,
        diagnostics=diagnostics,
    )
