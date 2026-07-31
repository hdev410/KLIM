"""Thin ADBench adapter for topological manifold-partitioned NFST."""

from __future__ import annotations

from typing import Any, Literal

from numpy.typing import NDArray
import numpy as np

from .model import TMPNFSTModel


class TMPNFST:
    """Adapter mỏng đưa TMPNFST vào giao diện unsupervised của ADBench."""

    def __init__(
        self,
        seed: int,
        model_name: str | None = None,
        n_neighbors: int = 10,
        n_components: int = 3,
        sigma: float | None = None,
        n_init: int = 10,
        eigen_tolerance: float = 1e-8,
        eigen_maxiter: int | None = None,
        rank_tolerance: float = 1e-10,
        null_tolerance: float = 1e-8,
        max_components: int | None = None,
        selection_mode: Literal["null", "smallest"] = "smallest",
        score_batch_size: int | None = None,
    ) -> None:
        """Nhận seed/model_name và chuyển hyperparameter vào TMPNFSTModel."""
        if model_name is not None and not isinstance(model_name, str):
            raise TypeError("model_name must be a string or None.")
        self.model = TMPNFSTModel(
            n_neighbors=n_neighbors,
            n_components=n_components,
            sigma=sigma,
            random_state=seed,
            n_init=n_init,
            eigen_tolerance=eigen_tolerance,
            eigen_maxiter=eigen_maxiter,
            rank_tolerance=rank_tolerance,
            null_tolerance=null_tolerance,
            max_components=max_components,
            selection_mode=selection_mode,
            score_batch_size=score_batch_size,
        )
        self.seed = self.model.random_state
        self.model_name = model_name

    def fit(self, X_train: Any, y_train: Any) -> TMPNFST:
        """Fit trên X_train và trả self; y_train chỉ để tương thích, không dùng."""
        self.model.fit(X_train)
        return self

    def predict_score(self, X_test: Any) -> NDArray[np.float64]:
        """Nhận X_test và trả một anomaly score cho mỗi hàng."""
        return self.model.decision_function(X_test)
