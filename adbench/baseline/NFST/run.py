"""Thin ADBench adapter for the NFST detector."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .model import NFSTModel
from .scatter import DEFAULT_MAX_GRAPH_BYTES


class NFST:
    """Expose the unsupervised ADBench detector interface."""

    def __init__(
        self,
        seed: int,
        model_name: str | None = None,
        n_anchors: int = 32,
        sigma: float | None = None,
        alpha: float = 0.5,
        batch_size: int = 1024,
        tol: float = 1e-8,
        rank_tolerance: float = 1e-10,
        null_tolerance: float = 1e-8,
        max_components: int | None = None,
        selection_mode: Literal["null", "smallest"] = "null",
        max_graph_bytes: int | None = DEFAULT_MAX_GRAPH_BYTES,
        score_batch_size: int | None = None,
    ) -> None:
        if model_name is not None and not isinstance(model_name, str):
            raise TypeError("model_name must be a string or None.")
        self.model = NFSTModel(
            n_anchors=n_anchors,
            sigma=sigma,
            alpha=alpha,
            random_state=seed,
            batch_size=batch_size,
            tol=tol,
            rank_tolerance=rank_tolerance,
            null_tolerance=null_tolerance,
            max_components=max_components,
            selection_mode=selection_mode,
            max_graph_bytes=max_graph_bytes,
            score_batch_size=score_batch_size,
        )
        self.seed = self.model.random_state
        self.model_name = model_name

    def fit(self, X_train: Any, y_train: Any) -> NFST:
        """Fit on the complete ADBench training matrix; labels are compatibility-only."""
        self.model.fit(X_train)
        return self

    def predict_score(self, X_test: Any) -> NDArray[np.float64]:
        return self.model.decision_function(X_test)
