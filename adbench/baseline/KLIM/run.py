"""Thin ADBench adapter for the KLIM model skeleton."""

from __future__ import annotations

from typing import Any

from .model import KLIMModel


class KLIM:
    """Expose the ADBench detector interface without implementing KLIM training."""

    def __init__(
        self,
        seed: int,
        model_name: str | None = None,
        n_anchors: int = 32,
        sigma: float | None = None,
        alpha: float = 0.5,
        null_dim: int | None = None,
        gamma: float = 0.0,
        batch_size: int = 1024,
        tol: float = 1e-8,
    ) -> None:
        if model_name is not None and not isinstance(model_name, str):
            raise TypeError("model_name must be a string or None.")

        self.model = KLIMModel(
            n_anchors=n_anchors,
            sigma=sigma,
            alpha=alpha,
            null_dim=null_dim,
            gamma=gamma,
            random_state=seed,
            batch_size=batch_size,
            tol=tol,
        )
        self.seed = self.model.random_state
        self.model_name = model_name

    def fit(self, X_train: Any, y_train: Any) -> KLIM:
        """Delegate to the standalone model skeleton."""
        self.model.fit(X_train)
        return self

    def predict_score(self, X_test: Any) -> Any:
        """Delegate score generation to the standalone model skeleton."""
        return self.model.decision_function(X_test)
