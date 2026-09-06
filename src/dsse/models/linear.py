"""Ridge regression baseline for DSSE."""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dsse.reproducibility import setup_logger

logger = setup_logger("model.linear")


class RidgeBaseline:
    """Sklearn Ridge regression with standardised inputs."""

    def __init__(self, alpha: float = 1.0, seed: int = 42) -> None:
        self.alpha = alpha
        self.seed = seed
        self._pipeline: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        """Fits the ridge regression pipeline."""
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self._pipeline.fit(X, y)
        logger.info(f"RidgeBaseline fitted: X={X.shape}, y={y.shape}, alpha={self.alpha}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts voltage magnitudes in per-unit."""
        if self._pipeline is None:
            raise RuntimeError("Model must be fitted before predicting.")
        return self._pipeline.predict(X)
