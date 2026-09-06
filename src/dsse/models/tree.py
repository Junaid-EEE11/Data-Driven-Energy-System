from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from dsse.reproducibility import setup_logger

logger = setup_logger("model.tree")


class TreeBaseline:
    """Random Forest multi-output tree ensemble regressor for DSSE."""

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 10,
        min_samples_split: int = 4,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeBaseline":
        """Fits the multi-output tree ensemble."""
        logger.info(f"Fitting TreeBaseline: X={X.shape}, y={y.shape}")
        self._model.fit(X, y)
        logger.info("TreeBaseline fitted successfully.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts voltage magnitudes."""
        return self._model.predict(X)
