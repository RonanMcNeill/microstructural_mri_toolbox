"""Noise likelihoods for model fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GaussianNoise:
    """Independent Gaussian measurement noise."""

    sigma: ArrayLike

    def standard_deviation(self, size: int) -> NDArray[np.float64]:
        sigma = np.asarray(self.sigma, dtype=float)
        if sigma.ndim == 0 or sigma.size == 1:
            sigma = np.full(size, float(sigma.reshape(-1)[0]))
        else:
            sigma = sigma.reshape(-1)
        if sigma.shape != (size,):
            raise ValueError("sigma must be scalar or have one value per measurement.")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("sigma must be finite and greater than zero.")
        return sigma

    def log_likelihood(
        self,
        observed: ArrayLike,
        predicted: ArrayLike,
    ) -> float:
        data = np.asarray(observed, dtype=float).reshape(-1)
        model = np.asarray(predicted, dtype=float).reshape(-1)
        if data.shape != model.shape:
            raise ValueError("Observed and predicted signals must have equal shape.")
        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(model)):
            return -np.inf
        sigma = self.standard_deviation(data.size)
        residual = (data - model) / sigma
        return float(
            -0.5 * np.sum(residual**2)
            - np.sum(np.log(sigma))
            - 0.5 * data.size * np.log(2.0 * np.pi)
        )

