"""Horizon-aware residual calibration for comparable uncertainty intervals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ResidualCalibrator:
    quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)
    residual_quantiles: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    fallback: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def fit(
        self, actual: np.ndarray, predicted: np.ndarray, horizons: np.ndarray
    ) -> ResidualCalibrator:
        residuals = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
        fallback = np.quantile(residuals, self.quantiles)
        self.fallback = (float(fallback[0]), float(fallback[1]), float(fallback[2]))
        calibration = pd.DataFrame({"residual": residuals, "horizon": horizons})
        for horizon, group in calibration.groupby("horizon", sort=True):
            values = np.quantile(group["residual"], self.quantiles)
            self.residual_quantiles[int(str(horizon))] = (
                float(values[0]),
                float(values[1]),
                float(values[2]),
            )
        return self

    def predict(
        self, predicted: np.ndarray, horizons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        predicted = np.asarray(predicted, dtype=float)
        offsets = np.asarray(
            [self.residual_quantiles.get(int(horizon), self.fallback) for horizon in horizons]
        )
        quantiles = np.maximum(predicted[:, None] + offsets, 0)
        quantiles.sort(axis=1)
        return quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
