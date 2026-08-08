"""Transparent baselines that establish the minimum useful forecast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SeasonalNaive:
    """Repeat the seven observations immediately preceding the origin."""

    name: str = "seasonal_naive"
    calibrator: Any = None

    def fit(self, frame: pd.DataFrame, target: str = "target") -> SeasonalNaive:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        horizons = frame["horizon"].to_numpy(dtype=int)
        source_lags = 7 - ((horizons - 1) % 7)
        values = np.empty(len(frame), dtype=float)
        for lag in range(1, 8):
            mask = source_lags == lag
            values[mask] = frame.loc[mask, f"lag_{lag}"].fillna(0).to_numpy(dtype=float)
        return np.clip(values, 0, None)


@dataclass
class MovingAverage:
    """Forecast every horizon with the trailing mean available at the origin."""

    window: int = 28
    name: str = "moving_average"
    calibrator: Any = None

    def fit(self, frame: pd.DataFrame, target: str = "target") -> MovingAverage:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(
            frame[f"rolling_mean_{self.window}"].fillna(0).to_numpy(dtype=float), 0, None
        )
