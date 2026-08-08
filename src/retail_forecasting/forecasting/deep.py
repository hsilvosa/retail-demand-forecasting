"""N-HiTS adapter for global probabilistic retail forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass
class NHITSForecaster:
    horizon: int = 28
    input_size: int = 168
    max_steps: int = 500
    seed: int = 42
    use_gpu: bool = True
    model: Any = None

    name: str = "nhits"
    future_columns: tuple[str, ...] = (
        "wday",
        "month",
        "snap",
        "sell_price",
        "price_missing",
    )

    @staticmethod
    def _panel(frame: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        columns = {
            "series_id": "unique_id",
            "date": "ds",
            "units": "y",
        }
        result = frame.rename(columns=columns).copy()
        result["ds"] = pd.to_datetime(result["ds"])
        result["snap"] = np.select(
            [result["state_id"].eq("CA"), result["state_id"].eq("TX")],
            [result.get("snap_CA", 0), result.get("snap_TX", 0)],
            default=result.get("snap_WI", 0),
        )
        result["price_missing"] = result["sell_price"].isna().astype("float32")
        result["sell_price"] = result["sell_price"].fillna(0)
        keep = ["unique_id", "ds", *NHITSForecaster.future_columns]
        if include_target:
            keep.append("y")
        return result[keep].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def fit(self, history: pd.DataFrame, validation_size: int = 28) -> NHITSForecaster:
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS

        torch.manual_seed(self.seed)
        model = NHITS(
            h=self.horizon,
            input_size=self.input_size,
            max_steps=self.max_steps,
            futr_exog_list=list(self.future_columns),
            loss=MQLoss(quantiles=[0.05, 0.5, 0.95]),
            scaler_type="robust",
            random_seed=self.seed,
            accelerator="gpu" if self.use_gpu else "cpu",
            devices=1,
            enable_progress_bar=False,
        )
        self.model = NeuralForecast(models=[model], freq="D")
        self.model.fit(df=self._panel(history), val_size=validation_size)
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("forecaster has not been fitted")
        raw = self.model.predict(futr_df=self._panel(future, include_target=False))
        value_columns = [column for column in raw.columns if column not in {"unique_id", "ds"}]
        median = next((column for column in value_columns if "median" in column.lower()), None)
        low = next((column for column in value_columns if "lo-90" in column.lower()), None)
        high = next((column for column in value_columns if "hi-90" in column.lower()), None)
        point = median or next(column for column in value_columns if "nhits" in column.lower())
        if low is None or high is None:
            quantile_columns = sorted(value_columns)
            low, point, high = (
                quantile_columns[0],
                quantile_columns[len(quantile_columns) // 2],
                quantile_columns[-1],
            )
        result = raw.rename(
            columns={
                "unique_id": "series_id",
                "ds": "target_date",
                point: "q50",
                low: "q05",
                high: "q95",
            }
        )[["series_id", "target_date", "q05", "q50", "q95"]]
        result["yhat"] = result["q50"]
        for column in ("q05", "q50", "q95", "yhat"):
            result[column] = result[column].clip(lower=0)
        return cast(pd.DataFrame, result)
