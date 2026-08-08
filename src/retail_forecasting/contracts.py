"""Stable records exchanged between forecasting and inventory stages."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    model_name: str
    model_version: str | None = None
    origin_date: date
    forecast_date: date
    series_id: str
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str
    horizon: int = Field(gt=0)
    yhat: float = Field(ge=0)
    q05: float = Field(ge=0)
    q50: float = Field(ge=0)
    q95: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_quantiles(self) -> "ForecastRecord":
        if not self.q05 <= self.q50 <= self.q95:
            raise ValueError("forecast quantiles must be monotonic")
        return self


class InventoryRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    series_id: str
    as_of_date: date
    lead_time_days: int = Field(gt=0)
    review_period_days: int = Field(gt=0)
    service_level: float = Field(gt=0, lt=1)
    on_hand: float = Field(ge=0)
    reorder_point: float = Field(ge=0)
    order_up_to: float = Field(ge=0)
    suggested_order_quantity: float = Field(ge=0)
