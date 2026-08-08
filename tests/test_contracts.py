from datetime import date

import pytest
from pydantic import ValidationError

from retail_forecasting.contracts import ForecastRecord


def make_forecast(**overrides: object) -> ForecastRecord:
    values = {
        "run_id": "run",
        "model_name": "xgboost",
        "origin_date": date(2016, 5, 22),
        "forecast_date": date(2016, 5, 23),
        "series_id": "item_store",
        "item_id": "item",
        "dept_id": "dept",
        "cat_id": "cat",
        "store_id": "store",
        "state_id": "CA",
        "horizon": 1,
        "yhat": 2.0,
        "q05": 1.0,
        "q50": 2.0,
        "q95": 4.0,
    }
    values.update(overrides)
    return ForecastRecord.model_validate(values)


def test_forecast_contract_accepts_monotonic_quantiles() -> None:
    record = make_forecast()
    assert record.q05 <= record.q50 <= record.q95


def test_forecast_contract_rejects_crossed_quantiles() -> None:
    with pytest.raises(ValidationError):
        make_forecast(q05=3.0, q50=2.0)


def test_forecast_contract_rejects_negative_demand() -> None:
    with pytest.raises(ValidationError):
        make_forecast(yhat=-1.0)
