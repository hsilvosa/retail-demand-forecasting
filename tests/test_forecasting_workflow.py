from datetime import date

import pandas as pd

from retail_forecasting.config import load_config
from retail_forecasting.forecasting.workflow import (
    BACKTEST_COLUMNS,
    _backtest_contract,
    _select_winner,
)


def _row(model_name: str, origin_date: object) -> dict[str, object]:
    return {
        "series_id": "item_store",
        "item_id": "item",
        "dept_id": "dept",
        "cat_id": "cat",
        "store_id": "store",
        "state_id": "CA",
        "origin_day": 1913,
        "fold_origin": 1913,
        "origin_date": origin_date,
        "day_num": 1914,
        "horizon": 1,
        "target_date": date(2016, 4, 25),
        "model_name": model_name,
        "target": 2.0,
        "yhat": 2.1,
        "q05": 1.0,
        "q50": 2.1,
        "q95": 3.5,
        "target_sell_price": 4.25,
    }


def test_backtest_contract_normalizes_missing_origin_dates() -> None:
    frames = [
        pd.DataFrame([_row("lightgbm", date(2016, 4, 24))]),
        pd.DataFrame([_row("nhits", float("nan"))]),
    ]

    result = _backtest_contract(frames)

    assert result.columns.tolist() == BACKTEST_COLUMNS
    assert result["origin_date"].isna().sum() == 0
    assert result["origin_date"].nunique() == 1
    assert pd.api.types.is_datetime64_any_dtype(result["origin_date"])
    assert result["unit_price"].eq(4.25).all()


def test_winner_is_best_candidate_that_passes_guardrails() -> None:
    summaries = {
        "accurate_but_biased": {
            "mean_wrmsse": 0.8,
            "bias": -0.08,
            "coverage": 0.90,
            "max_fold_degradation": 0.0,
        },
        "eligible": {
            "mean_wrmsse": 1.0,
            "bias": -0.02,
            "coverage": 0.89,
            "max_fold_degradation": 0.0,
        },
    }

    assert _select_winner(load_config("test"), summaries) == "eligible"
