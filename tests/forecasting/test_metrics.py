import numpy as np
import pandas as pd

from retail_forecasting.forecasting.metrics import hierarchical_wrmsse, point_metrics, rmsse


def test_point_metrics_are_zero_for_perfect_forecast() -> None:
    metrics = point_metrics([0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2])
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["wape"] == 0
    assert metrics["coverage"] == 1


def test_rmsse_ignores_leading_zero_history() -> None:
    assert rmsse([3, 4], [3, 4], [0, 0, 1, 2, 3]) == 0


def test_hierarchical_wrmsse_is_zero_for_perfect_bottom_up_forecast() -> None:
    dimensions = {
        "series_id": "item_CA_1",
        "item_id": "item",
        "dept_id": "dept",
        "cat_id": "cat",
        "store_id": "CA_1",
        "state_id": "CA",
    }
    history = pd.DataFrame(
        [
            {**dimensions, "day_num": day, "units": float(day), "sell_price": 2.0}
            for day in range(1, 10)
        ]
    )
    forecast = pd.DataFrame(
        [
            {**dimensions, "day_num": day, "target": float(day), "yhat": float(day)}
            for day in range(10, 13)
        ]
    )
    score, details = hierarchical_wrmsse(history, forecast, origin_day=9)
    assert score == 0
    assert set(details["level"]) == set(range(1, 13))


def test_point_metric_bias_preserves_direction() -> None:
    assert np.isclose(point_metrics([1, 1], [2, 2])["bias"], 1.0)
