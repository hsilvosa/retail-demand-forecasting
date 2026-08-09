import numpy as np
import pandas as pd

from retail_forecasting.forecasting.metrics import (
    hierarchical_wrmsse,
    pinball_loss,
    point_metrics,
    rmsse,
    summarize_backtest_points,
    summarize_bottom_rmsse_artifacts,
    summarize_wape_by_granularity,
)


def test_point_metrics_are_zero_for_perfect_forecast() -> None:
    metrics = point_metrics(
        [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]
    )
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["wape"] == 0
    assert metrics["coverage"] == 1
    assert metrics["mean_pinball_loss"] == 0


def test_pinball_loss_penalizes_errors_asymmetrically() -> None:
    assert np.isclose(pinball_loss([10.0], [8.0], 0.95), 1.9)
    assert np.isclose(pinball_loss([10.0], [12.0], 0.95), 0.1)


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


def test_backtest_point_metrics_are_averaged_across_folds() -> None:
    backtests = pd.DataFrame(
        {
            "model_name": ["model_a"] * 4,
            "fold_origin": [1, 1, 2, 2],
            "target": [1.0, 3.0, 2.0, 4.0],
            "yhat": [2.0, 2.0, 2.0, 4.0],
            "q05": [0.0, 1.0, 1.0, 3.0],
            "q50": [2.0, 2.0, 2.0, 4.0],
            "q95": [3.0, 4.0, 3.0, 5.0],
        }
    )

    summary = summarize_backtest_points(backtests).iloc[0]

    assert np.isclose(summary["mae"], 0.5)
    assert np.isclose(summary["wape"], 0.25)
    assert np.isclose(summary["bias"], 0.0)
    assert np.isclose(summary["coverage"], 1.0)


def test_empty_backtests_return_an_empty_metric_contract() -> None:
    columns = ["model_name", "fold_origin", "target", "yhat", "q05", "q50", "q95"]

    summary = summarize_backtest_points(pd.DataFrame(columns=columns))

    assert summary.empty
    assert "wape" in summary.columns


def test_granularity_metrics_aggregate_before_calculating_wape() -> None:
    backtests = pd.DataFrame(
        {
            "model_name": ["model_a"] * 4,
            "fold_origin": [1] * 4,
            "series_id": ["a", "b", "a", "b"],
            "store_id": ["store"] * 4,
            "cat_id": ["cat"] * 4,
            "target_date": pd.to_datetime(["2025-01-01"] * 2 + ["2025-01-02"] * 2),
            "horizon": [1, 1, 2, 2],
            "target": [0.0, 2.0, 2.0, 0.0],
            "yhat": [1.0, 1.0, 1.0, 1.0],
        }
    )

    summary = summarize_wape_by_granularity(backtests).set_index("granularity")

    assert np.isclose(summary.loc["sku_store_day", "wape"], 1.0)
    assert np.isclose(summary.loc["store_day", "wape"], 0.0)
    assert np.isclose(summary.loc["sku_store_28d", "wape"], 0.0)


def test_bottom_rmsse_is_read_from_fold_artifacts(tmp_path) -> None:
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()
    pd.DataFrame(
        {"level": [11, 12, 12], "rmsse": [9.0, 0.5, 1.5]}
    ).to_parquet(model_dir / "d_1-wrmsse.parquet", index=False)

    summary = summarize_bottom_rmsse_artifacts(tmp_path).iloc[0]

    assert summary["model_name"] == "model_a"
    assert np.isclose(summary["bottom_rmsse"], 1.0)
