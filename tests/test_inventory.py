import numpy as np
import pandas as pd

from retail_forecasting.inventory import (
    InventoryPolicy,
    bootstrap_order_up_to,
    simulate_series,
)


def policy() -> InventoryPolicy:
    return InventoryPolicy(
        lead_time_days=1,
        review_period_days=1,
        service_level=0.95,
        bootstrap_paths=100,
        fixed_order_cost=5,
        annual_holding_rate=0.2,
        stockout_price_multiplier=1,
        seed=42,
    )


def test_bootstrap_is_reproducible() -> None:
    forecast = np.array([2.0, 3.0])
    residuals = np.array([[-1.0, 1.0], [0.0, 2.0]])
    first = bootstrap_order_up_to(forecast, residuals, 0.95, 100, 42)
    second = bootstrap_order_up_to(forecast, residuals, 0.95, 100, 42)
    assert first == second


def test_inventory_daily_balance_and_cost_total() -> None:
    dates = pd.date_range("2025-01-01", periods=4)
    frame = pd.DataFrame(
        {
            "target_date": dates,
            "target": [2.0, 3.0, 1.0, 2.0],
            "yhat": [2.0, 2.0, 2.0, 2.0],
            "target_sell_price": [10.0] * 4,
        }
    )
    daily, summary = simulate_series(frame, frame, np.zeros((2, 4)), policy(), initial_on_hand=4)
    assert np.allclose(daily["sales"] + daily["lost_sales"], daily["demand"])
    assert (daily["ending_on_hand"] >= 0).all()
    assert np.isclose(
        summary["total_cost"],
        summary["holding_cost"] + summary["ordering_cost"] + summary["stockout_cost"],
    )
