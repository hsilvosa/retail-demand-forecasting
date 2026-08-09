"""Periodic-review inventory simulation driven by probabilistic forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.config import InventoryConfig, ProjectConfig
from retail_forecasting.data.spark import get_spark, table_path, write_delta


@dataclass(frozen=True)
class InventoryPolicy:
    lead_time_days: int
    review_period_days: int
    service_level: float
    bootstrap_paths: int
    fixed_order_cost: float
    annual_holding_rate: float
    stockout_price_multiplier: float
    seed: int = 42

    @classmethod
    def from_config(cls, config: InventoryConfig, seed: int) -> InventoryPolicy:
        return cls(**config.model_dump(), seed=seed)


def bootstrap_order_up_to(
    point_forecast: np.ndarray,
    residuals: np.ndarray,
    service_level: float,
    paths: int,
    seed: int,
) -> float:
    """Estimate protection-period demand while retaining residual dependence by blocks."""
    point = np.asarray(point_forecast, dtype=float)
    errors = np.asarray(residuals, dtype=float)
    if not len(point):
        return 0.0
    if errors.ndim == 1:
        errors = errors.reshape(-1, 1)
    if not errors.size:
        return float(point.sum())
    rng = np.random.default_rng(seed)
    sampled_rows = errors[rng.integers(0, len(errors), size=paths)]
    if sampled_rows.shape[1] < len(point):
        repeats = int(np.ceil(len(point) / sampled_rows.shape[1]))
        sampled_rows = np.tile(sampled_rows, (1, repeats))
    simulated = np.maximum(point[None, :] + sampled_rows[:, : len(point)], 0).sum(axis=1)
    return float(np.quantile(simulated, service_level))


def simulate_series(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    residual_blocks: np.ndarray,
    policy: InventoryPolicy,
    initial_on_hand: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate a lost-sales (R,S) policy for one SKU-store series."""
    prediction = forecast.sort_values("target_date").reset_index(drop=True)
    demand = actual.sort_values("target_date").reset_index(drop=True)
    if len(prediction) != len(demand):
        raise ValueError("forecast and actual must cover the same dates")
    protection = policy.lead_time_days + policy.review_period_days
    initial_target = bootstrap_order_up_to(
        prediction["yhat"].head(protection).to_numpy(),
        residual_blocks,
        policy.service_level,
        policy.bootstrap_paths,
        policy.seed,
    )
    on_hand = float(initial_target if initial_on_hand is None else initial_on_hand)
    arrivals: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for day in range(len(prediction)):
        row = prediction.iloc[day]
        arrived = arrivals.pop(day, 0.0)
        on_hand += arrived
        observed = float(demand.iloc[day]["target"])
        sales = min(on_hand, observed)
        lost_sales = observed - sales
        on_hand -= sales
        inventory_position = on_hand + sum(arrivals.values())
        order_quantity = 0.0
        order_up_to = np.nan
        if day % policy.review_period_days == 0:
            window = prediction["yhat"].iloc[day : day + protection].to_numpy(dtype=float)
            order_up_to = bootstrap_order_up_to(
                window,
                residual_blocks,
                policy.service_level,
                policy.bootstrap_paths,
                policy.seed + day,
            )
            order_quantity = max(0.0, order_up_to - inventory_position)
            if order_quantity > 0:
                arrival_day = day + policy.lead_time_days
                arrivals[arrival_day] = arrivals.get(arrival_day, 0.0) + order_quantity
        price = float(row.get("unit_price", row.get("target_sell_price", 0.0)) or 0.0)
        rows.append(
            {
                "target_date": row["target_date"],
                "demand": observed,
                "forecast": float(row["yhat"]),
                "arrivals": arrived,
                "sales": sales,
                "lost_sales": lost_sales,
                "ending_on_hand": on_hand,
                "inventory_position": inventory_position,
                "order_up_to": order_up_to,
                "order_quantity": order_quantity,
                "holding_cost": on_hand * price * policy.annual_holding_rate / 365,
                "ordering_cost": policy.fixed_order_cost if order_quantity > 0 else 0.0,
                "stockout_cost": lost_sales * price * policy.stockout_price_multiplier,
            }
        )
    daily = pd.DataFrame(rows)
    total_demand = max(float(daily["demand"].sum()), 1e-12)
    summary = {
        "fill_rate": float(daily["sales"].sum() / total_demand),
        "stockout_rate": float((daily["lost_sales"] > 0).mean()),
        "average_inventory": float(daily["ending_on_hand"].mean()),
        "order_count": float((daily["order_quantity"] > 0).sum()),
        "units_ordered": float(daily["order_quantity"].sum()),
        "lost_sales_units": float(daily["lost_sales"].sum()),
        "holding_cost": float(daily["holding_cost"].sum()),
        "ordering_cost": float(daily["ordering_cost"].sum()),
        "stockout_cost": float(daily["stockout_cost"].sum()),
    }
    summary["total_cost"] = (
        summary["holding_cost"] + summary["ordering_cost"] + summary["stockout_cost"]
    )
    return daily, summary


def _residual_blocks(backtests: pd.DataFrame) -> np.ndarray:
    residual = backtests.assign(residual=backtests["target"] - backtests["yhat"])
    matrix = residual.pivot_table(
        index=["fold_origin", "series_id"], columns="horizon", values="residual"
    )
    return matrix.fillna(0).to_numpy(dtype=float)


def build_recommendations(
    forecasts: pd.DataFrame, residuals: np.ndarray, policy: InventoryPolicy, run_id: str
) -> pd.DataFrame:
    recommendations = []
    protection = policy.lead_time_days + policy.review_period_days
    for series_id, group in forecasts.groupby("series_id", sort=False):
        ordered = group.sort_values("target_date")
        reorder_point = bootstrap_order_up_to(
            ordered["yhat"].head(policy.lead_time_days).to_numpy(),
            residuals,
            policy.service_level,
            policy.bootstrap_paths,
            policy.seed,
        )
        order_up_to = bootstrap_order_up_to(
            ordered["yhat"].head(protection).to_numpy(),
            residuals,
            policy.service_level,
            policy.bootstrap_paths,
            policy.seed,
        )
        synthetic_on_hand = reorder_point
        recommendations.append(
            {
                "run_id": run_id,
                "series_id": series_id,
                "as_of_date": ordered["origin_date"].iloc[0],
                "lead_time_days": policy.lead_time_days,
                "review_period_days": policy.review_period_days,
                "service_level": policy.service_level,
                "on_hand": synthetic_on_hand,
                "reorder_point": reorder_point,
                "order_up_to": order_up_to,
                "suggested_order_quantity": max(0.0, order_up_to - synthetic_on_hand),
                "assumption_source": "synthetic_configurable",
            }
        )
    return pd.DataFrame(recommendations)


def run_inventory(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    from pyspark.sql import functions as F

    spark = get_spark(config, "inventory")
    backtest_table = spark.read.format("delta").load(
        str(table_path(config, "gold", "backtest_forecasts"))
    )
    if "unit_price" not in backtest_table.columns:
        prices = (
            spark.read.format("delta")
            .load(str(table_path(config, "gold", "training_features")))
            .select(
                "series_id",
                "origin_day",
                "horizon",
                F.col("target_sell_price").alias("unit_price"),
            )
            .dropDuplicates(["series_id", "origin_day", "horizon"])
        )
        backtest_table = backtest_table.join(
            prices, ["series_id", "origin_day", "horizon"], "left"
        )
    backtests = backtest_table.toPandas()
    final_forecasts = spark.read.format("delta").load(
        str(table_path(config, "gold", "forecasts_bottom"))
    ).toPandas()
    latest_origin = int(backtests["fold_origin"].max())
    latest = backtests.loc[backtests["fold_origin"] == latest_origin].copy()
    winner = str(final_forecasts["model_name"].iloc[0])
    models = [name for name in (winner, "seasonal_naive") if name in latest["model_name"].unique()]
    policy = InventoryPolicy.from_config(config.inventory, config.seed)
    daily_outputs = []
    summary_outputs = []
    for model_name in models:
        model_backtests = latest.loc[latest["model_name"] == model_name].copy()
        residuals = _residual_blocks(
            backtests.loc[backtests["model_name"] == model_name]
        )
        for series_id, group in model_backtests.groupby("series_id", sort=False):
            simulated, summary = simulate_series(group, group, residuals, policy)
            simulated.insert(0, "series_id", series_id)
            simulated.insert(0, "model_name", model_name)
            simulated.insert(0, "run_id", run_id)
            daily_outputs.append(simulated)
            summary_outputs.append(
                {"run_id": run_id, "model_name": model_name, "series_id": series_id, **summary}
            )
    daily_frame = pd.concat(daily_outputs, ignore_index=True)
    summary_frame = pd.DataFrame(summary_outputs)
    winner_residuals = _residual_blocks(backtests.loc[backtests["model_name"] == winner])
    recommendations = build_recommendations(final_forecasts, winner_residuals, policy, run_id)
    write_delta(
        spark.createDataFrame(daily_frame), table_path(config, "gold", "inventory_daily")
    )
    write_delta(
        spark.createDataFrame(summary_frame), table_path(config, "gold", "inventory_kpis")
    )
    write_delta(
        spark.createDataFrame(recommendations),
        table_path(config, "gold", "inventory_recommendations"),
    )
    spark.stop()
    return {
        "run_id": run_id,
        "models": models,
        "daily_rows": len(daily_frame),
        "recommendations": len(recommendations),
        "tables": {
            "daily": str(table_path(config, "gold", "inventory_daily")),
            "kpis": str(table_path(config, "gold", "inventory_kpis")),
            "recommendations": str(table_path(config, "gold", "inventory_recommendations")),
        },
    }
