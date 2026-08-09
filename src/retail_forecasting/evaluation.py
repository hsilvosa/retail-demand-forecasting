"""Post-hoc evaluation of persisted forecasts without retraining models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from retail_forecasting.config import ProjectConfig
from retail_forecasting.data.spark import get_spark, table_path
from retail_forecasting.forecasting.metrics import (
    summarize_backtest_points,
    summarize_bottom_rmsse_artifacts,
    summarize_wape_by_granularity,
)


def _inventory_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    return inventory.groupby("model_name", as_index=False).agg(
        fill_rate=("fill_rate", "mean"),
        stockout_rate=("stockout_rate", "mean"),
        average_inventory=("average_inventory", "mean"),
        lost_sales_units=("lost_sales_units", "sum"),
        total_cost=("total_cost", "sum"),
    )


def evaluate_stored_results(
    config: ProjectConfig, mlflow_run_id: str | None = None
) -> dict[str, Any]:
    """Return accuracy, uncertainty, and inventory metrics from stored Delta outputs."""
    spark = get_spark(config, "stored-evaluation")
    backtests = spark.read.format("delta").load(
        str(table_path(config, "gold", "backtest_forecasts"))
    ).toPandas()
    model_metrics = spark.read.format("delta").load(
        str(table_path(config, "gold", "model_metrics"))
    ).toPandas()
    forecasts = spark.read.format("delta").load(
        str(table_path(config, "gold", "forecasts_bottom"))
    ).select("model_name").first()
    inventory = spark.read.format("delta").load(
        str(table_path(config, "gold", "inventory_kpis"))
    ).toPandas()
    spark.stop()
    if forecasts is None:
        raise ValueError("stored forecasts do not identify a champion")

    points = summarize_backtest_points(backtests)
    rmsse = summarize_bottom_rmsse_artifacts(config.paths.artifacts / "backtests")
    granularities = summarize_wape_by_granularity(backtests).pivot(
        index="model_name", columns="granularity", values="wape"
    ).reset_index()
    accuracy = (
        model_metrics[["model_name", "mean_wrmsse"]]
        .merge(points, on="model_name", how="left")
        .merge(rmsse, on="model_name", how="left")
        .merge(granularities, on="model_name", how="left")
        .sort_values("mean_wrmsse")
    )
    inventory_summary = _inventory_summary(inventory)
    champion_name = str(forecasts["model_name"])
    champion_accuracy = accuracy.loc[accuracy["model_name"] == champion_name].iloc[0]
    champion_inventory = inventory_summary.loc[
        inventory_summary["model_name"] == champion_name
    ].iloc[0]

    if mlflow_run_id:
        import mlflow

        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        client = mlflow.MlflowClient()
        for key in (
            "bottom_rmsse",
            "mean_pinball_loss",
            "pinball_q05",
            "pinball_q50",
            "pinball_q95",
        ):
            client.log_metric(mlflow_run_id, f"posthoc_{key}", float(champion_accuracy[key]))
        for key in (
            "fill_rate",
            "stockout_rate",
            "average_inventory",
            "lost_sales_units",
            "total_cost",
        ):
            client.log_metric(mlflow_run_id, f"inventory_{key}", float(champion_inventory[key]))

    return {
        "champion": champion_name,
        "accuracy": accuracy.to_dict(orient="records"),
        "inventory": inventory_summary.to_dict(orient="records"),
        "mlflow_run_id": mlflow_run_id,
        "trained_models": 0,
    }
