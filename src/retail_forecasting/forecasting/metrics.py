"""Point, interval, and M5 hierarchy evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def point_metrics(
    actual: Any,
    predicted: Any,
    q05: Any | None = None,
    q95: Any | None = None,
) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    error = yhat - y
    denominator = max(float(np.abs(y).sum()), 1e-12)
    result = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(np.abs(error).sum() / denominator),
        "bias": float(error.sum() / denominator),
    }
    if q05 is not None and q95 is not None:
        low = np.asarray(q05, dtype=float)
        high = np.asarray(q95, dtype=float)
        result["coverage"] = float(np.mean((y >= low) & (y <= high)))
        result["mean_interval_width"] = float(np.mean(high - low))
    return result


def summarize_backtest_points(backtests: pd.DataFrame) -> pd.DataFrame:
    """Average point and interval metrics across temporal folds for each model."""
    required = {"model_name", "fold_origin", "target", "yhat", "q05", "q95"}
    metric_names = ["mae", "rmse", "wape", "bias", "coverage", "mean_interval_width"]
    missing = required.difference(backtests.columns)
    if missing:
        raise ValueError(f"backtests are missing columns: {sorted(missing)}")
    if backtests.empty:
        return pd.DataFrame(columns=["model_name", *metric_names])
    rows: list[dict[str, float | str]] = []
    for (model_name, _), fold in backtests.groupby(
        ["model_name", "fold_origin"], sort=False
    ):
        rows.append(
            {
                "model_name": str(model_name),
                **point_metrics(fold["target"], fold["yhat"], fold["q05"], fold["q95"]),
            }
        )
    return pd.DataFrame(rows).groupby("model_name", as_index=False)[metric_names].mean()


def rmsse(actual: Any, predicted: Any, history: Any) -> float:
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    train = np.asarray(history, dtype=float)
    nonzero = np.flatnonzero(train != 0)
    if len(nonzero):
        train = train[nonzero[0] :]
    scale = float(np.mean(np.square(np.diff(train)))) if len(train) > 1 else 0.0
    if scale <= 0:
        return 0.0 if np.allclose(y, yhat) else float("inf")
    return float(np.sqrt(np.mean(np.square(y - yhat)) / scale))


def _node_id(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if not columns:
        return pd.Series("Total", index=frame.index)
    return frame.loc[:, list(columns)].astype(str).agg("/".join, axis=1)


def hierarchical_wrmsse(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    origin_day: int,
    revenue_window: int = 28,
) -> tuple[float, pd.DataFrame]:
    """Compute the official bottom-up weighted RMSSE across all 12 M5 levels."""
    from retail_forecasting.data.gold import HIERARCHY_LEVELS

    required_history = {"day_num", "units", "sell_price", *sum(HIERARCHY_LEVELS, ())}
    required_forecast = {"day_num", "target", "yhat", *sum(HIERARCHY_LEVELS, ())}
    if not required_history.issubset(history.columns) or not required_forecast.issubset(
        forecast.columns
    ):
        raise ValueError("history or forecast is missing hierarchy columns")
    level_results: list[pd.DataFrame] = []
    hist = history.loc[history["day_num"] <= origin_day].copy()
    recent = hist.loc[hist["day_num"] > origin_day - revenue_window].copy()
    recent["revenue"] = recent["units"] * recent["sell_price"].fillna(0)
    for level, columns in enumerate(HIERARCHY_LEVELS, start=1):
        level_history = hist.assign(node_id=_node_id(hist, columns))
        level_forecast = forecast.assign(node_id=_node_id(forecast, columns))
        level_recent = recent.assign(node_id=_node_id(recent, columns))
        history_nodes = (
            level_history.groupby(["node_id", "day_num"], as_index=False)["units"].sum()
        )
        forecast_nodes = level_forecast.groupby(["node_id", "day_num"], as_index=False)[
            ["target", "yhat"]
        ].sum()
        weights = level_recent.groupby("node_id")["revenue"].sum()
        weights = weights / max(float(weights.sum()), 1e-12) / len(HIERARCHY_LEVELS)
        rows = []
        for node_id, evaluated in forecast_nodes.groupby("node_id"):
            train = history_nodes.loc[history_nodes["node_id"] == node_id, "units"]
            score = rmsse(
                evaluated["target"].to_numpy(),
                evaluated["yhat"].to_numpy(),
                train.to_numpy(),
            )
            rows.append(
                {
                    "level": level,
                    "level_name": "/".join(columns) if columns else "total",
                    "node_id": node_id,
                    "rmsse": score,
                    "weight": float(weights.get(node_id, 0.0)),
                }
            )
        level_results.append(pd.DataFrame(rows))
    details = pd.concat(level_results, ignore_index=True)
    details["weighted_rmsse"] = details["rmsse"] * details["weight"]
    return float(details["weighted_rmsse"].sum()), details


def hierarchical_wrmsse_spark(
    history: DataFrame, forecast: DataFrame, origin_day: int
) -> tuple[float, DataFrame]:
    """Distributed equivalent used by the full profile without collecting sales history."""
    from functools import reduce

    from pyspark.sql import Window
    from pyspark.sql import functions as F

    from retail_forecasting.data.gold import HIERARCHY_LEVELS

    history_frame = history.filter(F.col("day_num") <= origin_day).cache()
    results = []
    for level, columns in enumerate(HIERARCHY_LEVELS, start=1):
        node = (
            F.concat_ws("/", *[F.col(column) for column in columns])
            if columns
            else F.lit("Total")
        )
        hist = history_frame.withColumn("node_id", node).groupBy("node_id", "day_num").agg(
            F.sum("units").alias("units"),
            F.sum(F.col("units") * F.coalesce("sell_price", F.lit(0.0))).alias("revenue"),
        )
        evaluated = forecast.withColumn("node_id", node).groupBy("node_id", "day_num").agg(
            F.sum("target").alias("target"), F.sum("yhat").alias("yhat")
        )
        first_sale = hist.filter(F.col("units") > 0).groupBy("node_id").agg(
            F.min("day_num").alias("first_sale_day")
        )
        scale_window = Window.partitionBy("node_id").orderBy("day_num")
        scales = (
            hist.join(first_sale, "node_id", "left")
            .filter(F.col("day_num") >= F.col("first_sale_day"))
            .withColumn("previous", F.lag("units").over(scale_window))
            .filter(F.col("previous").isNotNull())
            .groupBy("node_id")
            .agg(F.avg(F.pow(F.col("units") - F.col("previous"), 2)).alias("scale"))
        )
        weights = hist.filter(F.col("day_num") > origin_day - 28).groupBy("node_id").agg(
            F.sum("revenue").alias("node_revenue")
        )
        total_revenue = weights.agg(F.sum("node_revenue").alias("total_revenue"))
        weights = weights.crossJoin(total_revenue).withColumn(
            "weight",
            F.col("node_revenue")
            / F.greatest(F.col("total_revenue"), F.lit(1e-12))
            / F.lit(len(HIERARCHY_LEVELS)),
        )
        scores = (
            evaluated.groupBy("node_id")
            .agg(F.avg(F.pow(F.col("target") - F.col("yhat"), 2)).alias("mse"))
            .join(scales, "node_id", "left")
            .join(weights.select("node_id", "weight"), "node_id", "left")
            .withColumn(
                "rmsse",
                F.when(F.col("scale") > 0, F.sqrt(F.col("mse") / F.col("scale"))).otherwise(
                    F.lit(0.0)
                ),
            )
            .withColumn("weighted_rmsse", F.col("rmsse") * F.col("weight"))
            .withColumn("level", F.lit(level))
            .withColumn("level_name", F.lit("/".join(columns) if columns else "total"))
            .select("level", "level_name", "node_id", "rmsse", "weight", "weighted_rmsse")
        )
        results.append(scores)
    details = reduce(lambda left, right: left.unionByName(right), results)
    score_row = details.agg(F.sum("weighted_rmsse").alias("wrmsse")).first()
    if score_row is None:
        raise ValueError("WRMSSE aggregation returned no row")
    score = score_row["wrmsse"]
    history_frame.unpersist()
    return float(score), details
