"""Leakage-safe features and hierarchy tables for training and inference."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from retail_forecasting.config import ProjectConfig
from retail_forecasting.data.spark import get_spark, table_path, write_delta

HIERARCHY_LEVELS: tuple[tuple[str, ...], ...] = (
    (),
    ("state_id",),
    ("store_id",),
    ("cat_id",),
    ("dept_id",),
    ("state_id", "cat_id"),
    ("state_id", "dept_id"),
    ("store_id", "cat_id"),
    ("store_id", "dept_id"),
    ("item_id",),
    ("state_id", "item_id"),
    ("store_id", "item_id"),
)


def add_origin_features(daily: DataFrame, config: ProjectConfig) -> DataFrame:
    """Build features known at the end of the forecast origin day.

    ``lag_1`` intentionally denotes the observation on the origin itself. This
    convention avoids dropping the most recent known sale while keeping every
    feature available before the first forecast target (origin + 1).
    """
    history = Window.partitionBy("series_id").orderBy("day_num")
    frame = daily
    for lag in config.features.lags:
        offset = lag - 1
        value = F.col("units") if offset == 0 else F.lag("units", offset).over(history)
        frame = frame.withColumn(f"lag_{lag}", value)
    for width in config.features.rolling_windows:
        trailing = history.rowsBetween(-(width - 1), 0)
        frame = (
            frame.withColumn(f"rolling_mean_{width}", F.avg("units").over(trailing))
            .withColumn(f"rolling_std_{width}", F.stddev_pop("units").over(trailing))
            .withColumn(f"rolling_max_{width}", F.max("units").over(trailing))
            .withColumn(
                f"nonzero_rate_{width}",
                F.avg((F.col("units") > 0).cast("double")).over(trailing),
            )
        )
    all_history = history.rowsBetween(Window.unboundedPreceding, 0)
    price_history = history.rowsBetween(-27, 0)
    return (
        frame.withColumn(
            "last_sale_day",
            F.max(F.when(F.col("units") > 0, F.col("day_num"))).over(all_history),
        )
        .withColumn(
            "first_sale_day",
            F.min(F.when(F.col("units") > 0, F.col("day_num"))).over(all_history),
        )
        .withColumn("days_since_last_sale", F.col("day_num") - F.col("last_sale_day"))
        .withColumn("days_since_first_sale", F.col("day_num") - F.col("first_sale_day"))
        .withColumn("origin_sell_price", F.col("sell_price"))
        .withColumn("origin_price_mean_28", F.avg("sell_price").over(price_history))
        .withColumn(
            "short_long_trend",
            F.col("rolling_mean_7") / F.greatest(F.col("rolling_mean_28"), F.lit(0.1)),
        )
    )


def _add_derived_direct_features(frame: DataFrame) -> DataFrame:
    valid_origin_price = F.col("origin_sell_price").isNotNull() & (
        F.col("origin_sell_price") > 0
    )
    valid_mean_price = F.col("origin_price_mean_28").isNotNull() & (
        F.col("origin_price_mean_28") > 0
    )
    return (
        frame.withColumn("price_missing", F.col("target_sell_price").isNull().cast("int"))
        .withColumn("origin_price_missing", (~valid_origin_price).cast("int"))
        .withColumn(
            "price_ratio_to_origin",
            F.when(valid_origin_price, F.col("target_sell_price") / F.col("origin_sell_price"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "price_ratio_to_mean_28",
            F.when(
                valid_mean_price,
                F.col("target_sell_price") / F.col("origin_price_mean_28"),
            ).otherwise(F.lit(0.0)),
        )
        .fillna(
            0.0,
            subset=["target_sell_price", "origin_sell_price", "origin_price_mean_28"],
        )
    )


def _training_origins(config: ProjectConfig) -> list[int]:
    last = max(config.data.backtest_origins)
    first = max(
        max(config.features.lags),
        max(config.features.target_lags),
        last - config.data.history_days,
    )
    regular = list(range(first, last + 1, config.data.training_origin_step))
    return sorted(set(regular + config.data.backtest_origins))


def build_direct_features(daily: DataFrame, config: ProjectConfig) -> DataFrame:
    featured = add_origin_features(daily, config)
    origins = featured.filter(F.col("day_num").isin(_training_origins(config))).alias("origin")
    target_history = Window.partitionBy("series_id").orderBy("day_num")
    target_frame = daily
    for lag in config.features.target_lags:
        target_frame = target_frame.withColumn(
            f"target_lag_{lag}", F.lag("units", lag).over(target_history)
        )
    targets = target_frame.select(
        F.col("series_id").alias("target_series_id"),
        F.col("day_num").alias("target_day_num"),
        F.col("date").alias("target_date"),
        F.col("units").alias("target"),
        F.col("sell_price").alias("target_sell_price"),
        F.col("wday").alias("target_wday"),
        F.col("month").alias("target_month"),
        F.col("event_type_1").alias("target_event_type"),
        F.col("snap_CA").alias("target_snap_CA"),
        F.col("snap_TX").alias("target_snap_TX"),
        F.col("snap_WI").alias("target_snap_WI"),
        *[F.col(f"target_lag_{lag}") for lag in config.features.target_lags],
    ).alias("target")
    horizons = daily.sparkSession.range(1, config.data.horizon + 1).select(
        F.col("id").cast("int").alias("horizon")
    )
    expanded = origins.crossJoin(horizons).withColumn(
        "target_day", F.col("origin.day_num") + F.col("horizon")
    )
    selected = (
        expanded.join(
            targets,
            (F.col("origin.series_id") == F.col("target.target_series_id"))
            & (F.col("target_day") == F.col("target.target_day_num")),
            "inner",
        )
        .select(
            F.col("origin.series_id"),
            F.col("origin.item_id"),
            F.col("origin.dept_id"),
            F.col("origin.cat_id"),
            F.col("origin.store_id"),
            F.col("origin.state_id"),
            F.col("origin.day_num").alias("origin_day"),
            F.col("origin.date").alias("origin_date"),
            "horizon",
            "target_date",
            "target",
            "target_sell_price",
            "target_wday",
            "target_month",
            "target_event_type",
            "target_snap_CA",
            "target_snap_TX",
            "target_snap_WI",
            *[F.col(f"target.target_lag_{lag}") for lag in config.features.target_lags],
            *[F.col(f"origin.lag_{lag}") for lag in config.features.lags],
            *[
                F.col(f"origin.{stat}_{width}")
                for width in config.features.rolling_windows
                for stat in ("rolling_mean", "rolling_std", "rolling_max", "nonzero_rate")
            ],
            F.col("origin.days_since_last_sale"),
            F.col("origin.days_since_first_sale"),
            F.col("origin.short_long_trend"),
            F.col("origin.origin_sell_price"),
            F.col("origin.origin_price_mean_28"),
        )
    )
    return _add_derived_direct_features(selected)


def _add_future_target_lags(
    joined: DataFrame, daily: DataFrame, config: ProjectConfig
) -> DataFrame:
    lag_values = config.features.target_lags
    requested = (
        joined.select(
            F.col("origin.series_id").alias("_lag_series_id"),
            F.col("target_day").alias("_lag_target_day"),
        )
        .distinct()
        .withColumn("_target_lag", F.explode(F.array(*[F.lit(value) for value in lag_values])))
        .withColumn("_history_day", F.col("_lag_target_day") - F.col("_target_lag"))
    )
    history = daily.select(
        F.col("series_id").alias("_history_series_id"),
        F.col("day_num").alias("_history_day_num"),
        F.col("units").alias("_history_units"),
    )
    pivoted = (
        requested.join(
            history,
            (F.col("_lag_series_id") == F.col("_history_series_id"))
            & (F.col("_history_day") == F.col("_history_day_num")),
            "left",
        )
        .groupBy("_lag_series_id", "_lag_target_day")
        .pivot("_target_lag", lag_values)
        .agg(F.first("_history_units"))
        .select(
            "_lag_series_id",
            "_lag_target_day",
            *[F.col(str(lag)).alias(f"target_lag_{lag}") for lag in lag_values],
        )
    )
    return joined.join(
        pivoted,
        (F.col("origin.series_id") == F.col("_lag_series_id"))
        & (F.col("target_day") == F.col("_lag_target_day")),
        "left",
    )


def build_forecast_features(
    daily: DataFrame, calendar: DataFrame, prices: DataFrame, config: ProjectConfig
) -> DataFrame:
    """Create future-known rows at the final origin without using future demand."""
    featured = add_origin_features(daily, config)
    origin = featured.filter(F.col("day_num") == config.data.forecast_origin_day).alias("origin")
    future_calendar = calendar.select(
        F.col("day_num").alias("target_day_num"),
        F.col("date").alias("target_date"),
        "wm_yr_wk",
        F.col("wday").alias("target_wday"),
        F.col("month").alias("target_month"),
        F.col("event_type_1").alias("target_event_type"),
        F.col("snap_CA").alias("target_snap_CA"),
        F.col("snap_TX").alias("target_snap_TX"),
        F.col("snap_WI").alias("target_snap_WI"),
    ).alias("calendar")
    horizons = daily.sparkSession.range(1, config.data.horizon + 1).select(
        F.col("id").cast("int").alias("horizon")
    )
    expanded = origin.crossJoin(horizons).withColumn(
        "target_day", F.col("origin.day_num") + F.col("horizon")
    )
    joined = expanded.join(
        future_calendar, F.col("target_day") == F.col("calendar.target_day_num"), "inner"
    ).join(
        prices.alias("prices"),
        (F.col("origin.store_id") == F.col("prices.store_id"))
        & (F.col("origin.item_id") == F.col("prices.item_id"))
        & (F.col("calendar.wm_yr_wk") == F.col("prices.wm_yr_wk")),
        "left",
    )
    joined = _add_future_target_lags(joined, daily, config)
    selected = joined.select(
        F.col("origin.series_id"),
        F.col("origin.item_id"),
        F.col("origin.dept_id"),
        F.col("origin.cat_id"),
        F.col("origin.store_id"),
        F.col("origin.state_id"),
        F.col("origin.day_num").alias("origin_day"),
        F.col("origin.date").alias("origin_date"),
        "horizon",
        "target_date",
        F.col("prices.sell_price").alias("target_sell_price"),
        "target_wday",
        "target_month",
        "target_event_type",
        "target_snap_CA",
        "target_snap_TX",
        "target_snap_WI",
        *[F.col(f"target_lag_{lag}") for lag in config.features.target_lags],
        *[F.col(f"origin.lag_{lag}") for lag in config.features.lags],
        *[
            F.col(f"origin.{stat}_{width}")
            for width in config.features.rolling_windows
            for stat in ("rolling_mean", "rolling_std", "rolling_max", "nonzero_rate")
        ],
        F.col("origin.days_since_last_sale"),
        F.col("origin.days_since_first_sale"),
        F.col("origin.short_long_trend"),
        F.col("origin.origin_sell_price"),
        F.col("origin.origin_price_mean_28"),
    )
    return _add_derived_direct_features(selected)


def build_hierarchy(daily: DataFrame) -> DataFrame:
    dimensions = daily.select(
        "series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"
    ).distinct()
    frames = []
    for level, columns in enumerate(HIERARCHY_LEVELS, start=1):
        if columns:
            node_id = F.concat_ws("/", *[F.col(column) for column in columns])
        else:
            node_id = F.lit("Total")
        frames.append(
            dimensions.select(
                "series_id",
                F.lit(level).alias("level"),
                F.lit("/".join(columns) if columns else "total").alias("level_name"),
                node_id.alias("node_id"),
            )
        )
    result = frames[0]
    for frame in frames[1:]:
        result = result.unionByName(frame)
    return result


def run_gold(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    spark = get_spark(config, "gold")
    daily = spark.read.format("delta").load(str(table_path(config, "silver", "sales_daily")))
    calendar = spark.read.format("delta").load(str(table_path(config, "silver", "calendar")))
    prices = spark.read.format("delta").load(str(table_path(config, "silver", "sell_prices")))
    direct = build_direct_features(daily, config).withColumn("_run_id", F.lit(run_id))
    future = build_forecast_features(daily, calendar, prices, config).withColumn(
        "_run_id", F.lit(run_id)
    )
    hierarchy = build_hierarchy(daily).withColumn("_run_id", F.lit(run_id))
    write_delta(direct, table_path(config, "gold", "training_features"), ["origin_day"])
    write_delta(future, table_path(config, "gold", "forecast_features"))
    write_delta(hierarchy, table_path(config, "gold", "hierarchy"), ["level"])
    outputs = {
        "training_features": str(table_path(config, "gold", "training_features")),
        "forecast_features": str(table_path(config, "gold", "forecast_features")),
        "hierarchy": str(table_path(config, "gold", "hierarchy")),
        "feature_rows": direct.count(),
        "forecast_rows": future.count(),
        "hierarchy_rows": hierarchy.count(),
    }
    spark.stop()
    return outputs
