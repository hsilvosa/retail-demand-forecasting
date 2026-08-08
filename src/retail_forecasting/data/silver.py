"""Normalize wide M5 inputs into an analytics-ready daily sales table."""

from __future__ import annotations

import json
from typing import Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from retail_forecasting.config import PROJECT_ROOT, ProjectConfig
from retail_forecasting.data.quality import validate_silver_sales
from retail_forecasting.data.spark import get_spark, table_path, write_delta


def _sample_series(frame: DataFrame, config: ProjectConfig) -> DataFrame:
    sample_per_state = config.data.sample_per_state
    if sample_per_state is None:
        return frame
    rank_window = Window.partitionBy("state_id").orderBy(F.xxhash64("id", F.lit(config.seed)))
    return (
        frame.withColumn("_sample_rank", F.row_number().over(rank_window))
        .filter(F.col("_sample_rank") <= sample_per_state)
        .drop("_sample_rank")
    )


def _normalize_sales(wide: DataFrame, config: ProjectConfig) -> DataFrame:
    id_columns = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
    day_columns = tuple(sorted(
        (column for column in wide.columns if column.startswith("d_")),
        key=lambda value: int(value.removeprefix("d_")),
    ))
    selected = _sample_series(wide.select(*id_columns, *day_columns), config)
    return (
        selected.unpivot(id_columns, day_columns, "d", "units")
        .withColumn("day_num", F.regexp_extract("d", r"d_(\d+)", 1).cast("int"))
        .withColumn("series_id", F.concat_ws("_", "item_id", "store_id"))
        .withColumn("units", F.col("units").cast("double"))
    )


def run_silver(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    spark = get_spark(config, "silver")
    bronze = lambda name: spark.read.format("delta").load(  # noqa: E731
        str(table_path(config, "bronze", name))
    )
    calendar = (
        bronze("calendar")
        .drop("_source_file", "_source_sha256", "_ingested_at", "_run_id")
        .withColumn("date", F.to_date("date"))
        .withColumn("day_num", F.regexp_extract("d", r"d_(\d+)", 1).cast("int"))
    )
    prices = bronze("sell_prices").drop(
        "_source_file", "_source_sha256", "_ingested_at", "_run_id"
    )
    sales = _normalize_sales(bronze("sales_train_evaluation"), config)
    sales.createOrReplaceTempView("normalized_sales")
    calendar.createOrReplaceTempView("silver_calendar")
    prices.createOrReplaceTempView("silver_prices")
    silver_sql = (PROJECT_ROOT / "sql/silver_sales_daily.sql").read_text(encoding="utf-8")
    daily = spark.sql(silver_sql).withColumn("_run_id", F.lit(run_id))
    quality = validate_silver_sales(daily)
    if not quality["valid"]:
        raise ValueError(f"Silver validation failed: {json.dumps(quality)}")
    outputs = {
        "sales_daily": str(table_path(config, "silver", "sales_daily")),
        "calendar": str(table_path(config, "silver", "calendar")),
        "sell_prices": str(table_path(config, "silver", "sell_prices")),
        "quality": quality,
    }
    write_delta(daily, table_path(config, "silver", "sales_daily"), ["state_id", "year"])
    write_delta(calendar, table_path(config, "silver", "calendar"), ["year"])
    write_delta(prices, table_path(config, "silver", "sell_prices"), ["store_id"])
    spark.stop()
    return outputs
