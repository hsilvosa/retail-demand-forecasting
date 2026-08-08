"""Spark and Delta helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from retail_forecasting.config import ProjectConfig

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def get_spark(config: ProjectConfig, app_name: str) -> SparkSession:
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(f"{config.project_name}-{app_name}")
        .master(config.spark.master)
        .config("spark.driver.memory", config.spark.driver_memory)
        .config("spark.executor.memory", config.spark.executor_memory)
        .config("spark.sql.shuffle.partitions", str(config.spark.shuffle_partitions))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def table_path(config: ProjectConfig, layer: str, table: str) -> Path:
    return (config.paths.lakehouse / layer / table).resolve()


def write_delta(
    frame: DataFrame, path: Path, partition_by: list[str] | None = None
) -> None:
    writer = frame.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(str(path))


def read_delta(config: ProjectConfig, layer: str, table: str) -> DataFrame:
    spark = get_spark(config, f"read-{layer}-{table}")
    return spark.read.format("delta").load(str(table_path(config, layer, table)))
