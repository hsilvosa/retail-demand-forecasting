"""Bronze ingestion that preserves M5 source semantics and lineage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from retail_forecasting.config import ProjectConfig
from retail_forecasting.data.quality import csv_header, file_sha256, validate_source_files
from retail_forecasting.data.spark import get_spark, table_path, write_delta


def _schema_for(name: str, header: list[str]) -> StructType | None:
    if name.startswith("sales_train"):
        string_columns = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
        return StructType(
            [
                StructField(column, StringType() if column in string_columns else IntegerType(), True)
                for column in header
            ]
        )
    if name == "sell_prices.csv":
        return StructType(
            [
                StructField("store_id", StringType(), False),
                StructField("item_id", StringType(), False),
                StructField("wm_yr_wk", IntegerType(), False),
                StructField("sell_price", DoubleType(), True),
            ]
        )
    return None


def run_bronze(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    report = validate_source_files(config)
    if not report["valid"]:
        raise ValueError(f"Source validation failed: {report}")
    spark = get_spark(config, "bronze")
    ingested_at = datetime.now(UTC).isoformat()
    outputs: dict[str, Any] = {}
    for filename in config.data.required_files:
        source = config.paths.source / filename
        header = csv_header(source)
        schema = _schema_for(filename, header)
        reader = spark.read.option("header", "true").option("mode", "FAILFAST")
        frame = reader.schema(schema).csv(str(source)) if schema else reader.option(
            "inferSchema", "true"
        ).csv(str(source))
        checksum = file_sha256(source)
        frame = (
            frame.withColumn("_source_file", F.lit(filename))
            .withColumn("_source_sha256", F.lit(checksum))
            .withColumn("_ingested_at", F.lit(ingested_at).cast("timestamp"))
            .withColumn("_run_id", F.lit(run_id))
        )
        table = filename.removesuffix(".csv")
        destination = table_path(config, "bronze", table)
        write_delta(frame, destination)
        outputs[table] = {"path": str(destination), "rows": frame.count(), "sha256": checksum}
    spark.stop()
    return outputs
