"""Source and lakehouse data-quality checks."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from retail_forecasting.config import ProjectConfig

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

EXPECTED_COLUMNS = {
    "calendar.csv": {"date", "wm_yr_wk", "d", "event_name_1", "snap_CA"},
    "sell_prices.csv": {"store_id", "item_id", "wm_yr_wk", "sell_price"},
    "sales_train_evaluation.csv": {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d_1",
        "d_1941",
    },
    "sales_train_validation.csv": {"id", "item_id", "store_id", "d_1", "d_1913"},
    "sample_submission.csv": {"id", "F1", "F28"},
}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def validate_source_files(config: ProjectConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for name in config.data.required_files:
        path = config.paths.source / name
        exists = path.is_file() and path.stat().st_size > 0
        header = csv_header(path) if exists else []
        missing_columns = sorted(EXPECTED_COLUMNS.get(name, set()).difference(header))
        checks.append(
            {
                "file": name,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "columns": len(header),
                "missing_required_columns": missing_columns,
                "valid": exists and not missing_columns,
            }
        )
    return {"valid": all(check["valid"] for check in checks), "checks": checks}


def validate_silver_sales(frame: DataFrame) -> dict[str, Any]:
    from pyspark.sql import functions as F

    summary = frame.agg(
        F.count("*").alias("rows"),
        F.countDistinct("series_id").alias("series"),
        F.min("units").alias("min_units"),
        F.max("day_num").alias("max_day"),
        F.sum(F.col("date").isNull().cast("int")).alias("missing_dates"),
    ).first()
    duplicates = (
        frame.groupBy("series_id", "day_num").count().filter(F.col("count") > 1).limit(1).count()
    )
    if summary is None:
        raise ValueError("Silver sales summary returned no row")
    report = summary.asDict()
    report["duplicate_keys"] = duplicates
    report["valid"] = bool(
        report["rows"] > 0
        and report["series"] > 0
        and report["min_units"] >= 0
        and report["missing_dates"] == 0
        and duplicates == 0
    )
    return report
