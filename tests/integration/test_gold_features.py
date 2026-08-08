import pytest
from pyspark.sql import SparkSession

from retail_forecasting.config import load_config
from retail_forecasting.data.gold import add_origin_features


@pytest.mark.integration
def test_features_only_use_values_before_current_day() -> None:
    spark = SparkSession.builder.master("local[1]").appName("feature-test").getOrCreate()
    source = spark.range(1, 70).selectExpr(
        "'a' AS series_id",
        "CAST(id AS INT) AS day_num",
        "CAST(id AS DOUBLE) AS units",
        "CAST(1 AS DOUBLE) AS sell_price",
    )
    featured = add_origin_features(source, load_config("test"))
    row = featured.filter("day_num = 60").first()
    assert row["lag_1"] == 59
    assert row["rolling_mean_7"] == pytest.approx(sum(range(53, 60)) / 7)
    spark.stop()
