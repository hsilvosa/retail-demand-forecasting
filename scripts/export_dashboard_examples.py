"""Export compact README examples from the persisted dashboard data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from retail_forecasting.config import load_config
from retail_forecasting.data.spark import get_spark, table_path
from retail_forecasting.forecasting.metrics import summarize_wape_by_granularity

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

GREEN = "#20744a"
AMBER = "#d4932f"
BLUE = "#496d8c"
RED = "#a94b45"
GRAY = "#6b665f"
MODEL_COLORS = {
    "xgboost": GREEN,
    "lightgbm": BLUE,
    "moving_average": AMBER,
    "seasonal_naive": GRAY,
    "nhits": RED,
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "moving_average": "Moving average",
    "seasonal_naive": "Seasonal naive",
    "nhits": "N-HiTS (historical)",
}


def _read_gold(spark: Any, config: Any, name: str) -> pd.DataFrame:
    return spark.read.format("delta").load(str(table_path(config, "gold", name))).toPandas()


def _save(figure: Any, path: Path) -> None:
    figure.savefig(path, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _forecast_example(forecasts: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    selected_series = str(forecasts.groupby("series_id")["q50"].sum().idxmax())
    example = forecasts.loc[forecasts["series_id"] == selected_series].sort_values(
        "target_date"
    )
    dates = pd.to_datetime(example["target_date"])
    q05 = example["q05"].astype(float).to_numpy()
    q50 = example["q50"].astype(float).to_numpy()
    q95 = example["q95"].astype(float).to_numpy()

    figure, axis = plt.subplots(figsize=(10.2, 4.2))
    axis.fill_between(dates, q05, q95, color=GREEN, alpha=0.16, label="90% interval")
    axis.plot(dates, q50, color=GREEN, linewidth=2.2, label="Median forecast")
    axis.set_title(f"Forecast explorer example: {selected_series}", loc="left", weight="bold")
    axis.set_ylabel("Daily units")
    axis.set_xlabel("Target date")
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    _save(figure, output_dir / "dashboard-forecast-example.png")
    return {
        "series_id": selected_series,
        "store_id": str(example["store_id"].iloc[0]),
        "category": str(example["cat_id"].iloc[0]),
        "median_28d_units": float(q50.sum()),
        "peak_daily_median": float(q50.max()),
    }


def _model_comparison(
    model_metrics: pd.DataFrame, backtests: pd.DataFrame, output_dir: Path
) -> list[dict[str, Any]]:
    store_wape = summarize_wape_by_granularity(backtests).loc[
        lambda frame: frame["granularity"] == "store_day", ["model_name", "wape"]
    ]
    comparison = (
        model_metrics[["model_name", "mean_wrmsse"]]
        .merge(store_wape, on="model_name", how="left")
        .sort_values("mean_wrmsse")
        .reset_index(drop=True)
    )
    labels = [MODEL_LABELS.get(model, model) for model in comparison["model_name"]]
    colors = [MODEL_COLORS.get(model, GRAY) for model in comparison["model_name"]]
    positions = np.arange(len(comparison))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].barh(positions, comparison["mean_wrmsse"], color=colors)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("WRMSSE, lower is better")
    axes[0].set_title("Hierarchy-weighted accuracy", loc="left", weight="bold")
    axes[0].grid(axis="x", alpha=0.2)
    for index, value in enumerate(comparison["mean_wrmsse"]):
        axes[0].text(float(value) + 0.025, index, f"{value:.3f}", va="center")

    axes[1].barh(positions, comparison["wape"], color=colors)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_xlabel("Store-day WAPE, lower is better")
    axes[1].set_title("Operational store planning", loc="left", weight="bold")
    axes[1].grid(axis="x", alpha=0.2)
    for index, value in enumerate(comparison["wape"]):
        axes[1].text(float(value) + 0.008, index, f"{value:.1%}", va="center")
    figure.suptitle("Model comparison uses the same out-of-sample folds", weight="bold")
    figure.tight_layout()
    _save(figure, output_dir / "dashboard-model-comparison.png")
    return comparison.to_dict(orient="records")


def _inventory_example(
    inventory_kpis: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    policies = (
        inventory_kpis.groupby("model_name", as_index=False)
        .agg(fill_rate=("fill_rate", "mean"), total_cost=("total_cost", "sum"))
        .sort_values("total_cost")
    )
    recommendation = recommendations.sort_values(
        "suggested_order_quantity", ascending=False
    ).iloc[0]
    policy_labels = [MODEL_LABELS.get(model, model) for model in policies["model_name"]]
    colors = [MODEL_COLORS.get(model, GRAY) for model in policies["model_name"]]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(policy_labels, policies["total_cost"], color=colors, width=0.58)
    axes[0].set_title("Policy cost and service", loc="left", weight="bold")
    axes[0].set_ylabel("Simulated total cost")
    axes[0].grid(axis="y", alpha=0.2)
    for index, row in policies.reset_index(drop=True).iterrows():
        axes[0].text(
            index,
            float(row["total_cost"]) + 35,
            f"{row['total_cost']:,.0f}\nfill {row['fill_rate']:.1%}",
            ha="center",
        )

    quantities = [
        float(recommendation["on_hand"]),
        float(recommendation["reorder_point"]),
        float(recommendation["order_up_to"]),
    ]
    axes[1].barh(
        ["On hand", "Reorder point R", "Order-up-to S"],
        quantities,
        color=[BLUE, AMBER, GREEN],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Units")
    axes[1].set_title(
        f"{recommendation['series_id']}\n"
        f"Suggested order: {recommendation['suggested_order_quantity']:.0f} units",
        loc="left",
        weight="bold",
    )
    axes[1].grid(axis="x", alpha=0.2)
    for index, value in enumerate(quantities):
        axes[1].text(value + 8, index, f"{value:.0f}", va="center")
    figure.suptitle("Inventory simulator turns forecasts into an (R,S) decision", weight="bold")
    figure.tight_layout()
    _save(figure, output_dir / "dashboard-inventory-example.png")
    return {
        "series_id": str(recommendation["series_id"]),
        "on_hand": float(recommendation["on_hand"]),
        "reorder_point": float(recommendation["reorder_point"]),
        "order_up_to": float(recommendation["order_up_to"]),
        "suggested_order_quantity": float(recommendation["suggested_order_quantity"]),
        "policies": policies.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.profile)
    spark = get_spark(config, "export-dashboard-readme-assets")
    try:
        forecasts = _read_gold(spark, config, "forecasts_bottom")
        model_metrics = _read_gold(spark, config, "model_metrics")
        backtests = _read_gold(spark, config, "backtest_forecasts")
        inventory_kpis = _read_gold(spark, config, "inventory_kpis")
        recommendations = _read_gold(spark, config, "inventory_recommendations")
    finally:
        spark.stop()

    manifest = {
        "forecast": _forecast_example(forecasts, args.output_dir),
        "models": _model_comparison(model_metrics, backtests, args.output_dir),
        "inventory": _inventory_example(inventory_kpis, recommendations, args.output_dir),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
