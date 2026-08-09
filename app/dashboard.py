"""Streamlit dashboard for forecast, model, explanation, and inventory decisions."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from deltalake import DeltaTable

from retail_forecasting.config import load_config
from retail_forecasting.forecasting.metrics import (
    summarize_bottom_rmsse_artifacts,
    summarize_backtest_points,
    summarize_wape_by_granularity,
)
from retail_forecasting.inventory import InventoryPolicy, simulate_series

st.set_page_config(page_title="Retail Demand Control", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; max-width: 1500px;}
    [data-testid="stMetric"] {border: 1px solid #d9ddd8; border-radius: 6px; padding: 12px;}
    [data-testid="stSidebar"] {border-right: 1px solid #d9ddd8;}
    h1 {font-size: 1.8rem !important; letter-spacing: 0 !important;}
    h2 {font-size: 1.25rem !important; letter-spacing: 0 !important;}
    h3 {font-size: 1rem !important; letter-spacing: 0 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

CONFIG = load_config("dev")
GOLD = CONFIG.paths.lakehouse / "gold"


@st.cache_data(show_spinner=False)
def read_delta(name: str) -> pd.DataFrame:
    path = GOLD / name
    if not (path / "_delta_log").exists():
        return pd.DataFrame()
    return DeltaTable(str(path)).to_pandas()


@st.cache_data(show_spinner=False)
def read_bottom_rmsse() -> pd.DataFrame:
    return summarize_bottom_rmsse_artifacts(CONFIG.paths.artifacts / "backtests")


@st.cache_data(show_spinner=False, ttl=60)
def champion_mlflow_location() -> tuple[str, str] | None:
    base_url = CONFIG.mlflow.tracking_uri.rstrip("/")
    alias_query = urlencode(
        {"name": CONFIG.mlflow.registered_model, "alias": "champion"}
    )
    try:
        with urlopen(
            f"{base_url}/api/2.0/mlflow/registered-models/alias?{alias_query}",
            timeout=3,
        ) as response:
            model_version = json.load(response)["model_version"]
        run_id = model_version["run_id"]
        run_query = urlencode({"run_id": run_id})
        with urlopen(
            f"{base_url}/api/2.0/mlflow/runs/get?{run_query}", timeout=3
        ) as response:
            experiment_id = json.load(response)["run"]["info"]["experiment_id"]
    except (KeyError, OSError, ValueError):
        return None
    return experiment_id, run_id


forecasts = read_delta("forecasts_bottom")
metrics = read_delta("model_metrics")
backtests = read_delta("backtest_forecasts")
inventory_kpis = read_delta("inventory_kpis")
recommendations = read_delta("inventory_recommendations")

if not metrics.empty and not backtests.empty:
    point_summary = summarize_backtest_points(backtests)
    point_columns = [column for column in point_summary.columns if column != "model_name"]
    metrics = metrics.drop(columns=point_columns, errors="ignore").merge(
        point_summary, on="model_name", how="left"
    )
    bottom_rmsse = read_bottom_rmsse()
    if not bottom_rmsse.empty:
        metrics = metrics.drop(columns=["bottom_rmsse"], errors="ignore").merge(
            bottom_rmsse, on="model_name", how="left"
        )
    granularity_metrics = summarize_wape_by_granularity(backtests)
else:
    granularity_metrics = pd.DataFrame()

st.title("Retail Demand Control")
if forecasts.empty:
    st.error("Forecast data is not available.")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

stores = sorted(forecasts["store_id"].dropna().unique())
selected_store = st.sidebar.selectbox("Store", stores)
store_forecasts = forecasts.loc[forecasts["store_id"] == selected_store]
categories = sorted(store_forecasts["cat_id"].dropna().unique())
selected_category = st.sidebar.selectbox("Category", categories)
category_forecasts = store_forecasts.loc[store_forecasts["cat_id"] == selected_category]
series = sorted(category_forecasts["series_id"].unique())
selected_series = st.sidebar.selectbox("SKU", series)
series_forecast = category_forecasts.loc[
    category_forecasts["series_id"] == selected_series
].sort_values("target_date")

model_name = str(forecasts["model_name"].iloc[0])
run_id = str(forecasts["run_id"].iloc[0])
champion_metrics = metrics.loc[metrics["model_name"] == model_name]


def champion_metric(column: str) -> float:
    if champion_metrics.empty or column not in champion_metrics:
        return 0.0
    return float(champion_metrics[column].iloc[0])


def champion_granularity_metric(granularity: str, column: str) -> float:
    if granularity_metrics.empty:
        return 0.0
    selected = granularity_metrics.loc[
        (granularity_metrics["model_name"] == model_name)
        & (granularity_metrics["granularity"] == granularity)
    ]
    if selected.empty:
        return 0.0
    return float(selected[column].iloc[0])


recommended = recommendations.loc[recommendations["series_id"] == selected_series]

overview, explorer, comparison, explanation, inventory = st.tabs(
    ["Overview", "Forecast explorer", "Model comparison", "Explainability", "Inventory"]
)

with overview:
    columns = st.columns(4)
    columns[0].metric("Champion", model_name)
    columns[1].metric("Mean WRMSSE", f"{champion_metric('mean_wrmsse'):.4f}")
    columns[2].metric(
        "Store-day WAPE", f"{champion_granularity_metric('store_day', 'wape'):.1%}"
    )
    columns[3].metric("90% coverage", f"{champion_metric('coverage'):.1%}")
    columns = st.columns(4)
    columns[0].metric("SKU-store-day WAPE", f"{champion_metric('wape'):.1%}")
    columns[1].metric("MAE", f"{champion_metric('mae'):.3f}")
    columns[2].metric("RMSE", f"{champion_metric('rmse'):.3f}")
    columns[3].metric("Bias", f"{champion_metric('bias'):.1%}")
    if champion_metric("wape") > 0.5:
        st.warning(
            f"SKU-store-day WAPE is {champion_metric('wape'):.1%}, driven by intermittent "
            "demand. Use the granularity control in Model comparison to distinguish item-level "
            "inventory uncertainty from aggregate planning accuracy."
        )
    st.subheader("Demand outlook")
    summary = (
        store_forecasts.groupby(["target_date", "cat_id"], as_index=False)["q50"].sum()
    )
    figure = px.area(
        summary,
        x="target_date",
        y="q50",
        color="cat_id",
        labels={"q50": "Median demand", "cat_id": "Category"},
        color_discrete_sequence=["#20744a", "#d4932f", "#496d8c"],
    )
    figure.update_layout(margin=dict(l=0, r=0, t=16, b=0), legend_orientation="h")
    st.plotly_chart(figure, width="stretch")

with explorer:
    st.subheader(selected_series)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series_forecast["target_date"],
            y=series_forecast["q95"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=series_forecast["target_date"],
            y=series_forecast["q05"],
            fill="tonexty",
            fillcolor="rgba(32,116,74,0.18)",
            line=dict(width=0),
            name="90% interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=series_forecast["target_date"],
            y=series_forecast["q50"],
            line=dict(color="#20744a", width=2),
            name="Median",
        )
    )
    figure.update_layout(margin=dict(l=0, r=0, t=16, b=0), yaxis_title="Units")
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        series_forecast[["target_date", "q05", "q50", "q95"]],
        width="stretch",
        hide_index=True,
    )

with comparison:
    st.subheader("Backtest performance")
    if metrics.empty:
        st.warning("Model metrics are not available.")
    else:
        granularity_labels = {
            "SKU-store by day": "sku_store_day",
            "SKU-store over 28 days": "sku_store_28d",
            "Store by day": "store_day",
            "Store-category by day": "store_category_day",
            "Total by day": "total_day",
            "SKU-store by week": "sku_store_week",
            "Store-category by week": "store_category_week",
        }
        selected_granularity_label = st.selectbox(
            "Evaluation granularity", granularity_labels, index=2
        )
        selected_granularity = granularity_labels[selected_granularity_label]
        metric_options = {
            "WRMSSE": "mean_wrmsse",
            "WAPE": "wape",
            "MAE": "mae",
            "RMSE": "rmse",
            "Bottom-level RMSSE": "bottom_rmsse",
            "Mean pinball loss": "mean_pinball_loss",
            "Bias": "bias",
            "90% coverage": "coverage",
            "Interval width": "mean_interval_width",
            "Fold degradation": "max_fold_degradation",
        }
        selected_metric = st.selectbox("Metric", metric_options)
        metric_column = metric_options[selected_metric]
        chart_data = metrics.copy()
        if metric_column in {"wape", "mae", "rmse", "bias"}:
            selected_point_metrics = granularity_metrics.loc[
                granularity_metrics["granularity"] == selected_granularity,
                ["model_name", "wape", "mae", "rmse", "bias"],
            ]
            chart_data = chart_data.drop(
                columns=["wape", "mae", "rmse", "bias"], errors="ignore"
            ).merge(selected_point_metrics, on="model_name", how="left")
        chart_data["Model"] = chart_data["model_name"].replace(
            {
                "lightgbm": "LightGBM",
                "xgboost": "XGBoost",
                "nhits": "N-HiTS",
                "moving_average": "Moving Average",
                "seasonal_naive": "Seasonal Naive",
            }
        )
        figure = px.bar(
            chart_data.sort_values(metric_column),
            x="Model",
            y=metric_column,
            color="Model",
            color_discrete_sequence=["#20744a", "#d4932f", "#496d8c", "#a94b45", "#6b665f"],
            labels={metric_column: selected_metric},
        )
        if metric_column in {"wape", "bias", "coverage", "max_fold_degradation"}:
            figure.update_yaxes(tickformat=".1%")
        if metric_column == "bias":
            figure.add_hline(y=0, line_color="#6b665f", line_dash="dot")
        if metric_column == "coverage":
            figure.add_hline(y=0.9, line_color="#6b665f", line_dash="dot")
        figure.update_layout(showlegend=False, margin=dict(l=0, r=0, t=16, b=0))
        st.plotly_chart(figure, width="stretch")
        comparison_columns = [
            "Model",
            "mean_wrmsse",
            "bottom_rmsse",
            "wape",
            "mae",
            "rmse",
            "mean_pinball_loss",
            "bias",
            "coverage",
            "mean_interval_width",
            "max_fold_degradation",
        ]
        comparison_table = chart_data[
            [column for column in comparison_columns if column in chart_data]
        ].sort_values("mean_wrmsse")
        st.dataframe(
            comparison_table,
            width="stretch",
            hide_index=True,
            column_config={
                "mean_wrmsse": st.column_config.NumberColumn("WRMSSE", format="%.4f"),
                "bottom_rmsse": st.column_config.NumberColumn(
                    "Bottom RMSSE", format="%.3f"
                ),
                "wape": st.column_config.NumberColumn("WAPE", format="percent"),
                "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
                "mean_pinball_loss": st.column_config.NumberColumn(
                    "Pinball loss", format="%.3f"
                ),
                "bias": st.column_config.NumberColumn("Bias", format="percent"),
                "coverage": st.column_config.NumberColumn("Coverage", format="percent"),
                "mean_interval_width": st.column_config.NumberColumn(
                    "Interval width", format="%.3f"
                ),
                "max_fold_degradation": st.column_config.NumberColumn(
                    "Fold degradation", format="percent"
                ),
            },
        )
    mlflow_location = champion_mlflow_location()
    if mlflow_location:
        experiment_id, mlflow_run_id = mlflow_location
        st.link_button(
            "Open champion in MLflow",
            f"http://localhost:5000/#/experiments/{experiment_id}/runs/{mlflow_run_id}",
        )
    else:
        st.caption("The MLflow champion link is temporarily unavailable.")

with explanation:
    st.subheader("Feature attribution")
    explanation_dir = CONFIG.paths.artifacts / "explainability" / run_id
    images = sorted(explanation_dir.glob("*-shap-*.png"))
    if images:
        left, right = st.columns(2)
        for index, image in enumerate(images[:2]):
            (left if index == 0 else right).image(str(image), width="stretch")
    else:
        st.warning("Explanation artifacts are not available.")

with inventory:
    st.subheader("Periodic review scenario")
    if not inventory_kpis.empty:
        policy_comparison = (
            inventory_kpis.groupby("model_name", as_index=False)
            .agg(
                fill_rate=("fill_rate", "mean"),
                stockout_rate=("stockout_rate", "mean"),
                average_inventory=("average_inventory", "mean"),
                lost_sales_units=("lost_sales_units", "sum"),
                total_cost=("total_cost", "sum"),
            )
            .sort_values("total_cost")
        )
        st.dataframe(
            policy_comparison,
            width="stretch",
            hide_index=True,
            column_config={
                "model_name": "Policy model",
                "fill_rate": st.column_config.NumberColumn("Fill rate", format="percent"),
                "stockout_rate": st.column_config.NumberColumn(
                    "Stockout rate", format="percent"
                ),
                "average_inventory": st.column_config.NumberColumn(
                    "Average inventory", format="%.2f"
                ),
                "lost_sales_units": st.column_config.NumberColumn(
                    "Lost sales", format="%.1f"
                ),
                "total_cost": st.column_config.NumberColumn("Total cost", format="$%.2f"),
            },
        )
    controls = st.columns(4)
    lead_time = controls[0].number_input("Lead time", 1, 28, CONFIG.inventory.lead_time_days)
    review_period = controls[1].number_input(
        "Review period", 1, 28, CONFIG.inventory.review_period_days
    )
    service_level = controls[2].slider("Service level", 0.80, 0.99, 0.95, 0.01)
    on_hand = controls[3].number_input(
        "On hand",
        min_value=0.0,
        value=float(recommended["on_hand"].iloc[0]) if not recommended.empty else 0.0,
    )
    selected_backtest = backtests.loc[
        (backtests["series_id"] == selected_series)
        & (backtests["model_name"] == model_name)
        & (backtests["fold_origin"] == backtests["fold_origin"].max())
    ].copy()
    if selected_backtest.empty:
        st.warning("Backtest inventory inputs are not available.")
    else:
        residuals = (
            backtests.loc[backtests["model_name"] == model_name]
            .assign(residual=lambda frame: frame["target"] - frame["yhat"])
            .pivot_table(index=["fold_origin", "series_id"], columns="horizon", values="residual")
            .fillna(0)
            .to_numpy()
        )
        policy = InventoryPolicy(
            lead_time_days=int(lead_time),
            review_period_days=int(review_period),
            service_level=float(service_level),
            bootstrap_paths=CONFIG.inventory.bootstrap_paths,
            fixed_order_cost=CONFIG.inventory.fixed_order_cost,
            annual_holding_rate=CONFIG.inventory.annual_holding_rate,
            stockout_price_multiplier=CONFIG.inventory.stockout_price_multiplier,
            seed=CONFIG.seed,
        )
        daily, summary = simulate_series(
            selected_backtest, selected_backtest, residuals, policy, float(on_hand)
        )
        columns = st.columns(4)
        columns[0].metric("Fill rate", f"{summary['fill_rate']:.1%}")
        columns[1].metric("Stockout rate", f"{summary['stockout_rate']:.1%}")
        columns[2].metric("Average inventory", f"{summary['average_inventory']:,.1f}")
        columns[3].metric("Total cost", f"${summary['total_cost']:,.2f}")
        figure = px.line(
            daily,
            x="target_date",
            y=["ending_on_hand", "demand", "forecast"],
            labels={"value": "Units", "variable": "Measure"},
            color_discrete_sequence=["#20744a", "#a94b45", "#496d8c"],
        )
        figure.update_layout(margin=dict(l=0, r=0, t=16, b=0), legend_orientation="h")
        st.plotly_chart(figure, width="stretch")
