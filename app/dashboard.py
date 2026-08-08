"""Streamlit dashboard for forecast, model, explanation, and inventory decisions."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from deltalake import DeltaTable

from retail_forecasting.config import load_config
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


forecasts = read_delta("forecasts_bottom")
metrics = read_delta("model_metrics")
backtests = read_delta("backtest_forecasts")
inventory_kpis = read_delta("inventory_kpis")
recommendations = read_delta("inventory_recommendations")

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
coverage = (
    float(metrics.loc[metrics["model_name"] == model_name, "coverage"].iloc[0])
    if not metrics.empty
    else 0
)
wrmsse = (
    float(metrics.loc[metrics["model_name"] == model_name, "mean_wrmsse"].iloc[0])
    if not metrics.empty
    else 0
)
recommended = recommendations.loc[recommendations["series_id"] == selected_series]

overview, explorer, comparison, explanation, inventory = st.tabs(
    ["Overview", "Forecast explorer", "Model comparison", "Explainability", "Inventory"]
)

with overview:
    columns = st.columns(4)
    columns[0].metric("Champion", model_name)
    columns[1].metric("Mean WRMSSE", f"{wrmsse:.4f}")
    columns[2].metric("90% coverage", f"{coverage:.1%}")
    columns[3].metric("Forecast demand", f"{series_forecast['q50'].sum():,.1f}")
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
        figure = px.bar(
            metrics.sort_values("mean_wrmsse"),
            x="model_name",
            y="mean_wrmsse",
            color="model_name",
            color_discrete_sequence=["#20744a", "#d4932f", "#496d8c", "#a94b45", "#6b665f"],
            labels={"model_name": "Model", "mean_wrmsse": "Mean WRMSSE"},
        )
        figure.update_layout(showlegend=False, margin=dict(l=0, r=0, t=16, b=0))
        st.plotly_chart(figure, width="stretch")
        st.dataframe(metrics, width="stretch", hide_index=True)
    st.link_button("Open MLflow run", f"http://localhost:5000/#/experiments/0/runs/{run_id}")

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
