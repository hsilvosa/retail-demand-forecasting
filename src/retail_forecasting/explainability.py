"""Model-specific explanations persisted as MLflow and dashboard artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TreeShapAnalysis:
    """Quantitative views derived from persisted tree SHAP attributions."""

    global_importance: pd.DataFrame
    group_importance: pd.DataFrame
    horizon_importance: pd.DataFrame
    segment_importance: pd.DataFrame
    local_explanations: pd.DataFrame
    summary: dict[str, float | int | str]


def _feature_group(feature: str) -> str:
    if feature.endswith("_id") or feature in {"item_id", "dept_id", "cat_id"}:
        return "Identifiers"
    if "price" in feature or "sell_price" in feature:
        return "Price"
    if feature in {"origin_day", "horizon"} or feature.startswith(
        ("target_wday", "target_month", "target_event", "target_snap")
    ):
        return "Calendar and horizon"
    if feature.startswith(
        ("target_lag_", "lag_", "rolling_", "nonzero_", "days_since_", "short_long_")
    ):
        return "Demand history"
    return "Other"


def analyze_tree_shap(
    attributions: pd.DataFrame, top_n: int = 20, local_n: int = 10
) -> TreeShapAnalysis:
    """Summarize global, grouped, horizon, and local SHAP behavior.

    The persisted contract requires one ``shap_<feature>`` column plus series and
    horizon identifiers. Positive SHAP values raise a row's forecast relative to
    the explainer baseline; negative values lower it. They are associations, not
    causal effects.
    """
    if top_n < 1 or local_n < 1:
        raise ValueError("top_n and local_n must be positive")
    required = {"series_id", "horizon"}
    missing = required.difference(attributions.columns)
    if missing:
        raise ValueError(f"SHAP attributions are missing columns: {sorted(missing)}")
    shap_columns = [
        column for column in attributions.columns if column.startswith("shap_")
    ]
    if not shap_columns:
        raise ValueError("SHAP attributions contain no shap_<feature> columns")
    if attributions.empty:
        raise ValueError("SHAP attributions are empty")

    frame = attributions.reset_index(drop=True)
    values = frame[shap_columns].apply(pd.to_numeric, errors="raise")
    total_importance = max(float(values.abs().mean().sum()), 1e-12)
    global_importance = pd.DataFrame(
        {
            "feature": [column.removeprefix("shap_") for column in shap_columns],
            "mean_abs_shap": values.abs().mean().to_numpy(),
            "mean_shap": values.mean().to_numpy(),
            "positive_share": (values > 0).mean().to_numpy(),
            "negative_share": (values < 0).mean().to_numpy(),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    global_importance["importance_share"] = (
        global_importance["mean_abs_shap"] / total_importance
    )
    global_importance["feature_group"] = global_importance["feature"].map(
        _feature_group
    )
    global_importance.insert(0, "rank", np.arange(1, len(global_importance) + 1))

    group_importance = (
        global_importance.groupby("feature_group", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    group_importance["importance_share"] = (
        group_importance["mean_abs_shap"] / total_importance
    )

    horizon_wide = values.abs().assign(horizon=frame["horizon"]).groupby("horizon").mean()
    horizon_importance = horizon_wide.rename(
        columns=lambda column: column.removeprefix("shap_")
    ).melt(ignore_index=False, var_name="feature", value_name="mean_abs_shap")
    horizon_importance = horizon_importance.reset_index()
    horizon_importance["rank"] = horizon_importance.groupby("horizon")[
        "mean_abs_shap"
    ].rank(method="first", ascending=False)
    horizon_importance = horizon_importance.sort_values(
        ["horizon", "rank"], ignore_index=True
    )

    row_magnitude = values.abs().sum(axis=1)
    segments = frame["series_id"].astype(str).str.extract(
        r"^(?P<category>[^_]+)_.+_(?P<store>[A-Z]{2}_\d+)$"
    )
    segments["state"] = segments["store"].str.split("_").str[0]
    segment_rows: list[dict[str, float | int | str]] = []
    for segment_type in ("state", "store", "category"):
        for segment in sorted(segments[segment_type].dropna().unique()):
            mask = segments[segment_type] == segment
            segment_values = values.loc[mask]
            segment_feature_importance = segment_values.abs().mean()
            top_column = str(segment_feature_importance.idxmax())
            segment_rows.append(
                {
                    "segment_type": segment_type,
                    "segment": str(segment),
                    "rows": int(mask.sum()),
                    "series_count": int(frame.loc[mask, "series_id"].nunique()),
                    "mean_total_abs_shap": float(row_magnitude.loc[mask].mean()),
                    "top_feature": top_column.removeprefix("shap_"),
                    "top_feature_mean_abs_shap": float(
                        segment_feature_importance[top_column]
                    ),
                }
            )
    segment_importance = pd.DataFrame(segment_rows)
    if not segment_importance.empty:
        segment_importance = segment_importance.sort_values(
            ["segment_type", "mean_total_abs_shap"],
            ascending=[True, False],
            ignore_index=True,
        )

    local_rows: list[dict[str, float | int | str]] = []
    for index in row_magnitude.nlargest(min(local_n, len(frame))).index:
        row = values.loc[index]
        drivers = row.abs().nlargest(min(3, len(row))).index
        top_driver = drivers[0]
        top_value = float(row[top_driver])
        local_rows.append(
            {
                "series_id": str(frame.loc[index, "series_id"]),
                "horizon": int(frame.loc[index, "horizon"]),
                "total_abs_shap": float(row_magnitude.loc[index]),
                "top_driver": top_driver.removeprefix("shap_"),
                "top_driver_shap": top_value,
                "direction": "raises forecast" if top_value > 0 else "lowers forecast",
                "top_three_drivers": "; ".join(
                    f"{column.removeprefix('shap_')} ({float(row[column]):+.3f})"
                    for column in drivers
                ),
            }
        )
    local_explanations = pd.DataFrame(local_rows)

    group_shares = group_importance.set_index("feature_group")["importance_share"]
    top_feature = str(global_importance.loc[0, "feature"])
    top_horizon_count = int(
        (
            horizon_importance.loc[horizon_importance["rank"] == 1, "feature"]
            == top_feature
        ).sum()
    )
    summary: dict[str, float | int | str] = {
        "sample_rows": len(frame),
        "series_count": int(frame["series_id"].nunique()),
        "horizon_count": int(frame["horizon"].nunique()),
        "feature_count": len(shap_columns),
        "top_feature": top_feature,
        "top_feature_mean_abs_shap": float(global_importance.loc[0, "mean_abs_shap"]),
        "top_feature_share": float(global_importance.loc[0, "importance_share"]),
        "top_five_share": float(global_importance.head(5)["importance_share"].sum()),
        "top_feature_horizon_count": top_horizon_count,
        "demand_history_share": float(group_shares.get("Demand history", 0.0)),
        "calendar_horizon_share": float(group_shares.get("Calendar and horizon", 0.0)),
        "identifier_share": float(group_shares.get("Identifiers", 0.0)),
        "price_share": float(group_shares.get("Price", 0.0)),
    }
    return TreeShapAnalysis(
        global_importance=global_importance.head(top_n).reset_index(drop=True),
        group_importance=group_importance,
        horizon_importance=horizon_importance,
        segment_importance=segment_importance,
        local_explanations=local_explanations,
        summary=summary,
    )


def explain_tree(forecaster: Any, frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    sample = frame.sample(min(len(frame), 2000), random_state=forecaster.seed)
    features = forecaster._transform(sample)
    explainer = shap.TreeExplainer(forecaster.model)
    values = explainer(features)
    beeswarm_path = output_dir / f"{forecaster.name}-shap-beeswarm.png"
    bar_path = output_dir / f"{forecaster.name}-shap-bar.png"
    shap.plots.beeswarm(values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()
    shap.plots.bar(values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()
    attribution = pd.DataFrame(
        values.values,
        columns=[f"shap_{column}" for column in features.columns],
    )
    attribution.insert(0, "series_id", sample["series_id"].to_numpy())
    attribution.insert(1, "horizon", sample["horizon"].to_numpy())
    for column in features.columns:
        attribution[f"feature_{column}"] = features[column].to_numpy()
    base_values = np.asarray(values.base_values)
    attribution["base_value"] = (
        float(base_values) if base_values.ndim == 0 else base_values.reshape(-1)
    )
    attribution["model_output"] = np.asarray(
        forecaster.model.predict(features)
    ).reshape(-1)
    values_path = output_dir / f"{forecaster.name}-shap-values.parquet"
    attribution.to_parquet(values_path, index=False)
    return {"beeswarm": str(beeswarm_path), "bar": str(bar_path), "values": str(values_path)}


def explain_nhits(
    forecaster: Any, future: pd.DataFrame, output_dir: Path
) -> dict[str, str]:
    """Request Integrated Gradients from NeuralForecast and persist raw attributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = forecaster._panel(future, include_target=False)
    explained = forecaster.model.predict(
        futr_df=panel,
        explainer_config={"explainer": "IntegratedGradients"},
    )
    path = output_dir / "nhits-integrated-gradients.parquet"
    explained.to_parquet(path, index=False)
    return {"integrated_gradients": str(path)}
