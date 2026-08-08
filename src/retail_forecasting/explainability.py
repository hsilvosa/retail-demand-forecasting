"""Model-specific explanations persisted as MLflow and dashboard artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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
