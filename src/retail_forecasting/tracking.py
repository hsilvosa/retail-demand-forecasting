"""MLflow experiment, registry, and promotion policy."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd

from retail_forecasting.config import ProjectConfig


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


@contextmanager
def tracking_run(config: ProjectConfig, run_name: str) -> Iterator[Any]:
    import mlflow

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(
        f"{config.mlflow.experiment_prefix} - {config.profile.upper()}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "profile": config.profile,
                "git_sha": git_sha(),
                "project": config.project_name,
            }
        )
        mlflow.log_dict(config.model_dump(mode="json"), "config/resolved.json")
        yield run


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    import mlflow

    clean = {f"{prefix}{key}": float(value) for key, value in metrics.items()}
    mlflow.log_metrics(clean)


def log_forecaster(
    config: ProjectConfig,
    forecaster: Any,
    input_example: pd.DataFrame,
    model_name: str,
) -> str:
    """Package any adapter behind a consistent prepared-feature pyfunc contract."""
    import mlflow

    class ForecastPythonModel(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def predict(
            self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
        ) -> pd.DataFrame:
            predicted = self.wrapped.predict(model_input)
            if isinstance(predicted, pd.DataFrame):
                return predicted
            if hasattr(self.wrapped, "calibrator"):
                q05, q50, q95 = self.wrapped.calibrator.predict(
                    predicted, model_input["horizon"].to_numpy(dtype=int)
                )
                return pd.DataFrame(
                    {"yhat": predicted, "q05": q05, "q50": q50, "q95": q95}
                )
            return pd.DataFrame({"yhat": predicted})

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ForecastPythonModel(forecaster),
        input_example=input_example.head(5),
        registered_model_name=config.mlflow.registered_model,
        metadata={"candidate_model": model_name, "profile": config.profile},
    )
    versions = mlflow.MlflowClient().search_model_versions(
        f"name='{config.mlflow.registered_model}' AND run_id='{mlflow.active_run().info.run_id}'"
    )
    if not versions:
        raise RuntimeError("MLflow did not create a registered model version")
    version = max(versions, key=lambda item: int(item.version))
    client = mlflow.MlflowClient()
    client.set_model_version_tag(
        config.mlflow.registered_model, version.version, "model", model_name
    )
    client.set_registered_model_alias(config.mlflow.registered_model, "candidate", version.version)
    return str(version.version)


def _version_for_run(client: Any, model_name: str, run_id: str | None) -> Any:
    versions = client.search_model_versions(f"name='{model_name}'")
    if run_id:
        versions = [version for version in versions if version.run_id == run_id]
    if not versions:
        raise ValueError("No candidate model version found")
    return max(versions, key=lambda item: int(item.version))


def _eligible(config: ProjectConfig, metrics: dict[str, float]) -> tuple[bool, list[str]]:
    reasons = []
    if abs(metrics.get("bias", float("inf"))) > config.mlflow.max_abs_bias:
        reasons.append("bias_guardrail")
    coverage = metrics.get("coverage", -1.0)
    if not config.mlflow.coverage_min <= coverage <= config.mlflow.coverage_max:
        reasons.append("coverage_guardrail")
    if metrics.get("max_fold_degradation", float("inf")) > config.mlflow.max_fold_degradation:
        reasons.append("fold_degradation_guardrail")
    return not reasons, reasons


def promote_candidate(config: ProjectConfig, run_id: str | None = None) -> dict[str, Any]:
    import mlflow
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = mlflow.MlflowClient()
    candidate = _version_for_run(client, config.mlflow.registered_model, run_id)
    candidate_run = client.get_run(candidate.run_id)
    metrics = dict(candidate_run.data.metrics)
    eligible, reasons = _eligible(config, metrics)
    try:
        champion = client.get_model_version_by_alias(config.mlflow.registered_model, "champion")
    except MlflowException:
        champion = None
    if champion is not None:
        champion_metrics = dict(client.get_run(champion.run_id).data.metrics)
        required = champion_metrics["mean_wrmsse"] * (1 - config.mlflow.improvement_threshold)
        if metrics.get("mean_wrmsse", float("inf")) > required:
            eligible = False
            reasons.append("insufficient_wrmsse_improvement")
    if eligible:
        client.set_registered_model_alias(
            config.mlflow.registered_model, "champion", candidate.version
        )
    result = {
        "promoted": eligible,
        "version": str(candidate.version),
        "run_id": candidate.run_id,
        "reasons": reasons,
    }
    output = config.paths.artifacts / "model_registry" / f"promotion-{candidate.version}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
