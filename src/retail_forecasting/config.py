"""Validated configuration loading for every pipeline surface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path
    lakehouse: Path
    artifacts: Path
    state: Path


class SparkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    master: str
    driver_memory: str
    executor_memory: str
    shuffle_partitions: int = Field(gt=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon: int = Field(gt=0)
    history_days: int = Field(gt=0)
    training_origin_step: int = Field(gt=0)
    forecast_origin_day: int = Field(gt=0)
    backtest_origins: list[int]
    required_files: list[str]
    sample_series: int | None = Field(default=None, gt=0)
    sample_per_state: int | None = Field(default=None, gt=0)


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lags: list[int]
    target_lags: list[int]
    rolling_windows: list[int]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    names: list[str]
    quantiles: list[float]
    tune_fraction: float = Field(gt=0, le=1)
    nhits_input_size: int = Field(gt=0)
    require_gpu: bool
    optuna_trials: int = Field(gt=0)
    nhits_max_steps: int = Field(gt=0)


class MlflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tracking_uri: str
    experiment_prefix: str
    registered_model: str
    improvement_threshold: float = Field(ge=0)
    max_abs_bias: float = Field(ge=0)
    coverage_min: float = Field(ge=0, le=1)
    coverage_max: float = Field(ge=0, le=1)
    max_fold_degradation: float = Field(ge=0)


class InventoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_time_days: int = Field(gt=0)
    review_period_days: int = Field(gt=0)
    service_level: float = Field(gt=0, lt=1)
    bootstrap_paths: int = Field(gt=0)
    fixed_order_cost: float = Field(ge=0)
    annual_holding_rate: float = Field(ge=0)
    stockout_price_multiplier: float = Field(ge=0)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str
    profile: str
    seed: int
    paths: PathsConfig
    spark: SparkConfig
    data: DataConfig
    features: FeatureConfig
    models: ModelConfig
    mlflow: MlflowConfig
    inventory: InventoryConfig

    @model_validator(mode="after")
    def validate_intervals(self) -> ProjectConfig:
        if self.mlflow.coverage_min > self.mlflow.coverage_max:
            raise ValueError("coverage_min cannot exceed coverage_max")
        if sorted(self.models.quantiles) != self.models.quantiles:
            raise ValueError("model quantiles must be sorted")
        if min(self.features.target_lags) < self.data.horizon:
            raise ValueError("target_lags must be at least as large as the forecast horizon")
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(profile: str = "dev", config_dir: Path | None = None) -> ProjectConfig:
    """Load base settings and overlay a named execution profile."""
    if config_dir is None:
        configured = os.getenv("RETAIL_CONFIG_DIR")
        candidates = [Path(configured)] if configured else []
        candidates.extend([Path.cwd() / "config", DEFAULT_CONFIG_DIR])
        config_dir = next(
            (candidate for candidate in candidates if (candidate / "base.yaml").is_file()),
            candidates[0],
        )
    base_path = config_dir / "base.yaml"
    profile_path = config_dir / f"{profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Unknown profile {profile!r}: {profile_path}")
    with base_path.open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with profile_path.open(encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    resolved = _deep_merge(base, override)
    root = config_dir.parent.resolve()
    resolved["paths"] = {
        key: str((root / value).resolve()) if not Path(value).is_absolute() else value
        for key, value in resolved["paths"].items()
    }
    return ProjectConfig.model_validate(resolved)
