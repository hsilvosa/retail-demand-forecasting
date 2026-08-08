"""Temporal backtesting, champion training, registration, and batch inference."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from retail_forecasting.config import ProjectConfig
from retail_forecasting.data.spark import get_spark, table_path, write_delta
from retail_forecasting.explainability import explain_nhits, explain_tree
from retail_forecasting.forecasting.baselines import MovingAverage, SeasonalNaive
from retail_forecasting.forecasting.calibration import ResidualCalibrator
from retail_forecasting.forecasting.deep import NHITSForecaster
from retail_forecasting.forecasting.metrics import hierarchical_wrmsse_spark, point_metrics
from retail_forecasting.forecasting.models import (
    DirectTreeForecaster,
    ModelKind,
    tune_tree_model,
)
from retail_forecasting.tracking import log_forecaster, log_metrics, promote_candidate, tracking_run

TREE_MODELS = {"lightgbm", "xgboost"}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _split_fold(features: DataFrame, origin: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = features.filter(
        (F.col("origin_day") < origin) & (F.col("origin_day") + F.col("horizon") <= origin)
    ).toPandas()
    evaluation = features.filter(F.col("origin_day") == origin).toPandas()
    if train.empty or evaluation.empty:
        raise ValueError(f"Fold {origin} has no training or evaluation rows")
    return train, evaluation


def _validation_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_origin = int(train["origin_day"].max())
    validation = train.loc[train["origin_day"] == validation_origin].copy()
    fit = train.loc[train["origin_day"] < validation_origin].copy()
    return (fit if not fit.empty else train, validation if not validation.empty else train)


def _baseline_predictions(
    name: str, train: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[Any, pd.DataFrame]:
    model = SeasonalNaive() if name == "seasonal_naive" else MovingAverage()
    train_point = model.predict(train)
    calibrator = ResidualCalibrator().fit(
        train["target"].to_numpy(), train_point, train["horizon"].to_numpy()
    )
    point = model.predict(evaluation)
    q05, q50, q95 = calibrator.predict(point, evaluation["horizon"].to_numpy())
    model.calibrator = calibrator
    return model, pd.DataFrame({"yhat": point, "q05": q05, "q50": q50, "q95": q95})


def _tree_predictions(
    name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: ProjectConfig,
    parameters: dict[str, Any],
) -> tuple[DirectTreeForecaster, pd.DataFrame]:
    fit, validation = _validation_split(train)
    model = DirectTreeForecaster(
        cast(ModelKind, name),
        seed=config.seed,
        use_gpu=_cuda_available(),
        params=parameters,
    )
    model.fit(fit, validation)
    return model, model.predict(evaluation)


def _nhits_predictions(
    daily: DataFrame, origin: int, config: ProjectConfig
) -> tuple[NHITSForecaster, pd.DataFrame, pd.DataFrame]:
    lower = max(1, origin - config.data.history_days + 1)
    history = daily.filter((F.col("day_num") >= lower) & (F.col("day_num") <= origin)).toPandas()
    future = daily.filter(
        (F.col("day_num") > origin) & (F.col("day_num") <= origin + config.data.horizon)
    ).toPandas()
    model = NHITSForecaster(
        horizon=config.data.horizon,
        input_size=config.models.nhits_input_size,
        max_steps=config.models.nhits_max_steps,
        seed=config.seed,
        use_gpu=_cuda_available(),
    ).fit(history)
    predicted = model.predict(future)
    future["target_date"] = pd.to_datetime(future["date"])
    merged = future.merge(predicted, on=["series_id", "target_date"], how="inner")
    return (
        model,
        merged[["yhat", "q05", "q50", "q95"]],
        merged.drop(columns=["yhat", "q05", "q50", "q95"]),
    )


def _tune_parameters(
    features: DataFrame, config: ProjectConfig
) -> dict[str, dict[str, Any]]:
    if not TREE_MODELS.intersection(config.models.names):
        return {}
    origin = min(config.data.backtest_origins)
    train, _ = _split_fold(features, origin)
    fit, validation = _validation_split(train)
    fit_sample = fit.sample(frac=config.models.tune_fraction, random_state=config.seed)
    validation_sample = validation.sample(
        frac=config.models.tune_fraction, random_state=config.seed
    )
    return {
        name: tune_tree_model(
            cast(ModelKind, name),
            fit_sample,
            validation_sample,
            config.models.optuna_trials,
            config.seed,
            _cuda_available(),
        )
        for name in config.models.names
        if name in TREE_MODELS
    }


def _attach_predictions(
    evaluation: pd.DataFrame, predicted: pd.DataFrame, model_name: str, origin: int
) -> pd.DataFrame:
    identity = evaluation.reset_index(drop=True).copy()
    values = predicted.reset_index(drop=True)
    for column in ("yhat", "q05", "q50", "q95"):
        identity[column] = values[column].to_numpy()
    identity["model_name"] = model_name
    identity["fold_origin"] = origin
    identity["day_num"] = identity["origin_day"] + identity["horizon"]
    return identity


def _evaluate_fold(
    daily: DataFrame,
    forecast: pd.DataFrame,
    origin: int,
) -> tuple[dict[str, float], DataFrame]:
    metrics = point_metrics(
        forecast["target"], forecast["yhat"], forecast["q05"], forecast["q95"]
    )
    spark = daily.sparkSession
    spark_forecast = spark.createDataFrame(
        forecast[
            [
                "series_id",
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
                "day_num",
                "target",
                "yhat",
            ]
        ]
    )
    wrmsse, details = hierarchical_wrmsse_spark(daily, spark_forecast, origin)
    metrics["wrmsse"] = wrmsse
    return metrics, details


def _prepare_nhits_future(features: pd.DataFrame) -> pd.DataFrame:
    return features.rename(
        columns={
            "target_date": "date",
            "target_wday": "wday",
            "target_month": "month",
            "target_sell_price": "sell_price",
            "target_snap_CA": "snap_CA",
            "target_snap_TX": "snap_TX",
            "target_snap_WI": "snap_WI",
        }
    )


def _fit_final(
    winner: str,
    features: DataFrame,
    future: pd.DataFrame,
    daily: DataFrame,
    config: ProjectConfig,
    parameters: dict[str, dict[str, Any]],
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    training = features.filter(
        F.col("origin_day") + F.col("horizon") <= config.data.forecast_origin_day
    ).toPandas()
    if winner in TREE_MODELS:
        fit, validation = _validation_split(training)
        tree_model = DirectTreeForecaster(
            cast(ModelKind, winner),
            seed=config.seed,
            use_gpu=_cuda_available(),
            params=parameters.get(winner, {}),
        ).fit(fit, validation)
        return tree_model, tree_model.predict(future), future
    if winner in {"seasonal_naive", "moving_average"}:
        baseline_model, _ = _baseline_predictions(
            winner, training, training.tail(min(len(training), 1000))
        )
        point = baseline_model.predict(future)
        q05, q50, q95 = baseline_model.calibrator.predict(
            point, future["horizon"].to_numpy()
        )
        predicted = pd.DataFrame({"yhat": point, "q05": q05, "q50": q50, "q95": q95})
        return baseline_model, predicted, future
    lower = max(1, config.data.forecast_origin_day - config.data.history_days + 1)
    history = daily.filter(
        (F.col("day_num") >= lower) & (F.col("day_num") <= config.data.forecast_origin_day)
    ).toPandas()
    nhits_model = NHITSForecaster(
        horizon=config.data.horizon,
        input_size=config.models.nhits_input_size,
        max_steps=config.models.nhits_max_steps,
        seed=config.seed,
        use_gpu=_cuda_available(),
    ).fit(history)
    nhits_future = _prepare_nhits_future(future)
    output = nhits_model.predict(nhits_future)
    keyed = future.copy()
    keyed["target_date"] = pd.to_datetime(keyed["target_date"])
    merged = keyed.merge(output, on=["series_id", "target_date"], how="left")
    return nhits_model, merged[["yhat", "q05", "q50", "q95"]], nhits_future


def _write_forecast_tables(
    spark: Any,
    config: ProjectConfig,
    forecast: pd.DataFrame,
    run_id: str,
    model_name: str,
    model_version: str,
) -> dict[str, str]:
    output = forecast.copy()
    output["run_id"] = run_id
    output["model_name"] = model_name
    output["model_version"] = model_version
    output["horizon"] = output["horizon"].astype(int)
    contract_columns = [
        "run_id",
        "model_name",
        "model_version",
        "origin_date",
        "target_date",
        "series_id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "horizon",
        "yhat",
        "q05",
        "q50",
        "q95",
    ]
    bottom = spark.createDataFrame(output[contract_columns])
    bottom_path = table_path(config, "gold", "forecasts_bottom")
    write_delta(bottom, bottom_path)
    hierarchy = spark.read.format("delta").load(str(table_path(config, "gold", "hierarchy")))
    hierarchical = (
        bottom.join(hierarchy.select("series_id", "level", "level_name", "node_id"), "series_id")
        .groupBy(
            "run_id",
            "model_name",
            "origin_date",
            "target_date",
            "horizon",
            "level",
            "level_name",
            "node_id",
        )
        .agg(*[F.sum(column).alias(column) for column in ("yhat", "q05", "q50", "q95")])
    )
    hierarchy_path = table_path(config, "gold", "forecasts_hierarchy")
    write_delta(hierarchical, hierarchy_path, ["level"])
    return {"bottom": str(bottom_path), "hierarchy": str(hierarchy_path)}


def run_forecasting(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    import mlflow

    spark = get_spark(config, "forecasting")
    features = spark.read.format("delta").load(
        str(table_path(config, "gold", "training_features"))
    ).cache()
    daily = (
        spark.read.format("delta")
        .load(str(table_path(config, "silver", "sales_daily")))
        .cache()
    )
    future = spark.read.format("delta").load(
        str(table_path(config, "gold", "forecast_features"))
    ).drop("_run_id").toPandas()
    tuned = _tune_parameters(features, config)
    fold_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    all_predictions: list[pd.DataFrame] = []
    parent_run_id = ""
    with tracking_run(config, run_id) as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_dict(tuned, "tuning/best_parameters.json")
        for origin in config.data.backtest_origins:
            train, evaluation = _split_fold(features, origin)
            for name in config.models.names:
                with mlflow.start_run(run_name=f"{name}-d_{origin}", nested=True):
                    if name in {"seasonal_naive", "moving_average"}:
                        _, predicted = _baseline_predictions(name, train, evaluation)
                        evaluated_rows = evaluation
                    elif name in TREE_MODELS:
                        _, predicted = _tree_predictions(
                            name, train, evaluation, config, tuned.get(name, {})
                        )
                        evaluated_rows = evaluation
                    else:
                        _, predicted, future_actual = _nhits_predictions(daily, origin, config)
                        evaluated_rows = future_actual.rename(columns={"units": "target"})
                        evaluated_rows["origin_day"] = origin
                        evaluated_rows["horizon"] = evaluated_rows["day_num"] - origin
                        evaluated_rows["target_date"] = pd.to_datetime(evaluated_rows["date"])
                    forecast = _attach_predictions(evaluated_rows, predicted, name, origin)
                    metrics, hierarchy_details = _evaluate_fold(daily, forecast, origin)
                    hierarchy_path = (
                        config.paths.artifacts / "backtests" / name / f"d_{origin}-wrmsse.parquet"
                    )
                    hierarchy_path.parent.mkdir(parents=True, exist_ok=True)
                    hierarchy_details.toPandas().to_parquet(hierarchy_path, index=False)
                    mlflow.log_params({"model": name, "fold_origin": origin, **tuned.get(name, {})})
                    log_metrics(metrics)
                    mlflow.log_artifact(str(hierarchy_path), artifact_path="backtests")
                    fold_metrics[name].append(metrics)
                    all_predictions.append(forecast)
        summaries: dict[str, dict[str, float]] = {}
        for name, rows in fold_metrics.items():
            wrmsse_values = np.asarray([row["wrmsse"] for row in rows])
            summaries[name] = {
                "mean_wrmsse": float(wrmsse_values.mean()),
                "bias": float(np.mean([row["bias"] for row in rows])),
                "coverage": float(np.mean([row["coverage"] for row in rows])),
                "max_fold_degradation": 0.0,
            }
        baseline_folds = np.asarray(
            [row["wrmsse"] for row in fold_metrics.get("seasonal_naive", [])]
        )
        if len(baseline_folds):
            for name, rows in fold_metrics.items():
                candidate_folds = np.asarray([row["wrmsse"] for row in rows])
                summaries[name]["max_fold_degradation"] = float(
                    max(0.0, np.max(candidate_folds / np.maximum(baseline_folds, 1e-12) - 1))
                )
        winner = min(summaries, key=lambda name: summaries[name]["mean_wrmsse"])
        log_metrics(summaries[winner])
        mlflow.log_dict(summaries, "evaluation/model_summary.json")
        final_model, final_predicted, model_input = _fit_final(
            winner, features, future, daily, config, tuned
        )
        final = _attach_predictions(
            future.assign(target=np.nan), final_predicted, winner, config.data.forecast_origin_day
        )
        final["origin_date"] = pd.to_datetime(final["origin_date"]).dt.date
        final["target_date"] = pd.to_datetime(final["target_date"]).dt.date
        explanation_dir = config.paths.artifacts / "explainability" / run_id
        if winner in TREE_MODELS:
            explanations = explain_tree(final_model, model_input, explanation_dir)
        elif winner == "nhits":
            explanations = explain_nhits(final_model, model_input, explanation_dir)
        else:
            explanations = {}
        for artifact in explanations.values():
            mlflow.log_artifact(artifact, artifact_path="explainability")
        version = log_forecaster(config, final_model, model_input, winner)
        paths = _write_forecast_tables(spark, config, final, run_id, winner, version)
        mlflow.set_tags({"winner": winner, "registered_model_version": version})
    backtests = pd.concat(all_predictions, ignore_index=True)
    write_delta(spark.createDataFrame(backtests), table_path(config, "gold", "backtest_forecasts"))
    metrics_rows = [
        {"model_name": name, **metrics} for name, metrics in summaries.items()
    ]
    write_delta(
        spark.createDataFrame(pd.DataFrame(metrics_rows)),
        table_path(config, "gold", "model_metrics"),
    )
    promotion = promote_candidate(config, parent_run_id)
    features.unpersist()
    daily.unpersist()
    spark.stop()
    return {
        "run_id": run_id,
        "mlflow_run_id": parent_run_id,
        "winner": winner,
        "metrics": summaries[winner],
        "forecasts": paths,
        "promotion": promotion,
    }


def run_final_forecast(config: ProjectConfig, run_id: str) -> dict[str, Any]:
    path = table_path(config, "gold", "forecasts_bottom")
    if path.exists():
        return {"run_id": run_id, "forecast_table": str(path), "status": "already_materialized"}
    return run_forecasting(config, run_id)


def run_single_model(
    config: ProjectConfig, run_id: str, model_name: str, origin: int | None = None
) -> dict[str, Any]:
    """Execute one editable notebook experiment without duplicating workflow logic."""
    import mlflow

    if model_name not in config.models.names:
        raise ValueError(f"Model {model_name!r} is not enabled for profile {config.profile}")
    fold_origin = origin or config.data.backtest_origins[-1]
    spark = get_spark(config, f"notebook-{model_name}")
    features = spark.read.format("delta").load(
        str(table_path(config, "gold", "training_features"))
    )
    daily = spark.read.format("delta").load(str(table_path(config, "silver", "sales_daily")))
    train, evaluation = _split_fold(features, fold_origin)
    with tracking_run(config, run_id):
        if model_name in {"seasonal_naive", "moving_average"}:
            _, predicted = _baseline_predictions(model_name, train, evaluation)
            evaluated_rows = evaluation
            params: dict[str, Any] = {}
        elif model_name in TREE_MODELS:
            fit, validation = _validation_split(train)
            params = tune_tree_model(
                cast(ModelKind, model_name),
                fit.sample(frac=config.models.tune_fraction, random_state=config.seed),
                validation.sample(frac=config.models.tune_fraction, random_state=config.seed),
                config.models.optuna_trials,
                config.seed,
                _cuda_available(),
            )
            _, predicted = _tree_predictions(
                model_name, train, evaluation, config, params
            )
            evaluated_rows = evaluation
        else:
            _, predicted, future_actual = _nhits_predictions(daily, fold_origin, config)
            evaluated_rows = future_actual.rename(columns={"units": "target"})
            evaluated_rows["origin_day"] = fold_origin
            evaluated_rows["horizon"] = evaluated_rows["day_num"] - fold_origin
            evaluated_rows["target_date"] = pd.to_datetime(evaluated_rows["date"])
            params = {
                "input_size": config.models.nhits_input_size,
                "max_steps": config.models.nhits_max_steps,
            }
        forecast = _attach_predictions(evaluated_rows, predicted, model_name, fold_origin)
        metrics, _ = _evaluate_fold(daily, forecast, fold_origin)
        mlflow.log_params({"model": model_name, "fold_origin": fold_origin, **params})
        log_metrics(metrics)
    spark.stop()
    return {"model": model_name, "fold_origin": fold_origin, "metrics": metrics, "params": params}
