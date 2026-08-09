"""Benchmark LightGBM objectives on one untouched temporal fold."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from pyspark.sql import functions as F

from retail_forecasting.config import load_config
from retail_forecasting.data.gold import build_direct_features
from retail_forecasting.data.spark import get_spark, table_path
from retail_forecasting.forecasting.metrics import point_metrics
from retail_forecasting.forecasting.models import DirectTreeForecaster


def _best_postprocess(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    best = {"wape": float("inf"), "threshold": 0.0, "scale": 1.0}
    for threshold in np.linspace(0.0, 1.5, 16):
        thresholded = np.where(prediction < threshold, 0.0, prediction)
        for scale in np.linspace(0.6, 1.2, 25):
            score = point_metrics(actual, thresholded * scale)["wape"]
            if score < best["wape"]:
                best = {
                    "wape": float(score),
                    "threshold": float(threshold),
                    "scale": float(scale),
                }
    return best


def _best_unbiased_scale(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    best = {"wape": float("inf"), "scale": 1.0, "bias": float("inf")}
    for scale in np.linspace(0.5, 2.0, 61):
        metrics = point_metrics(actual, prediction * scale)
        if abs(metrics["bias"]) <= 0.05 and metrics["wape"] < best["wape"]:
            best = {
                "wape": float(metrics["wape"]),
                "scale": float(scale),
                "bias": float(metrics["bias"]),
            }
    return best


def _as_native_categories(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    native = frame.copy()
    for column in columns:
        native[column] = native[column].astype("int32").astype("category")
    return native


def _report_candidate(
    name: str,
    validation_actual: np.ndarray,
    validation_prediction: np.ndarray,
    evaluation_actual: np.ndarray,
    evaluation_prediction: np.ndarray,
) -> dict[str, object]:
    adjustment = _best_unbiased_scale(validation_actual, validation_prediction)
    adjusted = evaluation_prediction * adjustment["scale"]
    raw_metrics = point_metrics(evaluation_actual, evaluation_prediction)
    metrics = point_metrics(evaluation_actual, adjusted)
    print(
        f"{name:22s} raw={raw_metrics['wape']:.4f} "
        f"unbiased={metrics['wape']:.4f} bias={metrics['bias']:.4f} "
        f"scale={adjustment['scale']:.3f}"
    )
    return {
        "variant": name,
        "raw": raw_metrics,
        "unbiased_adjustment": adjustment,
        "adjusted": metrics,
    }


def _group_scale(
    calibration: pd.DataFrame,
    calibration_prediction: np.ndarray,
    scoring: pd.DataFrame,
    scoring_prediction: np.ndarray,
    columns: list[str],
    prior_units: float,
) -> np.ndarray:
    residual_frame = calibration[columns + ["target"]].copy()
    residual_frame["prediction"] = calibration_prediction
    totals = residual_frame.groupby(columns, observed=True)[["target", "prediction"]].sum()
    totals["scale"] = (totals["target"] + prior_units) / (
        totals["prediction"] + prior_units
    )
    totals["scale"] = totals["scale"].clip(0.5, 2.0)
    scales = scoring[columns].merge(
        totals[["scale"]].reset_index(), on=columns, how="left"
    )["scale"]
    return scoring_prediction * scales.fillna(1.0).to_numpy(dtype=float)


def _apply_postprocess(prediction: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    return np.where(prediction < parameters["threshold"], 0.0, prediction) * parameters[
        "scale"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Build the current feature code from Silver instead of reading stored Gold.",
    )
    args = parser.parse_args()
    config = load_config("dev")
    origin = max(config.data.backtest_origins)
    spark = get_spark(config, "tree-objective-benchmark")
    if args.rebuild_features:
        daily = spark.read.format("delta").load(
            str(table_path(config, "silver", "sales_daily"))
        )
        features = build_direct_features(daily, config)
    else:
        features = spark.read.format("delta").load(
            str(table_path(config, "gold", "training_features"))
        )
    train = features.filter(
        (F.col("origin_day") < origin) & (F.col("origin_day") + F.col("horizon") <= origin)
    ).toPandas()
    evaluation = features.filter(F.col("origin_day") == origin).toPandas()
    spark.stop()

    validation_origin = int(train["origin_day"].max())
    validation = train.loc[train["origin_day"] == validation_origin].copy()
    fit = train.loc[train["origin_day"] < validation_origin].copy()
    encoder = DirectTreeForecaster("lightgbm", seed=config.seed)
    encoded_fit = encoder._fit_transform(fit)
    encoded_validation = encoder._transform(validation)
    encoded_evaluation = encoder._transform(evaluation)
    native_fit = _as_native_categories(encoded_fit, encoder.categorical_features)
    native_validation = _as_native_categories(encoded_validation, encoder.categorical_features)
    native_evaluation = _as_native_categories(encoded_evaluation, encoder.categorical_features)
    fit_actual = fit["target"].to_numpy(dtype=float)
    validation_actual = validation["target"].to_numpy(dtype=float)
    evaluation_actual = evaluation["target"].to_numpy(dtype=float)

    common = {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": config.seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    variants = {
        "tweedie_1.1": {"objective": "tweedie", "tweedie_variance_power": 1.1},
        "tweedie_1.2": {"objective": "tweedie", "tweedie_variance_power": 1.2},
        "tweedie_1.5": {"objective": "tweedie", "tweedie_variance_power": 1.5},
        "l1": {"objective": "regression_l1"},
        "quantile_0.5": {"objective": "quantile", "alpha": 0.5},
        "poisson": {"objective": "poisson"},
        "huber": {"objective": "huber", "alpha": 0.9},
    }
    results = []
    for name, objective in variants.items():
        started = time.perf_counter()
        model = LGBMRegressor(**(common | objective))
        model.fit(encoded_fit, fit["target"].to_numpy(dtype=float))
        validation_prediction = np.clip(model.predict(encoded_validation), 0.0, None)
        postprocess = _best_postprocess(
            validation["target"].to_numpy(dtype=float), validation_prediction
        )
        evaluation_prediction = np.clip(model.predict(encoded_evaluation), 0.0, None)
        raw_metrics = point_metrics(evaluation["target"], evaluation_prediction)
        adjusted_metrics = point_metrics(
            evaluation["target"], _apply_postprocess(evaluation_prediction, postprocess)
        )
        row = {
            "variant": name,
            "validation_origin": validation_origin,
            "evaluation_origin": origin,
            "fit_rows": len(fit),
            "validation_rows": len(validation),
            "evaluation_rows": len(evaluation),
            "runtime_seconds": time.perf_counter() - started,
            "raw": raw_metrics,
            "postprocess": postprocess,
            "adjusted": adjusted_metrics,
        }
        results.append(row)
        print(
            f"{name:14s} raw={raw_metrics['wape']:.4f} "
            f"adjusted={adjusted_metrics['wape']:.4f} bias={adjusted_metrics['bias']:.4f}"
        )

    focused_common = common | {
        "objective": "tweedie",
        "tweedie_variance_power": 1.2,
        "n_estimators": 900,
        "learning_rate": 0.04,
    }
    native_tree = LGBMRegressor(**focused_common)
    native_tree.fit(
        native_fit,
        fit_actual,
        categorical_feature=encoder.categorical_features,
    )
    native_validation_prediction = np.clip(native_tree.predict(native_validation), 0.0, None)
    native_evaluation_prediction = np.clip(native_tree.predict(native_evaluation), 0.0, None)
    results.append(
        _report_candidate(
            "native_categorical",
            validation_actual,
            native_validation_prediction,
            evaluation_actual,
            native_evaluation_prediction,
        )
    )

    for label, columns, prior_units in (
        ("series_calibration", ["series_id"], 28.0),
        ("series_horizon_cal", ["series_id", "horizon"], 10.0),
        ("store_item_cal", ["store_id", "item_id"], 28.0),
        ("store_dept_cal", ["store_id", "dept_id"], 100.0),
        ("dept_horizon_cal", ["dept_id", "horizon"], 100.0),
    ):
        calibrated_evaluation = _group_scale(
            validation,
            native_validation_prediction,
            evaluation,
            native_evaluation_prediction,
            columns,
            prior_units,
        )
        metrics = point_metrics(evaluation_actual, calibrated_evaluation)
        print(
            f"{label:22s} wape={metrics['wape']:.4f} bias={metrics['bias']:.4f}"
        )
        results.append({"variant": label, "adjusted": metrics})

    classifier = LGBMClassifier(
        **(
            common
            | {
                "objective": "binary",
                "n_estimators": 700,
                "learning_rate": 0.04,
            }
        )
    )
    classifier.fit(
        native_fit,
        (fit_actual > 0).astype("int8"),
        categorical_feature=encoder.categorical_features,
    )
    positive = fit_actual > 0
    amount_model = LGBMRegressor(
        **(
            common
            | {
                "objective": "poisson",
                "n_estimators": 900,
                "learning_rate": 0.04,
            }
        )
    )
    amount_model.fit(
        native_fit.loc[positive],
        fit_actual[positive],
        categorical_feature=encoder.categorical_features,
    )
    validation_probability = classifier.predict_proba(native_validation)[:, 1]
    evaluation_probability = classifier.predict_proba(native_evaluation)[:, 1]
    validation_amount = np.clip(amount_model.predict(native_validation), 0.0, None)
    evaluation_amount = np.clip(amount_model.predict(native_evaluation), 0.0, None)
    results.append(
        _report_candidate(
            "hurdle_expected_value",
            validation_actual,
            validation_probability * validation_amount,
            evaluation_actual,
            evaluation_probability * evaluation_amount,
        )
    )

    target_lags = [f"target_lag_{lag}" for lag in (28, 35, 42, 49, 56)]
    validation_seasonal_mean = validation[target_lags].mean(axis=1).to_numpy(dtype=float)
    evaluation_seasonal_mean = evaluation[target_lags].mean(axis=1).to_numpy(dtype=float)
    results.append(
        _report_candidate(
            "seasonal_mean_5",
            validation_actual,
            validation_seasonal_mean,
            evaluation_actual,
            evaluation_seasonal_mean,
        )
    )
    best_blend: dict[str, float] | None = None
    for weight in np.linspace(0.0, 1.0, 11):
        validation_blend = (
            weight * native_validation_prediction
            + (1.0 - weight) * validation_seasonal_mean
        )
        adjustment = _best_unbiased_scale(validation_actual, validation_blend)
        if adjustment["wape"] == float("inf"):
            continue
        candidate = {
            "weight": float(weight),
            "validation_wape": float(adjustment["wape"]),
            "scale": float(adjustment["scale"]),
        }
        if best_blend is None or candidate["validation_wape"] < best_blend["validation_wape"]:
            best_blend = candidate
    if best_blend is None:
        raise RuntimeError("no bias-constrained blend was found")
    evaluation_blend = (
        best_blend["weight"] * native_evaluation_prediction
        + (1.0 - best_blend["weight"]) * evaluation_seasonal_mean
    )
    blend_metrics = point_metrics(
        evaluation_actual, evaluation_blend * best_blend["scale"]
    )
    print(
        f"{'calibrated_blend':22s} wape={blend_metrics['wape']:.4f} "
        f"bias={blend_metrics['bias']:.4f} tree_weight={best_blend['weight']:.2f}"
    )
    results.append(
        {"variant": "calibrated_blend", "selection": best_blend, "adjusted": blend_metrics}
    )

    output = Path("artifacts/experiments/tree_objectives_d1913.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
