"""GPU-aware global boosting models with a shared forecasting interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from retail_forecasting.forecasting.calibration import ResidualCalibrator

ModelKind = Literal["lightgbm", "xgboost"]
CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "target_event_type",
]
EXCLUDED_COLUMNS = {
    "series_id",
    "origin_date",
    "target_date",
    "target",
    "_run_id",
}


@dataclass
class DirectTreeForecaster:
    kind: ModelKind
    seed: int = 42
    use_gpu: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    encoder: OrdinalEncoder | None = None
    feature_names: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    calibrator: ResidualCalibrator = field(default_factory=ResidualCalibrator)

    @property
    def name(self) -> str:
        return self.kind

    def _make_model(self) -> Any:
        if self.kind == "lightgbm":
            from lightgbm import LGBMRegressor

            defaults: dict[str, Any] = {
                "objective": "tweedie",
                "tweedie_variance_power": 1.2,
                "n_estimators": 800,
                "learning_rate": 0.04,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": self.seed,
                "n_jobs": -1,
                "verbosity": -1,
            }
            return LGBMRegressor(**(defaults | self.params))
        from xgboost import XGBRegressor

        defaults = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": 1.2,
            "n_estimators": 800,
            "learning_rate": 0.04,
            "max_depth": 9,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": self.seed,
            "tree_method": "hist",
            "device": "cuda" if self.use_gpu else "cpu",
            "n_jobs": -1,
        }
        return XGBRegressor(**(defaults | self.params))

    def _fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.feature_names = [column for column in frame.columns if column not in EXCLUDED_COLUMNS]
        self.categorical_features = [
            column for column in CATEGORICAL_FEATURES if column in self.feature_names
        ]
        features = frame[self.feature_names].copy()
        if self.categorical_features:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
            )
            encoded = self.encoder.fit_transform(
                features[self.categorical_features].fillna("__missing__").astype(str)
            )
            features.loc[:, self.categorical_features] = encoded
        return cast(
            pd.DataFrame,
            features.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32"),
        )

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.reindex(columns=self.feature_names).copy()
        if self.encoder is not None and self.categorical_features:
            encoded = self.encoder.transform(
                features[self.categorical_features].fillna("__missing__").astype(str)
            )
            features.loc[:, self.categorical_features] = encoded
        return cast(
            pd.DataFrame,
            features.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32"),
        )

    def fit(
        self, train: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> DirectTreeForecaster:
        if "target" not in train:
            raise ValueError("training data must contain target")
        self.model = self._make_model()
        self.model.fit(self._fit_transform(train), train["target"].to_numpy(dtype=float))
        calibration = validation if validation is not None and len(validation) else train
        calibration_prediction = self.predict_point(calibration)
        self.calibrator.fit(
            calibration["target"].to_numpy(dtype=float),
            calibration_prediction,
            calibration["horizon"].to_numpy(dtype=int),
        )
        return self

    def predict_point(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("forecaster has not been fitted")
        return np.clip(np.asarray(self.model.predict(self._transform(frame)), dtype=float), 0, None)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        point = self.predict_point(frame)
        q05, q50, q95 = self.calibrator.predict(
            point, frame["horizon"].to_numpy(dtype=int)
        )
        return pd.DataFrame({"yhat": point, "q05": q05, "q50": q50, "q95": q95})


def tune_tree_model(
    kind: ModelKind,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    trials: int,
    seed: int,
    use_gpu: bool,
) -> dict[str, Any]:
    """Tune on a deterministic sample and return only serializable parameters."""
    import optuna

    from retail_forecasting.forecasting.metrics import point_metrics

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        common = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        if kind == "lightgbm":
            common["num_leaves"] = trial.suggest_int("num_leaves", 31, 127)
        else:
            common["max_depth"] = trial.suggest_int("max_depth", 5, 12)
            common["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 12)
        model = DirectTreeForecaster(kind, seed=seed, use_gpu=use_gpu, params=common)
        model.fit(train)
        predicted = model.predict_point(validation)
        return point_metrics(validation["target"], predicted)["wape"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params)
