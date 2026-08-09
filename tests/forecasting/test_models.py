import sys
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from retail_forecasting.forecasting import workflow
from retail_forecasting.forecasting.models import DirectTreeForecaster


class FakeRegressor:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def get_params(self) -> dict[str, Any]:
        return self.params


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, class_name: str) -> None:
    module = ModuleType(name)
    setattr(module, class_name, FakeRegressor)
    monkeypatch.setitem(sys.modules, name, module)


def test_lightgbm_remains_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_module(monkeypatch, "lightgbm", "LGBMRegressor")
    model = DirectTreeForecaster("lightgbm", use_gpu=True)._make_model()

    assert "device_type" not in model.get_params()


def test_xgboost_uses_cuda_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_module(monkeypatch, "xgboost", "XGBRegressor")
    model = DirectTreeForecaster("xgboost", use_gpu=True)._make_model()

    assert model.get_params()["device"] == "cuda"


def test_only_xgboost_requests_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow, "_cuda_available", lambda: True)

    assert workflow._tree_uses_gpu("xgboost")
    assert not workflow._tree_uses_gpu("lightgbm")


def test_lightgbm_uses_native_categorical_dtypes() -> None:
    frame = pd.DataFrame(
        {
            "series_id": ["a", "b"],
            "item_id": ["item_1", "item_2"],
            "store_id": ["store_1", "store_1"],
            "horizon": [1, 2],
            "lag_1": [0.0, 1.0],
            "target": [0.0, 2.0],
        }
    )

    transformed = DirectTreeForecaster("lightgbm")._fit_transform(frame)

    assert isinstance(transformed["item_id"].dtype, pd.CategoricalDtype)
    assert isinstance(transformed["store_id"].dtype, pd.CategoricalDtype)
    assert transformed["lag_1"].dtype == "float32"
