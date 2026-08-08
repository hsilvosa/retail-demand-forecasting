import numpy as np
import pandas as pd

from retail_forecasting.forecasting.baselines import MovingAverage, SeasonalNaive


def test_seasonal_naive_repeats_last_week_for_all_horizons() -> None:
    frame = pd.DataFrame({"horizon": np.arange(1, 15)})
    for lag in range(1, 8):
        frame[f"lag_{lag}"] = lag * 10
    predicted = SeasonalNaive().predict(frame)
    np.testing.assert_array_equal(predicted[:7], [70, 60, 50, 40, 30, 20, 10])
    np.testing.assert_array_equal(predicted[7:], predicted[:7])


def test_moving_average_clips_negative_values() -> None:
    frame = pd.DataFrame({"rolling_mean_28": [-1.0, 2.5]})
    np.testing.assert_array_equal(MovingAverage().predict(frame), [0.0, 2.5])
