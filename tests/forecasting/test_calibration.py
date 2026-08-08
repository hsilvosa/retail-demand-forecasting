import numpy as np

from retail_forecasting.forecasting.calibration import ResidualCalibrator


def test_calibration_is_horizon_specific_and_monotonic() -> None:
    predicted = np.full(12, 5.0)
    actual = predicted + np.array([-2, -1, 0, 1, 2, 3] * 2)
    horizons = np.array([1] * 6 + [2] * 6)
    calibrator = ResidualCalibrator().fit(actual, predicted, horizons)
    q05, q50, q95 = calibrator.predict(np.array([5.0, 5.0]), np.array([1, 99]))
    assert np.all(q05 <= q50)
    assert np.all(q50 <= q95)
    fallback = np.maximum(5.0 + np.asarray(calibrator.fallback), 0)
    np.testing.assert_allclose([q05[1], q50[1], q95[1]], fallback)
    assert 1 in calibrator.residual_quantiles
