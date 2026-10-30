import numpy as np

from src.cwfc.ar1 import ar1_forecast


def test_forecast_length():
    series = [10, 12, 11, 13, 12]

    predictions = ar1_forecast(series, 24)

    assert len(predictions) == 24


def test_constant_series():
    series = [10, 10, 10, 10, 10]

    predictions = ar1_forecast(series, 24)

    assert np.allclose(predictions, 10)


def test_forecast_values_are_finite():
    series = [10, 12, 11, 13, 12]

    predictions = ar1_forecast(series, 24)

    assert np.isfinite(predictions).all()

def test_no_future_data_leakage():
    train = [10, 12, 11, 13, 12]

    future_1 = [15, 16, 17]
    future_2 = [90, 95, 100]

    predictions_1 = ar1_forecast(train, len(future_1))
    predictions_2 = ar1_forecast(train, len(future_2))

    assert np.allclose(predictions_1, predictions_2)