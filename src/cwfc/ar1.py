import numpy as np


def ar1_forecast(series, horizon):
    values = np.asarray(series, dtype=float)

    mean_value = values.mean()
    last_value = values[-1]

    centered = values - mean_value

    previous = centered[:-1]
    current = centered[1:]

    numerator = (previous * current).sum()
    denominator = (previous * previous).sum()

    phi = numerator / denominator if denominator > 0 else 0.0

    predictions = []

    for h in range(1, horizon + 1):
        forecast_value = mean_value + phi**h * (last_value - mean_value)
        predictions.append(forecast_value)

    return np.array(predictions)