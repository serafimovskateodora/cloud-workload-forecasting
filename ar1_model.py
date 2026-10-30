from src.cwfc.baselines import persistence_forecast
import pandas as pd
from src.cwfc.metrics import mae, rmse, normalized_mae
import numpy as np

HORIZON = 24
MIN_TRAIN = 50
STEP = 6
TARGET = "aggregate_cpu_utilization"

cpu_table = pd.read_csv("challenge_data/vm_cpu_readings.csv.gz")

targets = cpu_table.groupby("timestamp").agg(
    active_vm_count=("vm_id", "nunique"),
    aggregate_cpu_utilization=("avg_cpu", "mean")
).reset_index()

train = targets.iloc[:-HORIZON].copy()
holdout = targets.iloc[-HORIZON:].copy()

origins = range(MIN_TRAIN, len(train) - HORIZON + 1, STEP)

all_actual = []
all_predictions = []
all_persistence = []

for origin in origins:
    fold_train=train.iloc[:origin]
    series = fold_train[TARGET]

    validation = train.iloc[origin:origin + HORIZON]
    actual = validation[TARGET].values

    mean_value = series.mean()
    last_value = series.iloc[-1]

    centered = series - mean_value


    previous = centered.iloc[:-1].values
    current = centered.iloc[1:].values

    numerator = (current * previous).sum()
    denominator = (previous * previous).sum()

    phi = numerator / denominator

    print("origin:", origin, " phi:", phi)

    forecast_list=[]
    for h in range(1, HORIZON + 1):
        forecast_value=mean_value+phi**h*(last_value-mean_value)
        forecast_list.append(forecast_value)

    assert fold_train["timestamp"].max() < validation["timestamp"].min()
    assert len(forecast_list) == len(actual)

    fold_mae = mae(actual, forecast_list)
    print("origin:", origin, " phi:", phi, " MAE:", fold_mae)

    fold_rmse = rmse(actual, forecast_list)
    print("origin:", origin, " MAE:",fold_mae , " RMSE:", fold_rmse)

    all_actual.extend(actual)
    all_predictions.extend(forecast_list)

    persistence = persistence_forecast(
        fold_train,
        validation,
        TARGET
    )

    all_persistence.extend(persistence.values)

    errors = np.abs(actual - np.array(forecast_list))

    max_error = errors.max()
    max_error_index = errors.argmax()

    print("Origin:", origin)
    print("Maximum error:", max_error)
    print("Timestamp:", validation.iloc[max_error_index]["timestamp"])
    print("Actual CPU:", actual[max_error_index])
    print("Predicted CPU:", forecast_list[max_error_index])

    signed_errors = actual - np.array(forecast_list)

    max_underprediction = max(0.0, signed_errors.max())

    print("Origin:", origin)
    print("Maximum underprediction:", max_underprediction)

print("AR(1) Mean MAE:", mae(all_actual, all_predictions))
print("AR(1) RMSE:", rmse(all_actual, all_predictions))
print("Actual:", len(all_actual))
print("AR(1):", len(all_predictions))
print("Persistence:", len(all_persistence))

ar1_normalized = normalized_mae(
    all_actual,
    all_predictions,
    all_persistence
)

print("AR(1) Normalized MAE:", ar1_normalized)
