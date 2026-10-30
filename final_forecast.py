
import pandas as pd
import numpy as np

from src.cwfc.ar1 import ar1_forecast
from src.cwfc.metrics import mae, rmse

HORIZON = 24
TARGET = "aggregate_cpu_utilization"

cpu_table = pd.read_csv("challenge_data/vm_cpu_readings.csv.gz")

targets = (
    cpu_table.groupby("timestamp")
    .agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean")
    )
    .reset_index()
    .sort_values("timestamp")
    .reset_index(drop=True)
)

train = targets.iloc[:-HORIZON].copy()
holdout = targets.iloc[-HORIZON:].copy()

series = train[TARGET]
last_value = series.iloc[-1]

forecast_list = ar1_forecast(series, HORIZON)

print("Number of CPU predictions:", len(forecast_list))
print("CPU predictions:", forecast_list)

last_vm_count = train["active_vm_count"].iloc[-1]
vm_predictions = [last_vm_count] * HORIZON

print("Last VM count:", last_vm_count)
print("Number of VM predictions:", len(vm_predictions))

predictions = pd.DataFrame({
    "timestamp": holdout["timestamp"].values,
    "active_vm_count": vm_predictions,
    "aggregate_cpu_utilization": forecast_list
})

assert len(predictions) == HORIZON
assert predictions.isna().sum().sum() == 0

print("\nFirst 5 predictions:")
print(predictions.head())

predictions.to_csv("predictions.csv", index=False)

print("\npredictions.csv created successfully!")

actual_cpu = holdout[TARGET].values
predicted_cpu = predictions[TARGET].values

print("\nAR(1) holdout evaluation:")
print("Holdout CPU MAE:", mae(actual_cpu, predicted_cpu))
print("Holdout CPU RMSE:", rmse(actual_cpu, predicted_cpu))

persistence_cpu = np.repeat(last_value, HORIZON)

print("\nPersistence holdout evaluation:")
print("Holdout Persistence MAE:", mae(actual_cpu, persistence_cpu))
print("Holdout Persistence RMSE:", rmse(actual_cpu, persistence_cpu))

actual_vm = holdout["active_vm_count"].values
predicted_vm = predictions["active_vm_count"].values

print("\nActive VM count holdout evaluation:")
print("Holdout VM MAE:", mae(actual_vm, predicted_vm))
print("Holdout VM RMSE:", rmse(actual_vm, predicted_vm))
