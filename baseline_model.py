from src.cwfc.baselines import persistence_forecast
import pandas as pd
from src.cwfc.metrics import mae, rmse, normalized_mae

HORIZON = 24
MIN_TRAIN = 50
STEP = 6
MA_WINDOW = 6
TARGET = "aggregate_cpu_utilization"

cpu_table = pd.read_csv("challenge_data/vm_cpu_readings.csv.gz")

targets = cpu_table.groupby("timestamp").agg(
    active_vm_count=("vm_id", "nunique"),
    aggregate_cpu_utilization=("avg_cpu", "mean")
).reset_index()

train = targets.iloc[:-HORIZON].copy()
holdout = targets.iloc[-HORIZON:].copy()

persistence_mae, persistence_rmse = [], []
mean_mae, mean_rmse = [], []
ma_mae, ma_rmse = [], []
last_values, train_means, val_means = [], [], []

all_actual, all_pers, all_mean, all_ma = [], [], [], []

vm_count_mae = []

for o in range(MIN_TRAIN, len(train) - HORIZON + 1, STEP):

    fold_train = train.iloc[:o].copy()
    fold_val = train.iloc[o:o + HORIZON].copy()

    assert fold_train["timestamp"].max() < fold_val["timestamp"].min()

    actual = fold_val[TARGET]

    pred = persistence_forecast(fold_train, fold_val, TARGET)
    persistence_mae.append(mae(actual, pred))
    persistence_rmse.append(rmse(actual, pred))

    mean_value = fold_train[TARGET].mean()
    pred_mean = pd.Series(mean_value, index=fold_val.index)
    mean_mae.append(mae(actual, pred_mean))
    mean_rmse.append(rmse(actual, pred_mean))

    ma_value = fold_train[TARGET].iloc[-MA_WINDOW:].mean()
    pred_ma = pd.Series(ma_value, index=fold_val.index)
    ma_mae.append(mae(actual, pred_ma))
    ma_rmse.append(rmse(actual, pred_ma))

    last_values.append(fold_train[TARGET].iloc[-1])
    train_means.append(mean_value)
    val_means.append(actual.mean())

    all_actual.extend(actual)
    all_pers.extend(pred)
    all_mean.extend(pred_mean)
    all_ma.extend(pred_ma)

    pred_vm = persistence_forecast(fold_train, fold_val, "active_vm_count")
    vm_count_mae.append(mae(fold_val["active_vm_count"], pred_vm))

print("=== aggregate_cpu_utilization: mean over folds ===")
print("Persistence      MAE:", sum(persistence_mae) / len(persistence_mae),
      " RMSE:", sum(persistence_rmse) / len(persistence_rmse))
print("Mean-baseline    MAE:", sum(mean_mae) / len(mean_mae),
      " RMSE:", sum(mean_rmse) / len(mean_rmse))
print("Moving-average   MAE:", sum(ma_mae) / len(ma_mae),
      " RMSE:", sum(ma_rmse) / len(ma_rmse))

print("\n=== Normalized MAE (relative to persistence, < 1 means better) ===")
print("Mean-baseline  :", normalized_mae(all_actual, all_mean, all_pers))
print("Moving-average :", normalized_mae(all_actual, all_ma, all_pers))

print("\n=== active_vm_count persistence MAE per fold ===")
print(vm_count_mae)

print("\norigin  pers_MAE  mean_MAE  ma_MAE  last_value  train_mean  val_mean")
origins = list(range(MIN_TRAIN, len(train) - HORIZON + 1, STEP))
for k, o in enumerate(origins):
    print(o,
          round(persistence_mae[k], 4), round(mean_mae[k], 4), round(ma_mae[k], 4),
          round(last_values[k], 3), round(train_means[k], 3), round(val_means[k], 3))
