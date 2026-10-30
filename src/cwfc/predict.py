
import argparse
import os

import numpy as np
import pandas as pd

from cwfc.ar1 import ar1_forecast

HORIZON = 24
CPU_FILE = "vm_cpu_readings.csv.gz"
TARGET = "aggregate_cpu_utilization"


def build_targets(cpu_table):
    targets = (
        cpu_table.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return targets


def make_forecast(targets, horizon):
    if len(targets) <= horizon + 1:
        raise ValueError("Not enough timestamps for the requested horizon")

    train = targets.iloc[:-horizon]
    holdout = targets.iloc[-horizon:]

    cpu_series = train[TARGET]
    cpu_forecast = ar1_forecast(cpu_series, horizon)
    cpu_forecast = np.clip(cpu_forecast, 0.0, 100.0)

    last_vm_count = train["active_vm_count"].iloc[-1]
    vm_forecast = [last_vm_count] * horizon

    predictions = pd.DataFrame({
        "timestamp": holdout["timestamp"].values,
        "active_vm_count": vm_forecast,
        "aggregate_cpu_utilization": cpu_forecast,
    })
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Create predictions.csv")
    parser.add_argument("--data", default="challenge_data",
                        help="folder that contains vm_cpu_readings.csv.gz")
    parser.add_argument("--output", default="predictions.csv",
                        help="path of the csv file to create")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help="number of timestamps to forecast")
    args = parser.parse_args()

    cpu_path = os.path.join(args.data, CPU_FILE)
    cpu_table = pd.read_csv(cpu_path)

    targets = build_targets(cpu_table)
    predictions = make_forecast(targets, args.horizon)

    assert len(predictions) == args.horizon
    assert predictions.isna().sum().sum() == 0

    predictions.to_csv(args.output, index=False)
    print("Created", args.output, "with", len(predictions), "rows")
    print(predictions.head())


if __name__ == "__main__":
    main()