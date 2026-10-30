import subprocess
import sys

import numpy as np
import pandas as pd

from cwfc.ar1 import ar1_forecast
from cwfc.predict import build_targets, make_forecast, HORIZON


def make_fake_cpu_table(vm_count, timestamp_count, seed):
    random_generator = np.random.default_rng(seed)
    rows = []
    for timestamp_index in range(timestamp_count):
        for vm_index in range(vm_count):
            rows.append({
                "timestamp": timestamp_index * 300,
                "vm_id": "vm" + str(vm_index),
                "min_cpu": 1.0,
                "max_cpu": 50.0,
                "avg_cpu": float(random_generator.uniform(5, 15)),
            })
    return pd.DataFrame(rows)


def test_ar1_formula_on_a_linear_series():
    predictions = ar1_forecast([1, 2, 3, 4, 5], 2)

    expected_1 = 3 + (4 / 6) * (5 - 3)
    expected_2 = 3 + (4 / 6) ** 2 * (5 - 3)

    assert np.allclose(predictions, [expected_1, expected_2])


def test_holdout_targets_do_not_change_predictions():
    cpu_table = make_fake_cpu_table(5, 60, 1)
    targets_a = build_targets(cpu_table)

    targets_b = targets_a.copy()
    holdout_index = targets_b.index[-HORIZON:]
    targets_b.loc[holdout_index, "aggregate_cpu_utilization"] = 99.0
    targets_b.loc[holdout_index, "active_vm_count"] = 1

    predictions_a = make_forecast(targets_a, HORIZON)
    predictions_b = make_forecast(targets_b, HORIZON)

    pd.testing.assert_frame_equal(predictions_a, predictions_b)


    targets_c = targets_a.copy()
    last_train_index = targets_c.index[-HORIZON - 1]
    targets_c.loc[last_train_index, "aggregate_cpu_utilization"] = 99.0
    predictions_c = make_forecast(targets_c, HORIZON)

    same_cpu = np.allclose(
        predictions_a["aggregate_cpu_utilization"],
        predictions_c["aggregate_cpu_utilization"],
    )
    assert not same_cpu


def test_command_line_on_a_small_dataset(tmp_path):
    timestamp_count = 40
    cpu_table = make_fake_cpu_table(5, timestamp_count, 2)
    cpu_table.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    output_path = tmp_path / "predictions.csv"

    subprocess.run(
        [sys.executable, "-m", "cwfc.predict",
         "--data", str(tmp_path),
         "--output", str(output_path)],
        check=True,
    )

    predictions = pd.read_csv(output_path)

    expected_timestamps = []
    for step in range(HORIZON):
        expected_timestamps.append((timestamp_count - HORIZON + step) * 300)

    assert list(predictions.columns) == [
        "timestamp", "active_vm_count", "aggregate_cpu_utilization"
    ]
    assert len(predictions) == HORIZON
    assert list(predictions["timestamp"]) == expected_timestamps
    assert (predictions["active_vm_count"] == 5).all()
    assert predictions["aggregate_cpu_utilization"].between(0, 100).all()
    assert predictions.isna().sum().sum() == 0
