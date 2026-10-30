import pandas as pd

def test_predictions_structure():
    predictions = pd.read_csv("predictions.csv")

    assert len(predictions) == 24

    assert list(predictions.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization"
    ]

    assert predictions.isna().sum().sum() == 0

def test_timestamps():
    predictions = pd.read_csv("predictions.csv")

    timestamps = predictions["timestamp"]

    assert timestamps.is_unique
    assert (timestamps.diff().dropna() == 300).all()

def test_cpu_range():
    predictions = pd.read_csv("predictions.csv")

    cpu = predictions["aggregate_cpu_utilization"]

    assert cpu.between(0, 100).all()

def test_vm_count():
    predictions = pd.read_csv("predictions.csv")

    vm_count = predictions["active_vm_count"]

    assert (vm_count >= 0).all()
    assert (vm_count == vm_count.astype(int)).all()
