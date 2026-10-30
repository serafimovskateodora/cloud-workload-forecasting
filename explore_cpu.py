
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

HORIZON = 24

cpu_table = pd.read_csv("challenge_data/vm_cpu_readings.csv.gz")

cpu_per_timestamp = (
    cpu_table.groupby("timestamp")["avg_cpu"]
    .mean()
    .sort_index()
)

train_cpu = cpu_per_timestamp.iloc[:-HORIZON]

train_table = cpu_table[
    cpu_table["timestamp"].isin(train_cpu.index)
].copy()

print("Training CPU observations:", train_table.shape)
print("Training timestamps:", len(train_cpu))

rolling_cpu = train_cpu.rolling(window=6).mean()

plt.figure(figsize=(10, 5))

plt.plot(
    train_cpu.index,
    train_cpu.values,
    label="Original CPU"
)

plt.plot(
    rolling_cpu.index,
    rolling_cpu.values,
    label="Rolling Mean (30 min)"
)

plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Aggregate CPU utilization (%)")
plt.title("Aggregate CPU Utilization Over Time (Training Data)")
plt.tight_layout()

plt.savefig("cpu_graph.png")
plt.show()

plot_acf(train_cpu, lags=20)
plt.title("CPU Autocorrelation (Training Data)")
plt.tight_layout()
plt.show()

invalid_min = (
    train_table["min_cpu"] > train_table["avg_cpu"]
).sum()

print("Invalid min_cpu values:", invalid_min)

invalid_max = (
    train_table["max_cpu"] < train_table["avg_cpu"]
).sum()

print("Invalid max_cpu values:", invalid_max)

invalid_negative = (
    train_table["min_cpu"] < 0
).sum()

print("Invalid negative min_cpu values:", invalid_negative)

invalid_above_100 = (
    train_table["max_cpu"] > 100
).sum()

print("Invalid max_cpu values above 100:", invalid_above_100)

vm_variation = (
    train_table.groupby("vm_id")["avg_cpu"]
    .std()
)

print("\nTop 10 VMs with highest CPU variability:")
print(vm_variation.sort_values(ascending=False).head(10))
