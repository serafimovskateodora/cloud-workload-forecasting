
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

HORIZON = 24

meta_table = pd.read_csv("challenge_data/vm_metadata.csv.gz")
cpu_table = pd.read_csv("challenge_data/vm_cpu_readings.csv.gz")

timestamps = sorted(cpu_table["timestamp"].unique())
train_timestamps = timestamps[:-HORIZON]

train_cpu = cpu_table[
    cpu_table["timestamp"].isin(train_timestamps)
].copy()

first_timestamp = train_timestamps[0]
last_timestamp = train_timestamps[-1]

print("First metadata rows:")
print(meta_table.head())

print("\nMetadata shape:", meta_table.shape)

print("\nMissing values per column:")
print(meta_table.isna().sum())

print("\nNumber of distinct VMs:", meta_table["vm_id"].nunique())

print("\nVMs per category:")
print(meta_table["vm_category"].value_counts())

cpu_ids = train_cpu["vm_id"].unique()
meta_ids = meta_table["vm_id"].unique()

in_cpu = meta_table["vm_id"].isin(cpu_ids)
in_meta = pd.Series(cpu_ids).isin(meta_ids)

print("\nVMs in metadata but not in training CPU:",
      (~in_cpu).sum())

print("VMs in training CPU but not in metadata:",
      (~in_meta).sum())

print("\nFirst training timestamp:", first_timestamp)
print("Last training timestamp:", last_timestamp)

created_after_start = (
    (meta_table["timestamp_vm_created"] > first_timestamp)
    & (meta_table["timestamp_vm_created"] <= last_timestamp)
).sum()

print("\nVMs created during training window:",
      created_after_start)

deleted_during_training = (
    (meta_table["timestamp_vm_deleted"] > first_timestamp)
    & (meta_table["timestamp_vm_deleted"] <= last_timestamp)
).sum()

print("VMs deleted during training window:",
      deleted_during_training)

vm_avg_cpu = train_cpu.groupby("vm_id")["avg_cpu"].mean()

print("\nMinimum VM average CPU:", vm_avg_cpu.min())
print("Maximum VM average CPU:", vm_avg_cpu.max())
print("Mean VM average CPU:", vm_avg_cpu.mean())

print("VMs with average CPU below 10%:",
      (vm_avg_cpu < 10).sum())

print("VMs with average CPU above 50%:",
      (vm_avg_cpu > 50).sum())

print("\nVMs per virtual core bucket:")
print(meta_table["virtual_core_bucket"].value_counts())

print("\nVMs per memory bucket:")
print(meta_table["memory_gb_bucket"].value_counts())
