import pandas as pd

from src.cwfc.capacity import add_safety_margin

predictions = pd.read_csv("predictions.csv")

predictions["cpu_with_margin"] = predictions[
    "aggregate_cpu_utilization"
].apply(add_safety_margin)

print(predictions.head())

print("\nMaximum predicted CPU:",
      predictions["aggregate_cpu_utilization"].max())

print("Maximum CPU with safety margin:",
      predictions["cpu_with_margin"].max())

max_underprediction = 0.527259

print("\nMaximum observed validation underprediction:",
      max_underprediction)

print("Illustrative safety margin:", 5.0)

print("Margin exceeds observed validation error:",
      5.0 > max_underprediction)