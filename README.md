# Cloud Workload Forecasting Challenge

**My solution:** see [Implemented Solution](#implemented-solution) below and the [final report](reports/final_report.md).

Quick start (after installing the project and extracting the dataset into `challenge_data/`):

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

AI/ML take-home challenge built from the **Microsoft Azure Public Dataset V2** VM trace.

## Start here

1. Read [`ASSIGNMENT.md`](ASSIGNMENT.md).
2. Read [`DATASET.md`](DATASET.md).
3. Download the **`data-v1` GitHub Release** dataset and extract it as `challenge_data/`.
4. Read [`SUBMISSION.md`](SUBMISSION.md).
5. Implement your analysis and forecasting pipeline.

The challenge dataset is distributed as a versioned GitHub Release rather than committed to the repository. The `data-v1` release contains the fixed curated dataset used for this assignment.

## Environment setup

Python **3.11 or newer** is required.

Create a virtual environment:

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python -m pytest -q
```

The repository includes a small test suite covering the starter forecasting utilities. Run it before beginning the assignment.

## Dataset download

The hiring dataset is distributed as the fixed `data-v1` release of this repository.

**Download:** [Azure Dataset v1 Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

Download `data-v1.tar.gz` from the release and extract it into the repository as `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

The resulting layout is:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Verify the dataset before starting the assignment:

```bash
cd challenge_data
sha256sum -c SHA256SUMS
```

You should see:

```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

The release is a real, curated subset of the Microsoft Azure Public Dataset V2. It is not synthetic data.

The dataset contains 10,000 VMs with 134 CPU measurements per VM at five-minute intervals, for a total of 1,340,000 CPU observations.

Use the exact `data-v1` release provided for this assignment. Do not substitute another dataset or a different version of the Azure Public Dataset.

## What you are solving

The underlying data contains anonymized VM workload telemetry sampled every five minutes. Your job is to understand the workload and build a defensible forecasting pipeline for near-term aggregate demand.

The challenge is intentionally open-ended. We evaluate reasoning, time-series methodology, leakage prevention, error analysis, reproducibility, and the quality of your operational conclusions—not just model complexity.

## Expected environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Model dependencies are intentionally not prescribed. Add only what you need and document them.

## Source

Microsoft Azure Public Dataset V2:
https://github.com/Azure/AzurePublicDataset

The official V2 documentation describes a 30-day VM trace with five-minute CPU readings, VM metadata, and encrypted subscription/deployment/VM identifiers. The full upstream trace is much larger than this challenge release.

## Submission

Submit your work as a repository or archive according to [`SUBMISSION.md`](SUBMISSION.md).

## Implemented Solution

This repository contains my solution to the Atomity Cloud Workload Forecasting Challenge.

The full write-up is in [`reports/final_report.md`](reports/final_report.md).

### Approach

The forecasting pipeline predicts two targets over a 24-step (2-hour) horizon:

- **active_vm_count:** Number of unique active virtual machines at each timestamp.
- **aggregate_cpu_utilization:** Mean CPU utilization across active virtual machines.

The first 110 timestamps are used for training, while the final 24 timestamps are reserved for holdout evaluation.

### Forecasting Models

Several simple forecasting methods were evaluated:

- Persistence baseline
- Historical mean baseline
- Moving average baseline
- Autoregressive AR(1) model

The AR(1) model was selected for aggregate CPU forecasting based on rolling-origin validation. Persistence forecasting was used for active VM count.

### Validation

The models were evaluated using time-aware rolling-origin validation.

Performance metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Relative improvement compared with the persistence baseline

The final holdout set was evaluated only after model selection.

### Generate Predictions

After installing the project dependencies and extracting the dataset, run:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This generates a CSV containing 24 forecast rows with the required columns:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

### Run Tests

```bash
python -m pytest tests/ -v
```

The test suite covers forecasting calculations, metrics, temporal validation, leakage prevention, prediction structure, and execution on a small synthetic dataset.

### Capacity Planning

A supplementary capacity analysis explores the effect of adding an illustrative five-percentage-point safety margin to aggregate CPU forecasts.

The analysis discusses the trade-off between underprovisioning risk and overprovisioning cost. The margin is not a statistically calibrated guarantee, and aggregate utilization alone is insufficient to recommend shutting down individual virtual machines.

Run the supplementary capacity analysis after generating
predictions.csv:

```bash
python capacity_analysis.py
```

### Results
#### Rolling-Origin Validation

Seven rolling-origin validation folds were used, with a
24-step forecasting horizon.

| Model | CPU MAE | CPU RMSE |
|---|---:|---:|
| Persistence | 0.2152 | 0.2514 |
| Historical Mean | 0.2088 | 0.2384 |
| Moving Average (6) | 0.2428 | 0.2724 |
| AR(1) | 0.2009 | 0.2373 |

AR(1) achieved approximately 6.68% lower MAE than
persistence during rolling-origin validation.

On the final 24-step holdout:

| Model | CPU MAE | CPU RMSE |
|---|---:|---:|
| AR(1) | 0.153321 | 0.183240 |
| Persistence | 0.146782 | 0.184785 |

Active VM count forecasting achieved zero MAE and RMSE on the holdout, where the VM count remained constant.

AR(1) slightly improved CPU RMSE but performed worse than persistence on CPU MAE. This limitation is discussed in the final report.




### Limitations

The dataset covers approximately 11 hours of observations,
which limits the ability to identify longer-term patterns.

AR(1) tends to smooth forecasts toward the historical mean,
potentially underestimating CPU peaks and overestimating
low-utilization periods.

The active VM count remains constant in this dataset,
so zero forecasting error does not demonstrate performance
under changing VM populations.