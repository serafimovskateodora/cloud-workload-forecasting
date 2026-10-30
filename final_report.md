Cloud Workload Forecasting — Final Report
Atomity AI/ML Engineering Intern Take-Home Challenge
Forecast horizon: 24 five-minute steps (2 hours)
Dataset: Curated subset of Microsoft Azure Public Dataset V2
1. Executive Summary
This project forecasts two cloud workload indicators: the number of active virtual machines (active_vm_count) and their mean CPU utilization (aggregate_cpu_utilization). The supplied subset contains 10,000 VMs observed at 134 consecutive five-minute timestamps. The first 110 timestamps were reserved for development and the final 24 for a single, post-selection holdout evaluation.

I compared persistence, historical mean, and six-step moving-average baselines with a simple autoregressive AR(1) forecast. Model selection used seven expanding-window rolling-origin validation folds entirely inside the development period. AR(1) achieved a pooled validation mean absolute error (MAE) of 0.2009 CPU percentage points, versus 0.2152 for persistence, a 6.68% reduction. Its pooled validation root mean squared error (RMSE) was 0.2373 percentage points.

The final holdout tells a more nuanced story. AR(1) achieved MAE 0.1533 and RMSE 0.1832, while persistence achieved MAE 0.1468 and RMSE 0.1848. Thus, AR(1) had slightly worse holdout MAE but slightly better RMSE. The selected AR(1) model was not changed after examining holdout performance. Active VM count remained 10,000 throughout this particular subset, so persistence predicted that target without error; this should not be generalized to dynamic production environments.

The operational takeaway is that short-term aggregate CPU forecasting is feasible with transparent, lightweight methods, but a low average error does not eliminate peak underprediction risk. A simple five-percentage-point planning margin is illustrated as a scenario, not a calibrated service-level guarantee.
2. Dataset, Targets, and Data Quality
The input consists of vm_cpu_readings.csv.gz and vm_metadata.csv.gz. The CPU file contains the columns timestamp, vm_id, min_cpu, max_cpu, and avg_cpu. At each timestamp, the targets are constructed as:
\[
\text{active\_vm\_count}_t = \operatorname{nunique}(\text{vm\_id at }t)
\]
\[
\text{aggregate\_cpu\_utilization}_t = \operatorname{mean}(\text{avg\_cpu at }t)
\]
The CPU dataset has 1,340,000 records, 10,000 distinct VMs, and 134 timestamps. The timestamps are separated by exactly 300 seconds, with no gaps or duplicate (timestamp, vm_id) pairs. No missing values were found in the five CPU columns. All 10,000 VMs have a reading at every timestamp. The development portion contains 1,100,000 records across 110 timestamps.

Training-period checks found zero cases where min_cpu > avg_cpu, max_cpu < avg_cpu, min_cpu < 0, or max_cpu > 100. The aggregate training CPU series had mean approximately 9.41%, standard deviation 0.19 percentage points, minimum 9.11%, and maximum 9.91%. It exhibited a temporary elevated period around timestamps 16,000–21,000 seconds, followed by a decline. This pattern is visible in the original series and a six-observation (30-minute) rolling mean; it does not establish a recurring cycle or long-term trend.

The training-series autocorrelation at lag one was approximately 0.66, suggesting that recent values could help forecast near-term CPU usage. However, the elevated period may itself contribute to this correlation, so predictive value must be judged by temporal validation rather than autocorrelation alone.

Metadata contains 10,000 rows and 11 columns, with no missing values and complete VM-ID overlap with the CPU file. The VM categories are approximately 61.8% delay-insensitive, 30.9% interactive, and 7.3% unknown. Hardware buckets provide descriptive context about workload diversity. Metadata fields such as avg_cpu, max_cpu, p95_max_cpu, and timestamp_vm_deleted summarize a broader trace and could incorporate information unavailable at the forecast origin; they were not used as predictive features. The selected model uses only historical aggregate target values.
3. Experimental Design and Leakage Prevention
The first 110 timestamps form the development set. The final 24 timestamps form a strictly chronological two-hour holdout. No random split was used, because it would mix past and future observations and give an unrealistic estimate of forecasting performance.

Within the development set, I used expanding-window rolling-origin validation with a 24-step forecast horizon, minimum training length of 50, and origin step of 6. The seven origins were 50, 56, 62, 68, 74, 80, and 86. At each origin, model parameters were estimated using only observations before that origin; predictions were compared against the next 24 observations. The final validation window ends at development index 109, so none of the 24 holdout target values enters validation.

Because the step is shorter than the horizon, validation windows overlap. Consequently, the seven folds and their 168 pooled forecast–actual pairs are not independent. Small performance differences should be interpreted cautiously, and the short development period limits confidence about behavior across other workloads and dates.

The final forecast is fitted once using the development period. Holdout timestamps can be used to label the output rows, but holdout target values are used only for final evaluation, not for model fitting, tuning, or selection.
4. Forecasting Methods
Four methods were evaluated for aggregate CPU:
1. Persistence: repeat the latest observed aggregate CPU value for all 24 future steps.
2. Historical mean: repeat the mean of the available training history.
3. Moving average (six observations): repeat the average of the latest six measurements (30 minutes).
4. AR(1): estimate how strongly the most recent observation should influence a forecast that gradually returns toward the training mean.
For AR(1), the forecast at horizon \(h\) is:
\[
\hat y_{t+h} = \bar y_{\mathrm{train}} + \hat\phi^{h}\big(y_t-\bar y_{\mathrm{train}}\big)
\]
Here, \(\bar y_{\mathrm{train}}\) is the mean of the current training window, \(y_t\) is its last observation, and \(\hat\phi\) is estimated from consecutive mean-centered training observations by a lag-one regression through the origin. These quantities are re-estimated independently in each validation fold. The implementation guards against a zero denominator when estimating \(\phi\) on a constant series.

This model is a reasonable intermediate choice between persistence and a historical-mean forecast: recent deviations matter most at short horizons, while their influence decays with \(h\). It is simple enough to audit and appropriate to test with only 110 development timestamps. The estimated \(\phi\) varied substantially across validation folds (approximately 0.028 to 0.648), another indication that the short series is not entirely stable.

For active_vm_count, the development series was constant at 10,000, so persistence was chosen instead of fitting an unnecessarily complex model. This choice is specific to the supplied subset.
5. Rolling-Origin Validation Results
All errors below are in CPU percentage points, not relative percent error. MAE is the mean absolute deviation from actual CPU utilization; RMSE squares errors before averaging and therefore penalizes larger misses more strongly.
Model	Validation MAE	Validation RMSE*
Persistence	0.2152	0.2514
Historical mean	0.2088	0.2384
Moving average (6)	0.2428	0.2724
AR(1)	0.2009	0.2373


Reporting qualification: The recorded baseline script reports the mean of seven fold RMSE values, whereas the recorded AR(1) result is RMSE pooled across all 168 validation predictions. These RMSE figures are therefore not strictly aggregation-equivalent. They are included for transparency, but the cross-model ranking should be based primarily on the consistently aggregated MAE until baseline pooled RMSE is recomputed. All folds have the same horizon, so mean fold MAE and pooled MAE coincide.

The normalized pooled AR(1) MAE relative to persistence was 0.93318, or a 6.68% reduction in MAE. Relative to historical mean, AR(1)'s MAE improvement was modest (approximately 3.8%). The moving-average baseline performed worse than persistence despite its smoothing; simply averaging recent values did not solve the changing-level problem.

Performance also differed across origins. AR(1)'s per-fold MAE ranged from approximately 0.1502 (origin 62) to 0.3026 (origin 50). The historical mean and persistence baselines alternated in effectiveness as the observed series moved through and away from the temporary elevated period. AR(1) did not outperform every baseline in every fold. Together with overlapping windows, this argues against claiming a robust or universal improvement from a small validation margin.

For active VM count, persistence achieved zero MAE across all seven folds, consistent with the constant development series.
6. Peak Behavior and Failure Analysis
Average accuracy alone can hide operationally important misses. In the fold beginning at origin 50, the largest AR(1) error occurred at timestamp 16,500 seconds: actual aggregate CPU was 9.912175%, while the prediction was 9.384916%. This was an underprediction of 0.527259 percentage points. Other elevated-period folds also showed large underpredictions, including approximately 0.4999 and 0.4025 percentage points.

Later folds exhibited the opposite behavior. At timestamp 27,300 seconds, one validation fold predicted 9.466739% while actual CPU was 9.115401%, an overprediction of 0.351338 percentage points. In the final validation fold (origin 86), all 24 predictions were above actual values.

The pattern is consistent with the AR(1) forecast reverting toward the historical mean: it can underestimate a temporarily elevated load and overestimate a lower subsequent load. This is an interpretation supported by the observed errors, not proof of the underlying cause. In a capacity-planning context, underprediction can leave too little headroom, whereas overprediction can lead to unnecessary resource allocation. Neither actual service-level impacts nor monetary costs were measured in this challenge.
7. Final Forecast and Holdout Evaluation
After model selection using development-only validation, AR(1) was fitted to all 110 training timestamps. The fitted training mean was 9.414089%, the last observed training CPU was 9.164556%, and the estimated \(\phi\) was 0.672319. Its first forecast was approximately 9.246323%; by step 24 it approached 9.414071%. The active-VM forecast was 10,000 at every step.

The generated predictions.csv has 24 chronological rows with exactly these columns:
timestamp,active_vm_count,aggregate_cpu_utilization
The forecast timestamps begin at 33000, then advance by 300 seconds per row. The CSV contains no index column. The reproducible pipeline command is:
python -m cwfc.predict --data challenge_data --output predictions.csv
Final holdout results, evaluated once after choosing the model, were:
Target and model	Holdout MAE	Holdout RMSE
Aggregate CPU — AR(1)	0.1533	0.1832
Aggregate CPU — persistence	0.1468	0.1848
Active VM count — persistence	0.0000	0.0000


AR(1)'s holdout MAE was approximately 4.45% higher than persistence, although its RMSE was approximately 0.84% lower. This differs from the validation MAE ranking. The discrepancy is an important result: the training-validation advantage did not translate into a consistent holdout advantage. I retained the preselected AR(1) model rather than switching methods or tuning against the holdout.
8. Capacity Planning Interpretation
A forecast can inform capacity decisions, but aggregate utilization is not itself a direct measure of the number of VMs that can safely be removed. The project includes a simple illustrative scenario that adds five CPU percentage points of headroom to the predicted aggregate CPU utilization, capped at 100%:
\[
\text{planning utilization}_t = \min(100,\ \hat y_t + 5)
\]
With the AR(1) forecast reaching approximately 9.4141%, this produces a maximum illustrative planning value of approximately 14.4141%. The margin is not estimated from a formal uncertainty distribution, tied to a target SLA, or validated as sufficient for future spikes. It is a transparent sensitivity example showing how a planner might explicitly account for underprediction risk.

Under-provisioning could cause saturation or latency problems, especially for interactive workloads. Over-provisioning raises infrastructure costs and leaves resources idle. A production policy would require workload-specific CPU peaks, per-VM constraints, memory and core requirements, placement restrictions, acceptable risk, and cost data. Low fleet-average CPU does not imply that individual machines can safely be shut down or consolidated.
9. Limitations, Reproducibility, and Next Steps
Limited temporal coverage. The subset spans only about 11.1 hours. It cannot establish daily or weekly seasonality, and the model may behave differently during larger changes or longer horizons.

Artificially stable VM count. All 10,000 VMs appear at every timestamp. This could reflect subset construction rather than real-world absence of VM arrivals and departures. The result should not be generalized without checking the dataset documentation and evaluating traces with actual VM churn.

Aggregate behavior masks heterogeneity. Individual VMs vary much more than the fleet average. Aggregate CPU alone is insufficient for host-level placement or SLA-aware capacity decisions.

Uncertainty and peak risk. AR(1) reverts toward a mean and can miss temporary peaks. The five-point margin is illustrative, not a calibrated prediction interval. Validation folds overlap and do not represent independent samples.

Model comparison precision. Baseline and AR(1) RMSE should be recomputed using the same pooled definition before asserting a precise RMSE ranking. The holdout is reserved for reporting and should not be used for further model selection.

The project uses a Python package under src/cwfc/, a CLI for generating predictions.csv, and automated tests covering forecast behavior, output structure, temporal separation, and a small synthetic-data pipeline. Thirteen tests passed in the development environment. Dependencies are declared in pyproject.toml; development dependencies include pytest, matplotlib, and statsmodels. Raw datasets and virtual environments should not be committed to the repository. A final clean-checkout and smoke-dataset reproduction check remains important before submission.

Useful future work would include a longer evaluation period, non-overlapping or more varied temporal backtests, calibrated forecast intervals, workload-segment analysis, and comparison with a level-adaptive method such as exponential smoothing. Those extensions should be evaluated on new development data rather than chosen based on the existing final holdout.
10. Conclusion
This challenge demonstrates an end-to-end, time-aware forecasting workflow: data inspection, leakage-conscious target construction, transparent baselines, rolling-origin validation, AR(1) modeling, peak-error analysis, holdout evaluation, reproducible prediction generation, and a cautious capacity-planning interpretation.

AR(1) improved development-validation MAE over persistence by 6.68%, but did not improve final holdout MAE. The most important conclusion is therefore not that a more complex model always wins; it is that simple models, careful temporal evaluation, and explicit treatment of peak and operational risks provide a credible foundation for cloud workload forecasting.