<<<<<<< HEAD
# AI-Based Network Data Analytics for 5G: Forecasting Total Traffic Demand from Recent History

**Course:** Advanced Computer Networks (ACN) — End-Semester Evaluation  
**Institution:** Department of Information Technology, National Institute of Technology Karnataka (NITK), Surathkal  
**Project Team:**
- **Sanchit Sanjay Jadhav** (Roll No: 231IT059)
- **Sambhav Jetendra Purohit** (Roll No: 231IT053)
- **Kunal Laxmikant Katkar** (Roll No: 231IT033)

---

## 1. Project Objective

Fifth-generation (5G) cellular networks host three fundamentally distinct traffic service classes over shared physical infrastructure:
- **eMBB (Enhanced Mobile Broadband):** High-throughput video streaming, AR/VR, and mobile broadband characterized by strong diurnal cycles and evening peaks.
- **URLLC (Ultra-Reliable Low-Latency Communication):** Mission-critical industrial telemetry and autonomous control traffic demanding stringent latency guarantees during business hours.
- **mMTC (Massive Machine-Type Communication):** Periodic sensor telemetry from high-density IoT deployments with steady background load.

Because radio spectrum, Physical Resource Blocks (PRBs), compute, and transport backhaul are finite, reacting to traffic surges after congestion occurs violates Service Level Agreements (SLAs). 

This project implements an AI-driven network analytics pipeline aligned with 3GPP **NWDAF (Network Data Analytics Function)** and O-RAN **Non-RT RIC** specifications. The pipeline forecasts total aggregate traffic demand ($y_t = y_{eMBB} + y_{URLLC} + y_{mMTC}$) for the upcoming hour using recent historical observations, enabling proactive radio resource scaling.

---

## 2. Implementation Summary: Mid-Sem vs. End-Sem Deliverables

| Pipeline Component | Mid-Semester Status (`acn_midsem.pdf`) | End-Semester Final Status | Implementation Details |
|---|:---:|:---:|---|
| **Synthetic 5G Data Generator** | Implemented (Total = eMBB + URLLC + mMTC) | Enhanced & Standardized | 60 days (1,440 hourly samples) default, seed 42. Strict additive integrity verified. |
| **Leakage-Free Feature Engineering** | Implemented (6 lags, 2 rolling means) | Preserved & Generalized | Lags (1, 2, 3, 6, 12, 24 h), pre-shifted rolling means (6, 24 h). Zero future lookahead. |
| **Chronological Train/Test Split** | Implemented (80% train / 20% test) | Preserved | Strict sequential split without shuffling; preserves temporal causality. |
| **Naive Lag-1 Baseline** | Implemented ($y_{t+1} = y_t$) | Preserved & Benchmarked | Standard persistence benchmark for hourly time series. |
| **Random Forest Regressor** | Implemented | Calibrated to PPT Spec | 180 trees, max depth 10, min samples leaf 3, random state 42. |
| **ARIMA Forecasting Baseline** | Planned in Slide 8 | **Fully Implemented** | Box-Jenkins ARIMA(2, 1, 2) fit strictly on train split; leakage-safe 1-step test forecasts. |
| **LSTM Deep Learning Forecaster** | Planned in Slide 8 | **Fully Implemented** | PyTorch sequential LSTM network (seq_len=24); scaler fitted strictly on train split. |
| **Per-Slice Disaggregated Forecasting** | Planned in Slide 8 | **Fully Implemented** | Individual forecasting for eMBB, URLLC, mMTC, plus Bottom-Up sum validation vs total. |
| **Capacity Recommendation Panel** | Planned in Slide 8 | **Fully Implemented** | Rule-based decision panel with configurable threshold, headroom calc, & advisory disclaimer. |
| **Interactive Web UI Dashboard** | Planned in Slide 8 | **Fully Implemented** | Flask academic dashboard with Chart.js interactive charts, live slider, & pipeline runner. |
| **Automated Test Suite** | 4 unit tests | **Expanded to 12 tests** | Schema, zero-leakage, ARIMA, LSTM, per-slice, capacity rules, and web REST API tests. |
| **Real Telecom Traces / Closed-Loop RIC** | Scope limitation in Slide 8 | Documented Future Scope | Kept as explicit academic boundary; no fabricated operator claims. |

---

## 3. Discrepancy Analysis & Dataset Specification

The mid-semester presentation (`acn_midsem.pdf`, Slides 5 & 6) specifies:
- **Dataset:** 60 days of synthetic hourly traffic (1,440 hourly samples before dropping initial 24 feature-related NaN rows).
- **Random Forest Hyperparameters:** 180 trees, max depth 10, min samples leaf 3, random state 42.
- **Midsem Presentation Table I (Slide 7):** Reported RF MAE 31.52 Mbps, RMSE 39.25 Mbps, $R^2$ 0.849; Naive Baseline MAE 43.73 Mbps, RMSE 54.26 Mbps, $R^2$ 0.711 (+27.9% MAE reduction).

In the existing codebase, an earlier 90-day run (2,160 hours) with default 100-tree unconstrained Random Forest had been executed and recorded in `results/`. 

To maintain strict academic integrity and follow the prompt's instructions:
1. **The PPT's 60-day specification is adopted as the primary benchmark standard.** The codebase defaults to `num_days=60` and the exact PPT Random Forest hyperparameters (180 trees, depth 10, leaf 3).
2. **The historical 90-day run has been preserved intact** in `results/historical_midsem/`.
3. Both sets of metrics are displayed side-by-side in this README and in the Web UI dashboard for transparent viva defense.

---

## 4. End-to-End Forecasting Pipeline Architecture

```
+---------------------------------------------------------------------------------+
| 1. Synthetic 5G Multi-Slice Data Generation (src/traffic_forecasting/data.py)   |
|    - 60 days (1,440 hours) of hourly traffic; Random Seed = 42                  |
|    - eMBB (Diurnal + Evening Peak), URLLC (Workday), mMTC (Periodic Telemetry)  |
|    - Exact relation: y_total = y_embb + y_urllc + y_mmtc                        |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| 2. Leakage-Safe Feature Engineering (src/traffic_forecasting/features.py)       |
|    - Lags: lag_1, lag_2, lag_3, lag_6, lag_12, lag_24                          |
|    - Shifted Rolling Means: rolling_mean_6, rolling_mean_24 (computed on t-1)   |
|    - Initial 24 NaN rows dropped -> 1,416 clean feature samples                 |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| 3. Chronological Train / Test Split (src/traffic_forecasting/model.py)          |
|    - Chronological 80% Train: 1,132 samples (2026-01-02 00:00 to 2026-02-18)    |
|    - Chronological 20% Held-Out Test: 284 samples (2026-02-18 to 2026-03-01)    |
|    - Zero shuffling: guarantees temporal causality                              |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| 4. Multi-Model Forecasting Benchmark (model.py, arima_model.py, lstm_model.py) |
|    - Naive Lag-1 Persistence Baseline (y_hat_{t+1} = y_t)                       |
|    - ARIMA(2, 1, 2) Classical Box-Jenkins Forecaster                            |
|    - PyTorch Sequential LSTM Neural Network (seq_len=24)                        |
|    - Random Forest Regressor (180 trees, depth 10, min leaf 3)                  |
|    - Per-Slice Forecasters (eMBB, URLLC, mMTC) & Bottom-Up Summation           |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| 5. Proactive Capacity Decision Rule (src/traffic_forecasting/capacity.py)       |
|    - Evaluates next-hour forecast against configurable threshold (default 700M) |
|    - Emits "ADD CAPACITY" (scale PRBs) or "NOMINAL" (maintain profile)          |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| 6. Academic Web UI Dashboard & REST API (src/traffic_forecasting/app.py)        |
|    - Interactive Chart.js graphs, capacity slider, live pipeline execution      |
+---------------------------------------------------------------------------------+
```

---

## 5. Experimental Evaluation Results

All models were evaluated on the **identical unseen chronological test period** (284 hours).

### Multi-Model Benchmark (Total Traffic Demand)

| Model Architecture | Model Family | MAE (Mbps) | RMSE (Mbps) | $R^2$ Score | Relative MAE Reduction |
|---|---|:---:|:---:|:---:|:---:|
| **Naive Baseline (Lag-1)** | Persistence Rule | 38.68 | 48.61 | 0.8789 | 0.00% (Baseline) |
| **ARIMA(2, 1, 2)** | Classical Time Series | 31.04 | 38.80 | 0.9228 | **+19.75%** |
| **LSTM Sequential Forecaster** | Deep Recurrent Network | 22.34 | 27.95 | 0.9600 | **+42.24%** |
| **Random Forest Regressor (PPT Spec)** | Tree Bagging Ensemble | **20.33** | **25.64** | **0.9663** | **+47.44%** |
| **Bottom-Up Slice Summation** | Disaggregated Slice Sum | 21.01 | 26.95 | 0.9628 | **+45.68%** |

### Per-Slice Disaggregated Forecasting Benchmark

| Traffic Slice | Service Profile | MAE (Mbps) | RMSE (Mbps) | $R^2$ Score | Next-Hour Forecast (Mbps) |
|---|---|:---:|:---:|:---:|:---:|
| **eMBB** | High-throughput video streaming | 20.43 | 25.80 | 0.9563 | 131.91 |
| **URLLC** | Low-latency mission-critical control | 4.63 | 5.73 | 0.9471 | 23.99 |
| **mMTC** | Massive IoT periodic telemetry | 7.15 | 9.04 | 0.5707 | 81.79 |
| **BOTTOM-UP SUM** | Sum of eMBB + URLLC + mMTC | **21.01** | **26.95** | **0.9628** | **237.69** |

*Key Technical Insight:* The bottom-up summation of the three individual slice forecasts achieves an MAE of 21.01 Mbps and $R^2$ of 0.9628, closely tracking the direct aggregate Random Forest forecast (20.33 Mbps, $R^2$ 0.9663). This confirms that slice-level resource orchestration can be performed independently without losing aggregate macro-level predictive accuracy.

---

## 6. Proactive Capacity Recommendation Logic

Planned in PPT Slide 8, the capacity recommendation module (`src/traffic_forecasting/capacity.py`) implements an automated decision-support rule:

$$\text{Decision} = \begin{cases} 
\text{"ADD\_CAPACITY"}, & \text{if } \hat{y}_{t+1} \ge \text{Threshold}_{\text{capacity}} \\ 
\text{"NOMINAL"}, & \text{if } \hat{y}_{t+1} < \text{Threshold}_{\text{capacity}} 
\end{cases}$$

- **Default Threshold:** 700.0 Mbps (configurable via the Web UI slider from 450 to 850 Mbps).
- **High-Demand Action:** Emits an advisory hint recommending proactive scale-out of Physical Resource Blocks (PRBs), carrier aggregation activation, or pre-warming virtualized DU/CU instances.
- **Normal Action:** Emits a neutral status indicating sufficient headroom and advising maintenance of the standard base-station power profile.
- **Academic Disclaimer:** This is an analytical decision-support rule designed for 5G Non-RT RIC / NWDAF simulation. It does not actuate physical network equipment or modify active radio resource controllers.

---

## 7. Web UI Dashboard

The project includes a functional, responsive web interface built with Flask and Chart.js:

- **Executive Summary:** Key performance indicators, next-hour demand forecast, and capacity badge.
- **Pipeline Metadata:** Visual 5-step diagram and dataset split status.
- **Model Comparison Table:** Interactive benchmarking table with live MAE improvement calculations.
- **Interactive Visualizations:**
  - *Test Set Actual vs Predicted Demand:* Toggleable overlays for RF, ARIMA, LSTM, and Naive Baseline.
  - *Per-Slice Forecast Breakdown:* Disaggregated curves for eMBB, URLLC, mMTC, and Bottom-Up Sum.
  - *Historical Traffic Explorer:* Multi-slice diurnal cycles over 7-day windows.
  - *Feature Importance Inspection:* Horizontal bar chart ranking engineered lag and rolling features.
- **Interactive Capacity Panel:** Live slider adjusting threshold and updating recommendation status instantaneously.
- **Pipeline Runner Control:** Allows on-demand execution of simulations with custom day lengths and seeds, updating results without page reloads.

---

## 8. Quickstart & Installation Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.14)
- Virtual environment (`venv`)

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies & Package
```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run Automated Unit & Integration Tests
```bash
pytest -v
```
All 12 unit and integration tests will execute and pass:
```
tests/test_pipeline.py::test_data_generator_columns_and_no_nan PASSED
tests/test_pipeline.py::test_aggregate_traffic_relationship PASSED
tests/test_pipeline.py::test_feature_builder_expected_columns_and_no_nan PASSED
tests/test_pipeline.py::test_no_leakage_lag_1 PASSED
tests/test_pipeline.py::test_no_future_leakage_rolling_means PASSED
tests/test_pipeline.py::test_chronological_split_strict_ordering PASSED
tests/test_pipeline.py::test_model_training_and_evaluation_pipeline PASSED
tests/test_pipeline.py::test_arima_baseline_forecasting PASSED
tests/test_pipeline.py::test_lstm_baseline_forecasting PASSED
tests/test_pipeline.py::test_per_slice_forecasting_and_summation PASSED
tests/test_pipeline.py::test_capacity_recommendation_rules PASSED
tests/test_pipeline.py::test_flask_web_ui_endpoints PASSED
```

### 4. Execute the Complete Forecasting Pipeline (CLI)
```bash
python scripts/run_experiment.py --days 60 --seed 42
```
This runs the full 6-step pipeline and writes all generated datasets, CSV metrics, and high-resolution plots to `results/`.

### 5. Launch the Web UI Dashboard
```bash
python scripts/run_app.py --port 5000
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 9. Docker Execution (Optional)

To run inside a self-contained Docker container:

```bash
# Build the Docker image
docker build -t 5g-traffic-forecasting .

# Run the forecasting benchmark
docker run --rm -v $(pwd)/results:/app/results 5g-traffic-forecasting

# Run the Web UI dashboard
docker run --rm -p 5000:5000 5g-traffic-forecasting python scripts/run_app.py --host 0.0.0.0 --port 5000
```

---

## 10. Project Directory Structure

```
.
├── Dockerfile                             # Containerized build specification
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Python dependencies (statsmodels, torch, flask)
├── setup.py                               # Package installation configuration
├── docs/
│   └── MIDSEM_NOTES.md                    # Mid-sem demo script, concepts & viva Q&A
├── results/                               # Generated datasets, metrics, and plots
│   ├── feature_importance.csv             # Random Forest MDI scores
│   ├── feature_importance.png             # Feature importance visualization
│   ├── forecast_actual_vs_predicted.png   # Multi-model test comparison plot
│   ├── forecast_metrics.csv               # Unified benchmark metrics table
│   ├── forecast_predictions.csv           # Model test predictions (RF, ARIMA, LSTM, Naive)
│   ├── historical_midsem/                 # Preserved 90-day historical execution run
│   ├── slice_forecasts.png                # Per-slice and bottom-up sum visual curves
│   ├── slice_metrics.csv                  # eMBB, URLLC, mMTC evaluation metrics
│   ├── slice_predictions.csv              # Per-slice test predictions
│   └── synthetic_5g_traffic.csv           # 60-day hourly synthetic 5G dataset
├── scripts/
│   ├── run_app.py                         # Web UI dashboard server launcher
│   └── run_experiment.py                  # CLI pipeline runner
├── src/
│   └── traffic_forecasting/
│       ├── __init__.py                    # Public API exports
│       ├── app.py                         # Flask web application & REST API
│       ├── arima_model.py                 # ARIMA(2, 1, 2) baseline implementation
│       ├── capacity.py                    # Rule-based capacity recommendation logic
│       ├── data.py                        # Synthetic 5G multi-slice data generator
│       ├── experiment.py                  # End-to-end benchmark orchestration & plotting
│       ├── features.py                    # Zero-leakage lag and rolling features
│       ├── lstm_model.py                  # PyTorch LSTM deep learning forecaster
│       ├── model.py                       # Random Forest, persistence baseline, metrics
│       ├── slice_forecasting.py           # Per-slice modeling & bottom-up aggregation
│       └── templates/
│           └── dashboard.html             # Academic web dashboard template
└── tests/
    └── test_pipeline.py                   # Pytest test suite (12 unit & integration tests)
```

---

## 11. Limitations & Academic Scope Boundaries

1. **Synthetic Traffic Traces:** While the synthetic generator realistically models diurnal variation, weekend entertainment boosts for eMBB, business-hour boosts for URLLC, and 12-hour telemetry bursts for mMTC, real-world cellular deployments exhibit spatial handovers, weather attenuation, and flash crowds. Evaluation on real public operator traces (such as Telecom Italia Big Data or CRAWDAD) remains planned future work.
2. **Advisory Rule Panel vs. Closed-Loop Controller:** The capacity recommendation panel implements threshold-based heuristics to suggest PRB scaling. It is an analytical decision-support tool (NWDAF / Non-RT RIC simulation) and does not directly actuate physical gNodeB radio equipment or interface with the O-RAN E2 interface.

---

## 12. References Cited

1. **[1] 3GPP**, *"System architecture for the 5G System (5GS),"* 3GPP Technical Specification TS 23.501, Rel. 18, 2023.
2. **[2] X. Foukas, G. Patounas, A. Elmokashfi, and M. K. Marina**, *"Network Slicing in 5G: Survey and Challenges,"* IEEE Communications Magazine, vol. 55, no. 5, pp. 94–100, May 2017.
3. **[3] L. Breiman**, *"Random Forests,"* Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.
4. **[4] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung**, *Time Series Analysis: Forecasting and Control*, 5th ed., Hoboken, NJ: John Wiley & Sons, 2015.
=======
# ACN-PROJECT
>>>>>>> origin/main
