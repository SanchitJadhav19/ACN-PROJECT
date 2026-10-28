# End-Semester Evaluation Viva Preparation Guide & Defense Notes

**Project Title:** AI-Based Network Data Analytics for 5G — Forecasting Total Traffic Demand from Recent History  
**Course:** Advanced Computer Networks (ACN)  
**Team Members:**  
- Sanchit Sanjay Jadhav (231IT059)
- Sambhav Jetendra Purohit (231IT053)
- Kunal Laxmikant Katkar (231IT033)
**Institution:** Department of Information Technology, NITK Surathkal

---

## 1. Five-Minute Final Demo Script for Examiners

| Time | Presenter | Screen / Action | Speaking Script |
|---|---|---|---|
| **0:00 - 1:00** | **Speaker 1 (Sanchit)** | Open Web UI at `http://127.0.0.1:5000` | *"Respected Examiners, our project addresses proactive resource allocation in 5G networks using AI network data analytics. In our mid-semester review, we demonstrated a 5-step Random Forest pipeline for total traffic demand. For this end-semester evaluation, we have fulfilled all planned work from our presentation: implementing an interactive Web UI dashboard, adding classical Box-Jenkins ARIMA and deep learning PyTorch LSTM baselines, building disaggregated per-slice forecasting for eMBB, URLLC, and mMTC, and connecting a rule-based proactive capacity recommendation panel."* |
| **1:00 - 2:15** | **Speaker 2 (Sambhav)** | Point to Benchmark Table in Section 2 of UI | *"Looking at our benchmark table on the held-out 20% chronological test block: Our Naive persistence baseline achieves an MAE of 38.68 Mbps. The classical ARIMA(2,1,2) model reduces MAE to 31.04 Mbps (+19.75% improvement). The PyTorch sequential LSTM further lowers MAE to 22.34 Mbps (+42.24% improvement). Our PPT reference Random Forest model achieves the highest accuracy with an MAE of 20.33 Mbps, RMSE of 25.64 Mbps, and R² of 0.9663 (+47.44% error reduction). In all models, preprocessing, scaling, and fitting strictly observe the temporal boundary with zero future data leakage."* |
| **2:15 - 3:30** | **Speaker 3 (Kunal)** | Scroll to Section 3B & Section 4 (Per-Slice Forecasts) | *"In 5G network slicing, total traffic is the sum of eMBB, URLLC, and mMTC. We extended our pipeline to forecast each slice independently. eMBB achieves an MAE of 20.43 Mbps; URLLC achieves 4.63 Mbps; and mMTC achieves 7.15 Mbps. Crucially, when we sum the three slice forecasts bottom-up, the resulting aggregate achieves an MAE of 21.01 Mbps and R² of 0.9628, confirming that individual slice orchestrators can predict their own demand while faithfully recovering total macro-cell load."* |
| **3:30 - 4:30** | **Speaker 1 (Sanchit)** | Interact with Capacity Slider in Section 5 | *"Planned in PPT Slide 8, here is our rule-based proactive capacity panel. The model forecasts next-hour demand as 250.41 Mbps. When our operator threshold is 700 Mbps, the status is 'NOMINAL'. If we slide the threshold down to 200 Mbps to simulate peak network congestion, the system dynamically alerts 'ADD CAPACITY', recommending proactive PRB scale-out. As noted in our disclaimer, this is an advisory NWDAF decision-support tool, not an active radio controller."* |
| **4:30 - 5:00** | **Team (All)** | Show terminal running `pytest -v` | *"All 12 unit and integration tests pass in under 4 seconds, verifying mathematical additive consistency, zero lookahead leakage, model correctness, and REST API integration. We are ready for your questions."* |

---

## 2. Top 10 Likely Examiner Questions & Detailed Answers

### Q1: Why did you compare Random Forest against both ARIMA and LSTM?
> **Answer:** 
> In network traffic forecasting literature:
> - **ARIMA (Box-Jenkins, 1970):** Represents the classical linear benchmark for stationary and difference-stationary time series. It models autocorrelations and moving-average residuals.
> - **LSTM (Hochreiter & Schmidhuber, 1997):** Represents modern deep recurrent sequence modeling. It uses input, forget, and output gates to retain temporal memory across historical sequences without vanishing gradients.
> - **Random Forest (Breiman, 2001):** Represents non-linear tree bagging ensembles over tabular lag features. 
> Comparing all three establishes where tabular lag regression stands relative to classical linear filters and recurrent deep networks.

### Q2: Why does Random Forest perform slightly better than LSTM on this dataset?
> **Answer:** 
> Cellular traffic exhibits a strong 24-hour diurnal cycle where historical values from exactly 12 hours (`lag_12`) and 24 hours (`lag_24`) ago are decisive predictive features. In Random Forest, decision trees can split directly on `lag_24` at root nodes (Gini importance ~0.93). In contrast, an LSTM must propagate information step-by-step through 24 sequential recurrence unrollings, which dilutes the direct impact of the 24th lag through intermediate cell states. Furthermore, tree ensembles are less sensitive to hyperparameter tuning and noise on moderate sample sizes (1,440 hours).

### Q3: How did you ensure that ARIMA and LSTM did not suffer from lookahead data leakage?
> **Answer:** 
> We enforced strict temporal boundaries:
> 1. **ARIMA:** The ARIMA(2, 1, 2) parameters were estimated *exclusively* on the first 80% chronological training samples (`y_train`). During test evaluation, we used Kalman filtering (`model_fit.apply(full_series, refit=False)`), where the 1-step forecast for observation $t$ uses the fixed training parameters applied only to observations $0$ to $t-1$. No test ground truth was used for parameter optimization.
> 2. **LSTM:** The `StandardScaler` was fitted *strictly* on `y_train` (`fit_transform`) and applied to `y_test` via `transform`. For test sample $t$, the input sequence was strictly drawn from the preceding 24 hours $[t-24 : t]$.
> 3. **Rolling Features:** Rolling averages were computed on `shift(1)`, guaranteeing time step $t$ only averages values up to $t-1$.

### Q4: Explain the difference between "Direct Aggregate Forecasting" and "Bottom-Up Slice Summation".
> **Answer:** 
> In 5G networks:
> - **Direct Aggregate Forecasting:** We train one model directly on total traffic ($y_{total}$). It optimizes loss directly against macro demand but provides no visibility into which slice is causing a traffic surge.
> - **Bottom-Up Slice Summation:** We train three distinct models—one for eMBB, one for URLLC, and one for mMTC—and compute total forecast as $\hat{y}_{total} = \hat{y}_{embb} + \hat{y}_{urllc} + \hat{y}_{mmtc}$.
> Our results show that Bottom-Up Summation achieves an MAE of 21.01 Mbps and $R^2$ of 0.9628, remarkably close to Direct Aggregate's 20.33 Mbps and $R^2$ of 0.9663. This proves that slice-level orchestration does not compromise macro-level network visibility.

### Q5: Why is mMTC $R^2$ lower (~0.57) compared to eMBB (~0.96) and URLLC (~0.95)?
> **Answer:** 
> In 5G networks, mMTC traffic consists of sporadic IoT sensor updates and smart meter telemetry with a lower mean magnitude (80 Mbps) and higher Gaussian reporting noise relative to its signal amplitude. Because the signal-to-noise ratio (SNR) is lower, regression variance explained ($R^2$) is naturally lower (~0.57), even though its absolute error (MAE 7.15 Mbps) is small. In contrast, eMBB has high volume (400–600 Mbps) with dominant sinusoidal human diurnal rhythms, yielding high $R^2$ (>0.95).

### Q6: Why did the PPT mention 60 days while the earlier codebase had 90 days?
> **Answer:** 
> During mid-semester development, the team tested an extended 90-day simulation in the codebase. However, the approved mid-semester presentation (`acn_midsem.pdf` Slides 5 and 6) explicitly documented the 60-day specification (1,440 hours) with 180 trees, max depth 10, and min samples leaf 3. To remain faithful to the presentation specification, we restored the 60-day run as our primary benchmark reference, while preserving the 90-day historical results in `results/historical_midsem/` for complete transparency.

### Q7: Where does this project fit into 3GPP and O-RAN architecture?
> **Answer:** 
> - In **3GPP Rel. 16/17/18**, this capability maps to the **NWDAF (Network Data Analytics Function)** under TS 23.501, which collects NF load metrics and provides predictive analytics to the PCF (Policy Control Function) and NSSF (Network Slice Selection Function).
> - In **O-RAN**, this model acts as an **rApp** in the Non-Real-Time RIC (RAN Intelligent Controller) operating on control loops $> 1$ second, or an **xApp** in the Near-RT RIC (10 ms – 1 s). It drives proactive Physical Resource Block (PRB) allocation and energy-saving carrier sleep modes.

### Q8: What does the rule-based capacity recommendation panel actually do?
> **Answer:** 
> Planned in PPT Slide 8, it implements a threshold comparator: if predicted next-hour traffic $\hat{y}_{t+1}$ meets or exceeds the operator's configured threshold (default 700 Mbps), it triggers an "ADD CAPACITY" advisory hint recommending that the network orchestrator expand PRB allocation or engage carrier aggregation. When demand is below threshold, it emits "NOMINAL". As prominently stated in our UI, this is an analytical advisory tool for decision support, not an active physical hardware controller.

### Q9: How is the Web UI implemented, and does it use mock data?
> **Answer:** 
> The Web UI is built with Python Flask and vanilla JavaScript with Chart.js. It does **NOT** use mock or fabricated data:
> 1. All metrics and charts are read directly from `results/forecast_metrics.csv`, `forecast_predictions.csv`, `slice_predictions.csv`, and `feature_importance.csv`.
> 2. The UI includes an interactive REST API (`/api/run-pipeline`) that allows the user to re-execute the Python pipeline on demand with fresh parameters and updates the interface in real time.

### Q10: What are the main limitations of this project?
> **Answer:** 
> 1. **Synthetic vs. Real Traces:** Traffic is generated with realistic diurnal, weekend, and noise patterns, but lacks spatial cell-to-cell user handover dynamics and unexpected physical events (stadium flash crowds, fiber cuts).
> 2. **Open-Loop Decision Support:** The capacity recommendation module provides guidance rather than an active closed-loop controller connected to a live gNodeB via the O-RAN E2 interface.

