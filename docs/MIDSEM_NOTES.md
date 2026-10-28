# Mid-Semester Evaluation Notes & Viva Preparation Guide

**Project Title:** AI-Based Network Data Analytics for 5G  
**Course:** Advanced Computer Networks (ACN)  
**Team Members:**  
- Sanchit Sanjay Jadhav (231IT059)
- Sambhav Jetendra Purohit (231IT053)
- Kunal Laxmikant Katkar (231IT033)

---

## 1. Five-Minute Demo Script

| Time | Presenter | Action / Screen | What to Say |
|---|---|---|---|
| **0:00 - 1:00** | **Speaker 1 (Sanchit)** | Open Terminal & Project Tree | *"Good morning/afternoon Professors. Our project addresses proactive network resource management in 5G cellular networks using AI-based traffic analytics. 5G carries diverse service slices—eMBB for high-throughput video, URLLC for mission-critical low-latency tasks, and mMTC for IoT telemetry. By predicting total next-hour traffic demand, network orchestrators can dynamically allocate PRBs (Physical Resource Blocks) and avoid SLA violations."* |
| **1:00 - 2:15** | **Speaker 2 (Sambhav)** | Run `pytest -v` | *"We designed a strictly leakage-free pipeline in 5 steps: synthetic traffic synthesis across the 3 slices, temporal lag and rolling features, chronological train/test split, Random Forest modeling, and comparative evaluation against a naive baseline. Running `pytest -v`, all 4 unit tests pass, confirming schema validity and mathematical verification that no future data leaks into the feature set."* |
| **2:15 - 3:45** | **Speaker 3 (Kunal)** | Run `python scripts/run_experiment.py` | *"Executing the experiment, we train on the first 80% of data chronologically and test on the unseen final 20%. Our Random Forest model achieves an MAE of 19.79 Mbps, RMSE of 24.76 Mbps, and R² of 0.9688. Compared to the Naive Persistence baseline (MAE 39.21 Mbps), our model achieves a **49.53% reduction in error**."* |
| **3:45 - 5:00** | **Team (All)** | Open `results/*.png` plots | *"Looking at `forecast_actual_vs_predicted.png`, our model accurately tracks diurnal peaks and troughs without lag delay. In `feature_importance.png`, `lag_24` is the most critical feature (importance ~0.93), capturing the 24-hour periodic daily cycle of cellular usage. For final submission, we will extend this with deep learning models (LSTM) and per-slice forecasting."* |

---

## 2. Core Concepts Explained Simply

### 1. Lag Features
* **What it is:** Using traffic values from previous time steps ($t-1, t-2, \dots, t-24$ hours) as input variables to predict traffic at time $t$.
* **Why it matters:** Traffic exhibits strong autocorrelation. What happened 1 hour ago ($lag_1$) and 24 hours ago ($lag_{24}$) provides strong predictive signals.

### 2. Rolling Mean
* **What it is:** The moving average of traffic over a fixed historical window (e.g., 6 hours or 24 hours).
* **Why it matters:** Smooths out short-term high-frequency noise and captures medium-term upward or downward demand trends.

### 3. Data Leakage (and How We Prevented It)
* **What it is:** When information from the future (or the target test set) inadvertently leaks into the training feature set, creating artificially inflated accuracy that fails in production.
* **How we prevented it:**
  - `lag_k` strictly uses values shifted by $k \ge 1$.
  - `rolling_mean_w` is computed on `shift(1)` of the series, ensuring at time $t$ we only average historical points from $t-w$ up to $t-1$.

### 4. Why a Time-Aware Split (No Shuffling)
* **What it is:** We allocate the first 80% chronologically for training and the remaining 20% for testing (`shuffle=False`).
* **Why:** In real-world cellular deployments, you cannot train on tomorrow's traffic to predict today's traffic. Standard random k-fold shuffling destroys the temporal order and leaks future data.

### 5. Why Random Forest Regressor
* **What it is:** An ensemble of decision trees with bootstrap aggregation (bagging) and feature subsetting.
* **Why:**
  - Captures complex non-linear diurnal patterns and interactions between different lag intervals.
  - Robust against outliers and noise without requiring extensive normalization.
  - Provides interpretable feature importance scores.

### 6. Why Compare with a Naive Baseline
* **What it is:** The naive persistence baseline predicts $\hat{y}_t = y_{t-1}$ (the traffic next hour will equal the traffic this hour).
* **Why:** In time-series forecasting, persistence is surprisingly strong due to autocorrelation. A machine learning model is only justified if it measurably outperforms this zero-cost baseline.

### 7. Evaluation Metrics: MAE, RMSE, R²
* **MAE (Mean Absolute Error):** Average magnitude of forecasting errors in Mbps: $\frac{1}{N}\sum |y_i - \hat{y}_i|$. Easy to interpret physically.
* **RMSE (Root Mean Squared Error):** Square root of the average squared errors: $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$. Penalizes large outlier errors more heavily.
* **$R^2$ (Coefficient of Determination):** Proportion of variance in 5G traffic explained by the model ($1.0$ is perfect; $>0.90$ is very strong).

---

## 3. Top 8 Likely Examiner Questions & Answers

### Q1: Why did you forecast total 5G traffic instead of individual slices first?
> **Answer:** In cellular base stations (gNodeB), physical transmission resources (PRBs) and radio frequency channels are drawn from a shared pool. Forecasting aggregate total demand provides the core baseline needed for macro-level spectrum allocation and power grid management, while slice-level isolation is planned for the final phase.

### Q2: Why is `lag_24` by far the most important feature?
> **Answer:** Cellular traffic follows a strong 24-hour diurnal cycle dictated by human routine—morning wakeups, business hours, and peak evening streaming (19:00–22:00). Traffic at 8 PM today is mathematically and behaviorally most similar to traffic at 8 PM yesterday ($lag_{24}$).

### Q3: What happens if a flash crowd or sudden sporting event occurs?
> **Answer:** Historical lag features will capture the initial rise within 1 hour ($lag_1$), but unexpected burst spikes beyond historical distribution cause temporary residuals. In a production 5G network, analytics forecasts provide proactive planning, while closed-loop reactive schedulers handle real-time millisecond bursts.

### Q4: Why did you not use a standard random train/test split?
> **Answer:** Time series data has temporal dependency. If we randomly shuffle samples, the model trains on data points adjacent in time to test points, which leaks future autocorrelation and yields fake high accuracy. A chronological cutoff simulates real deployment.

### Q5: How did you ensure your rolling average feature doesn't cheat?
> **Answer:** Before calculating any 6-hour or 24-hour rolling window, we explicitly shift the series by 1 time step (`df['total_mbps'].shift(1)`). Therefore, when predicting traffic at time $t$, the rolling window strictly averages observations from $t-w$ up to $t-1$.

### Q6: How does this model help 5G Network Slicing and O-RAN?
> **Answer:** In O-RAN architecture, this model acts as an **rApp** in the Non-Real-Time RAN Intelligent Controller (Non-RT RIC) or an **xApp** in the Near-RT RIC. It informs SLA-aware dynamic slice scaling and energy-saving sleep modes for idle base station carriers.

### Q7: Why did you select Random Forest over Linear Regression or a Deep Neural Network for midsem?
> **Answer:** Linear Regression cannot capture the non-linear inflection points of peak-to-trough traffic transitions. Deep neural networks (LSTM/GRU) require heavier tuning and more compute. Random Forest provides high non-linear capacity, interpretability, and robust generalization with zero risk of vanishing gradients.

### Q8: What are your planned deliverables for the final semester evaluation?
> **Answer:** 
> 1. Comparing Random Forest against sequential deep learning architectures (LSTM, GRU, Temporal Convolutional Networks).
> 2. Per-slice multi-output forecasting ($eMBB, URLLC, mMTC$).
> 3. Validation on real public 5G base station datasets (e.g., Telecom Italia / CRAWDAD).
> 4. Closed-loop dynamic resource block allocation simulation.
