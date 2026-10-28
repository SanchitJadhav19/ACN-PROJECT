"""Experiment orchestration and visualization module for 5G traffic analytics.

Runs the complete enhanced multi-model pipeline:
1. Data generation (synthetic 5G traffic across eMBB, URLLC, mMTC for 60 days reference)
2. Feature engineering (lag features + rolling means without future data leakage)
3. Chronological 80/20 train/test split
4. Multi-model forecasting:
   - Naive Persistence Baseline (Lag-1)
   - Random Forest Regressor (180 trees, depth 10, min leaf 3, seed 42)
   - ARIMA Baseline (2, 1, 2)
   - PyTorch LSTM Sequential Forecaster (seq_len 24)
   - Per-Slice Forecasters (eMBB, URLLC, mMTC) & Bottom-up summation
5. Next-hour demand forecasting & rule-based capacity recommendation
6. Comparative evaluation (MAE, RMSE, R2, % MAE improvement)
7. Generation of high-resolution publication-quality plots and CSV outputs
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic_forecasting.data import generate_synthetic_5g_traffic
from traffic_forecasting.features import create_lag_and_rolling_features
from traffic_forecasting.model import (
    calculate_metrics,
    evaluate_forecasts,
    evaluate_multiple_models,
    extract_feature_importances,
    split_time_series_data,
    train_random_forest,
)
from traffic_forecasting.arima_model import train_and_evaluate_arima
from traffic_forecasting.lstm_model import train_and_evaluate_lstm
from traffic_forecasting.slice_forecasting import train_and_evaluate_slices
from traffic_forecasting.capacity import evaluate_capacity_recommendation


def plot_actual_vs_predicted(
    timestamps: pd.Series,
    y_actual: pd.Series,
    y_pred_rf: np.ndarray,
    y_pred_baseline: pd.Series | np.ndarray,
    y_pred_arima: np.ndarray | None = None,
    y_pred_lstm: np.ndarray | None = None,
    output_path: str = "results/forecast_actual_vs_predicted.png",
    sample_hours: int = 168,
) -> None:
    """Generate high-resolution multi-model comparison plot of actual vs predicted 5G traffic demand.

    Args:
        timestamps: Test set timestamps.
        y_actual: Actual total traffic demand.
        y_pred_rf: Random Forest predictions.
        y_pred_baseline: Naive baseline predictions.
        y_pred_arima: Optional ARIMA predictions.
        y_pred_lstm: Optional LSTM predictions.
        output_path: Destination image path.
        sample_hours: Number of hours to display in zoom view (default: 168 = 1 week).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    limit = min(sample_hours, len(timestamps))
    ts_subset = pd.to_datetime(timestamps.iloc[:limit])
    actual_sub = y_actual.iloc[:limit].to_numpy() if hasattr(y_actual, "iloc") else y_actual[:limit]
    rf_sub = y_pred_rf[:limit]
    baseline_sub = (
        y_pred_baseline.iloc[:limit].to_numpy()
        if hasattr(y_pred_baseline, "iloc")
        else y_pred_baseline[:limit]
    )

    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)

    # Plot actual ground truth
    ax.plot(
        ts_subset,
        actual_sub,
        label="Actual 5G Demand",
        color="#0F172A",
        linewidth=2.4,
        alpha=0.95,
        zorder=4,
    )

    # Plot Random Forest (Primary PPT model)
    ax.plot(
        ts_subset,
        rf_sub,
        label="Random Forest (180 trees)",
        color="#2563EB",
        linewidth=1.9,
        linestyle="-",
        alpha=0.9,
        zorder=3,
    )

    # Plot ARIMA if available
    if y_pred_arima is not None:
        ax.plot(
            ts_subset,
            y_pred_arima[:limit],
            label="ARIMA(2, 1, 2) Baseline",
            color="#059669",
            linewidth=1.5,
            linestyle="-.",
            alpha=0.85,
            zorder=2,
        )

    # Plot LSTM if available
    if y_pred_lstm is not None:
        ax.plot(
            ts_subset,
            y_pred_lstm[:limit],
            label="LSTM Sequential Baseline",
            color="#9333EA",
            linewidth=1.6,
            linestyle=":",
            alpha=0.85,
            zorder=2,
        )

    # Plot Naive Baseline
    ax.plot(
        ts_subset,
        baseline_sub,
        label="Naive Baseline (Lag-1)",
        color="#F59E0B",
        linewidth=1.3,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )

    # Styling
    ax.set_title(
        "5G Total Traffic Demand Forecasting: Multi-Model Benchmark (Test Period — 7 Days Zoom)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Timestamp (Hourly)", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Traffic Demand (Mbps)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.55, color="#94A3B8")
    ax.legend(
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=9.5,
        loc="upper right",
        ncol=2,
    )
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(
    fi_df: pd.DataFrame,
    output_path: str = "results/feature_importance.png",
) -> None:
    """Generate bar chart showing feature importances for Random Forest.

    Args:
        fi_df: DataFrame with columns 'feature' and 'importance'.
        output_path: Destination image path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    plot_df = fi_df.sort_values(by="importance", ascending=True)

    bars = ax.barh(
        plot_df["feature"],
        plot_df["importance"],
        color="#3B82F6",
        edgecolor="#1D4ED8",
        height=0.62,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#1E293B",
            fontweight="semibold",
        )

    ax.set_title(
        "Random Forest Feature Importance for 5G Traffic Forecasting (PPT Specs: 180 Trees)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Relative Importance Score (MDI / Gini Impurity Reduction)", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Engineered Feature", fontsize=10.5, labelpad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#94A3B8")
    ax.set_xlim(0, max(plot_df["importance"]) * 1.18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_slice_forecasts(
    timestamps: pd.Series,
    slice_actuals: dict[str, np.ndarray],
    slice_predictions: dict[str, np.ndarray],
    bottom_up_preds: np.ndarray,
    total_actual: np.ndarray,
    output_path: str = "results/slice_forecasts.png",
    sample_hours: int = 168,
) -> None:
    """Generate 4-panel visual breakdown of per-slice and aggregate forecasts.

    Args:
        timestamps: Test set timestamps.
        slice_actuals: Dict of actual arrays per slice.
        slice_predictions: Dict of predicted arrays per slice.
        bottom_up_preds: Sum of slice predictions.
        total_actual: Actual total traffic demand.
        output_path: Destination image path.
        sample_hours: Hours to display in zoom view.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    limit = min(sample_hours, len(timestamps))
    ts_sub = pd.to_datetime(timestamps.iloc[:limit])

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, dpi=300)

    slice_configs = [
        ("embb", "eMBB (Enhanced Mobile Broadband)", "#2563EB", axes[0]),
        ("urllc", "URLLC (Ultra-Reliable Low-Latency)", "#D97706", axes[1]),
        ("mmtc", "mMTC (Massive Machine-Type Comm.)", "#059669", axes[2]),
    ]

    for key, title, color, ax in slice_configs:
        act = slice_actuals[key][:limit]
        pred = slice_predictions[key][:limit]
        ax.plot(ts_sub, act, label=f"Actual {key.upper()}", color="#0F172A", linewidth=2.0)
        ax.plot(ts_sub, pred, label=f"Predicted {key.upper()}", color=color, linewidth=1.8, linestyle="--")
        ax.set_ylabel("Demand (Mbps)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    # Panel 4: Bottom-up summation vs actual total
    ax_tot = axes[3]
    ax_tot.plot(ts_sub, total_actual[:limit], label="Actual Total Demand (y_total)", color="#0F172A", linewidth=2.2)
    ax_tot.plot(
        ts_sub,
        bottom_up_preds[:limit],
        label="Bottom-Up Slices Sum (y_embb + y_urllc + y_mmtc)",
        color="#7C3AED",
        linewidth=1.9,
        linestyle="-.",
    )
    ax_tot.set_ylabel("Total (Mbps)", fontsize=10)
    ax_tot.set_title("Total 5G Demand: Slices Summation vs Actual Total", fontsize=11, fontweight="bold", loc="left")
    ax_tot.set_xlabel("Timestamp (Hourly)", fontsize=11)
    ax_tot.grid(True, linestyle=":", alpha=0.5)
    ax_tot.legend(loc="upper right", fontsize=9, frameon=True)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run_full_experiment(
    num_days: int = 60,
    random_seed: int = 42,
    results_dir: str = "results",
) -> dict[str, object]:
    """Execute the complete 5G traffic forecasting pipeline end-to-end.

    Defaults to num_days=60 matching the mid-semester presentation reference.

    Args:
        num_days: Number of days for synthetic traffic generation (default: 60 days = 1440 hours).
        random_seed: Seed for reproducibility.
        results_dir: Directory where all metrics, CSVs, and plots are stored.

    Returns:
        dict: Complete results dictionary containing all dataframes, models, and predictions.
    """
    os.makedirs(results_dir, exist_ok=True)
    raw_csv_path = os.path.join(results_dir, "synthetic_5g_traffic.csv")
    metrics_csv_path = os.path.join(results_dir, "forecast_metrics.csv")
    predictions_csv_path = os.path.join(results_dir, "forecast_predictions.csv")
    actual_vs_pred_plot_path = os.path.join(results_dir, "forecast_actual_vs_predicted.png")
    feature_imp_csv_path = os.path.join(results_dir, "feature_importance.csv")
    feature_imp_plot_path = os.path.join(results_dir, "feature_importance.png")
    slice_metrics_csv_path = os.path.join(results_dir, "slice_metrics.csv")
    slice_preds_csv_path = os.path.join(results_dir, "slice_predictions.csv")
    slice_plot_path = os.path.join(results_dir, "slice_forecasts.png")

    print("=" * 75)
    print("AI-BASED 5G NETWORK DATA ANALYTICS — END-TO-END FORECASTING PIPELINE")
    print("=" * 75)

    # 1. Data Generation
    print(f"\n[STEP 1/6] Generating synthetic 5G traffic data ({num_days} days, seed {random_seed})...")
    raw_df = generate_synthetic_5g_traffic(
        num_days=num_days,
        start_date="2026-01-01 00:00:00",
        random_seed=random_seed,
        output_path=raw_csv_path,
    )
    print(f"  Generated {len(raw_df)} hourly records across eMBB, URLLC, mMTC.")
    print(f"  Saved raw data to: {raw_csv_path}")

    # 2. Feature Engineering
    print("\n[STEP 2/6] Engineering leakage-safe lag and rolling window features...")
    feature_df = create_lag_and_rolling_features(
        raw_df,
        target_col="total_mbps",
        lag_hours=(1, 2, 3, 6, 12, 24),
        rolling_windows=(6, 24),
    )
    feature_cols = [col for col in feature_df.columns if col.startswith("lag_") or col.startswith("rolling_mean_")]
    print(f"  Feature columns created: {feature_cols}")
    print(f"  Total clean rows after dropping initial NaN rows: {len(feature_df)}")

    # 3. Train / Test Split
    print("\n[STEP 3/6] Chronological train/test split (80% train / 20% test, no shuffling)...")
    X_train, X_test, y_train, y_test, ts_train, ts_test = split_time_series_data(
        feature_df,
        feature_cols=feature_cols,
        target_col="total_mbps",
        train_ratio=0.8,
    )
    print(f"  Train set: {len(X_train)} samples ({ts_train.iloc[0]} to {ts_train.iloc[-1]})")
    print(f"  Test set:  {len(X_test)} samples ({ts_test.iloc[0]} to {ts_test.iloc[-1]})")

    # 4. Model Training & Evaluation
    print("\n[STEP 4/6] Training and benchmarking forecasting models...")

    # A. Naive Persistence Baseline (lag-1)
    y_pred_baseline = X_test["lag_1"]
    next_baseline_forecast = float(y_test.iloc[-1])

    # B. Random Forest (PPT configuration: 180 trees, max_depth 10, min_samples_leaf 3)
    print("  -> Training Random Forest Regressor (180 trees, depth 10, leaf 3)...")
    rf_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=180,
        max_depth=10,
        min_samples_leaf=3,
        random_state=random_seed,
    )
    y_pred_rf = rf_model.predict(X_test)

    # Next hour RF forecast using latest available features
    next_feat_dict = {}
    for lag in (1, 2, 3, 6, 12, 24):
        target_idx = len(raw_df) - lag
        next_feat_dict[f"lag_{lag}"] = raw_df["total_mbps"].iloc[target_idx]
    for w in (6, 24):
        next_feat_dict[f"rolling_mean_{w}"] = raw_df["total_mbps"].iloc[-w:].mean()
    next_feat_df = pd.DataFrame([next_feat_dict])[feature_cols]
    next_rf_forecast = float(np.round(rf_model.predict(next_feat_df)[0], 2))

    # C. ARIMA Baseline
    print("  -> Fitting ARIMA(2, 1, 2) Baseline...")
    arima_res = train_and_evaluate_arima(y_train, y_test, order=(2, 1, 2))
    y_pred_arima = arima_res["predictions"]
    next_arima_forecast = arima_res["next_hour_forecast"]

    # D. LSTM Baseline
    print("  -> Training PyTorch LSTM Sequential Forecaster (seq_len 24)...")
    lstm_res = train_and_evaluate_lstm(y_train, y_test, seq_len=24, hidden_dim=32, epochs=25, random_seed=random_seed)
    y_pred_lstm = lstm_res["predictions"]
    next_lstm_forecast = lstm_res["next_hour_forecast"]

    # E. Per-Slice Forecasters & Bottom-up summation
    print("  -> Training Per-Slice Forecasters (eMBB, URLLC, mMTC)...")
    slice_res = train_and_evaluate_slices(raw_df, train_ratio=0.8, random_state=random_seed)
    y_pred_bottom_up = slice_res["bottom_up_predictions"]
    next_slice_forecasts = slice_res["next_hour_forecasts"]
    next_bottom_up_forecast = slice_res["bottom_up_next_hour"]

    # Unified Multi-Model Metrics Evaluation Table
    predictions_dict = {
        "Naive Baseline (Lag-1)": y_pred_baseline,
        "ARIMA(2, 1, 2)": y_pred_arima,
        "LSTM Sequential Forecaster": y_pred_lstm,
        "Random Forest Regressor": y_pred_rf,
        "Bottom-Up Slice Summation": y_pred_bottom_up,
    }
    metrics_df = evaluate_multiple_models(y_test, predictions_dict, baseline_name="Naive Baseline (Lag-1)")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save Multi-Model Test Predictions
    predictions_df = pd.DataFrame(
        {
            "timestamp": ts_test.values,
            "actual_total_mbps": y_test.values,
            "baseline_pred_mbps": np.round(y_pred_baseline.values, 2),
            "random_forest_pred_mbps": np.round(y_pred_rf, 2),
            "arima_pred_mbps": np.round(y_pred_arima, 2),
            "lstm_pred_mbps": np.round(y_pred_lstm, 2),
            "bottom_up_slices_pred_mbps": np.round(y_pred_bottom_up, 2),
        }
    )
    predictions_df.to_csv(predictions_csv_path, index=False)

    # Save Slice Predictions and Metrics
    slice_metrics_df = slice_res["metrics_df"]
    slice_metrics_df.to_csv(slice_metrics_csv_path, index=False)

    slice_preds_df = pd.DataFrame(
        {
            "timestamp": ts_test.values,
            "actual_embb_mbps": slice_res["slice_actuals"]["embb"],
            "pred_embb_mbps": slice_res["slice_predictions"]["embb"],
            "actual_urllc_mbps": slice_res["slice_actuals"]["urllc"],
            "pred_urllc_mbps": slice_res["slice_predictions"]["urllc"],
            "actual_mmtc_mbps": slice_res["slice_actuals"]["mmtc"],
            "pred_mmtc_mbps": slice_res["slice_predictions"]["mmtc"],
            "bottom_up_total_mbps": y_pred_bottom_up,
            "actual_total_mbps": y_test.values,
        }
    )
    slice_preds_df.to_csv(slice_preds_csv_path, index=False)

    # 5. Capacity Recommendation Rule
    print("\n[STEP 5/6] Evaluating rule-based capacity recommendation...")
    capacity_rec = evaluate_capacity_recommendation(forecasted_mbps=next_rf_forecast, threshold_mbps=700.0)
    print(f"  Next-Hour Forecast: {next_rf_forecast} Mbps")
    print(f"  Capacity Status:    {capacity_rec['status']} ({capacity_rec['hint']})")

    # 6. Feature Importances and Visualizations
    print("\n[STEP 6/6] Computing feature importance and generating visualizations...")
    fi_df = extract_feature_importances(rf_model, feature_cols)
    fi_df.to_csv(feature_imp_csv_path, index=False)

    plot_actual_vs_predicted(
        ts_test,
        y_test,
        y_pred_rf,
        y_pred_baseline,
        y_pred_arima=y_pred_arima,
        y_pred_lstm=y_pred_lstm,
        output_path=actual_vs_pred_plot_path,
    )
    plot_feature_importance(fi_df, output_path=feature_imp_plot_path)
    plot_slice_forecasts(
        ts_test,
        slice_actuals=slice_res["slice_actuals"],
        slice_predictions=slice_res["slice_predictions"],
        bottom_up_preds=y_pred_bottom_up,
        total_actual=y_test.values,
        output_path=slice_plot_path,
    )

    print(f"  Saved multi-model forecast plot: {actual_vs_pred_plot_path}")
    print(f"  Saved feature importance plot:   {feature_imp_plot_path}")
    print(f"  Saved slice forecast plot:       {slice_plot_path}")

    print("\n" + "=" * 75)
    print("FINAL MULTI-MODEL BENCHMARK RESULTS (Test Split):")
    print("=" * 75)
    print(metrics_df.to_string(index=False))
    print("=" * 75)
    print("\nPER-SLICE FORECASTING BENCHMARK:")
    print("=" * 75)
    print(slice_metrics_df.to_string(index=False))
    print("=" * 75)

    return {
        "raw_df": raw_df,
        "feature_df": feature_df,
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
        "fi_df": fi_df,
        "slice_metrics_df": slice_metrics_df,
        "slice_preds_df": slice_preds_df,
        "capacity_rec": capacity_rec,
        "next_hour_forecasts": {
            "Random Forest": next_rf_forecast,
            "ARIMA(2, 1, 2)": next_arima_forecast,
            "LSTM Forecaster": next_lstm_forecast,
            "Naive Baseline": next_baseline_forecast,
            "Bottom-Up Slices Sum": next_bottom_up_forecast,
            "eMBB": next_slice_forecasts["embb"],
            "URLLC": next_slice_forecasts["urllc"],
            "mMTC": next_slice_forecasts["mmtc"],
        },
    }
