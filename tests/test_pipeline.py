"""Comprehensive unit and integration tests for the 5G traffic analytics pipeline."""

import numpy as np
import pandas as pd
import pytest

from traffic_forecasting.data import generate_synthetic_5g_traffic
from traffic_forecasting.features import create_lag_and_rolling_features
from traffic_forecasting.model import (
    calculate_metrics,
    evaluate_forecasts,
    evaluate_multiple_models,
    split_time_series_data,
    train_random_forest,
)
from traffic_forecasting.arima_model import train_and_evaluate_arima
from traffic_forecasting.lstm_model import train_and_evaluate_lstm
from traffic_forecasting.slice_forecasting import train_and_evaluate_slices
from traffic_forecasting.capacity import evaluate_capacity_recommendation
from traffic_forecasting.app import app


def test_data_generator_columns_and_no_nan():
    """Test that data generator creates expected columns and contains no missing values."""
    df = generate_synthetic_5g_traffic(num_days=7, random_seed=42, output_path=None)

    expected_cols = ["timestamp", "embb_mbps", "urllc_mbps", "mmtc_mbps", "total_mbps"]
    assert list(df.columns) == expected_cols
    assert len(df) == 7 * 24
    assert df.isna().sum().sum() == 0

    # Total should equal sum of slices within rounding tolerance
    computed_total = np.round(df["embb_mbps"] + df["urllc_mbps"] + df["mmtc_mbps"], 2)
    assert np.allclose(df["total_mbps"], computed_total, atol=0.01)


def test_aggregate_traffic_relationship():
    """Verify strict mathematical relationship y_total = y_embb + y_urllc + y_mmtc across all time steps."""
    df = generate_synthetic_5g_traffic(num_days=30, random_seed=42, output_path=None)
    slice_sum = df["embb_mbps"] + df["urllc_mbps"] + df["mmtc_mbps"]
    diff = np.abs(df["total_mbps"] - np.round(slice_sum, 2))
    assert np.max(diff) < 1e-4


def test_feature_builder_expected_columns_and_no_nan():
    """Test that feature builder creates all expected lag and rolling features with no NaNs."""
    df = generate_synthetic_5g_traffic(num_days=7, random_seed=42, output_path=None)
    df_feat = create_lag_and_rolling_features(
        df,
        target_col="total_mbps",
        lag_hours=(1, 2, 3, 6, 12, 24),
        rolling_windows=(6, 24),
    )

    expected_feature_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "lag_24",
        "rolling_mean_6",
        "rolling_mean_24",
    ]
    for col in expected_feature_cols:
        assert col in df_feat.columns

    # Check that initial NaN rows are dropped
    assert df_feat.isna().sum().sum() == 0
    # Since max lag/window is 24, length should be total - 24
    assert len(df_feat) == len(df) - 24


def test_no_leakage_lag_1():
    """Test that lag_1 at time t strictly equals total_mbps at time t-1."""
    df = generate_synthetic_5g_traffic(num_days=5, random_seed=42, output_path=None)
    df_feat = create_lag_and_rolling_features(df, target_col="total_mbps")

    for i in range(min(50, len(df_feat))):
        raw_idx = i + 24
        assert df_feat.loc[i, "lag_1"] == df.loc[raw_idx - 1, "total_mbps"]


def test_no_future_leakage_rolling_means():
    """Test that rolling mean at time t averages observations up to t-1, never including time t."""
    df = generate_synthetic_5g_traffic(num_days=5, random_seed=42, output_path=None)
    df_feat = create_lag_and_rolling_features(df, target_col="total_mbps")

    for i in range(min(30, len(df_feat))):
        raw_idx = i + 24
        # rolling_mean_6 at row i must equal mean of raw df total_mbps from (raw_idx - 6) to (raw_idx - 1)
        expected_mean_6 = df.loc[raw_idx - 6 : raw_idx - 1, "total_mbps"].mean()
        assert np.isclose(df_feat.loc[i, "rolling_mean_6"], expected_mean_6, atol=1e-5)


def test_chronological_split_strict_ordering():
    """Test that chronological split strictly maintains temporal separation without future leakage."""
    df = generate_synthetic_5g_traffic(num_days=10, random_seed=42, output_path=None)
    df_feat = create_lag_and_rolling_features(df)
    feature_cols = [c for c in df_feat.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]

    X_train, X_test, y_train, y_test, ts_train, ts_test = split_time_series_data(
        df_feat,
        feature_cols=feature_cols,
        target_col="total_mbps",
        train_ratio=0.8,
    )

    # Max train timestamp must be strictly less than min test timestamp
    assert pd.to_datetime(ts_train.iloc[-1]) < pd.to_datetime(ts_test.iloc[0])
    assert len(X_train) + len(X_test) == len(df_feat)
    assert len(y_train) == int(len(df_feat) * 0.8)


def test_model_training_and_evaluation_pipeline():
    """Test full train/test split, Random Forest fitting, and evaluation outputs."""
    df = generate_synthetic_5g_traffic(num_days=14, random_seed=42, output_path=None)
    df_feat = create_lag_and_rolling_features(df)
    feature_cols = [c for c in df_feat.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]

    X_train, X_test, y_train, y_test, _, _ = split_time_series_data(
        df_feat,
        feature_cols=feature_cols,
        target_col="total_mbps",
        train_ratio=0.8,
    )

    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=30,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
    )
    y_pred_rf = model.predict(X_test)
    y_pred_baseline = X_test["lag_1"]

    metrics_df = evaluate_forecasts(y_test, y_pred_rf, y_pred_baseline)
    assert "Model" in metrics_df.columns
    assert "MAE (Mbps)" in metrics_df.columns
    assert "RMSE (Mbps)" in metrics_df.columns
    assert "R2 Score" in metrics_df.columns
    assert len(metrics_df) == 2


def test_arima_baseline_forecasting():
    """Test ARIMA(2, 1, 2) baseline fitting, 1-step ahead prediction, and metrics."""
    df = generate_synthetic_5g_traffic(num_days=14, random_seed=42, output_path=None)
    df_feat = create_lag_and_rolling_features(df)
    feature_cols = [c for c in df_feat.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]

    _, _, y_train, y_test, _, _ = split_time_series_data(
        df_feat,
        feature_cols=feature_cols,
        target_col="total_mbps",
        train_ratio=0.8,
    )

    arima_res = train_and_evaluate_arima(y_train, y_test, order=(2, 1, 2))
    assert arima_res["status"] == "SUCCESS"
    assert len(arima_res["predictions"]) == len(y_test)
    assert "mae" in arima_res["metrics"]
    assert arima_res["metrics"]["mae"] > 0
    assert arima_res["metrics"]["r2"] > 0.7
    assert isinstance(arima_res["next_hour_forecast"], float)


def test_lstm_baseline_forecasting():
    """Test PyTorch LSTM sequential forecaster, normalization, and predictions."""
    df = generate_synthetic_5g_traffic(num_days=14, random_seed=42, output_path=None)
    df_feat = create_lag_and_rolling_features(df)
    feature_cols = [c for c in df_feat.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]

    _, _, y_train, y_test, _, _ = split_time_series_data(
        df_feat,
        feature_cols=feature_cols,
        target_col="total_mbps",
        train_ratio=0.8,
    )

    lstm_res = train_and_evaluate_lstm(y_train, y_test, seq_len=24, hidden_dim=16, epochs=10, random_seed=42)
    assert lstm_res["status"] == "SUCCESS"
    assert len(lstm_res["predictions"]) == len(y_test)
    assert "mae" in lstm_res["metrics"]
    assert lstm_res["metrics"]["mae"] > 0
    assert isinstance(lstm_res["next_hour_forecast"], float)


def test_per_slice_forecasting_and_summation():
    """Test individual eMBB, URLLC, mMTC forecasting and bottom-up sum evaluation."""
    df = generate_synthetic_5g_traffic(num_days=14, random_seed=42, output_path=None)
    slice_res = train_and_evaluate_slices(df, train_ratio=0.8, random_state=42)

    for s in ("embb", "urllc", "mmtc"):
        assert s in slice_res["slice_predictions"]
        assert s in slice_res["slice_metrics"]
        assert slice_res["slice_metrics"][s]["mae"] > 0

    # Bottom-up sum length must match test length
    assert len(slice_res["bottom_up_predictions"]) == len(slice_res["slice_predictions"]["embb"])
    # Next hour forecast must equal sum of slice forecasts
    expected_sum = (
        slice_res["next_hour_forecasts"]["embb"]
        + slice_res["next_hour_forecasts"]["urllc"]
        + slice_res["next_hour_forecasts"]["mmtc"]
    )
    assert np.isclose(slice_res["bottom_up_next_hour"], expected_sum, atol=0.05)


def test_capacity_recommendation_rules():
    """Test rule-based capacity recommendation logic and threshold evaluations."""
    # Test high traffic condition (exceeds threshold)
    rec_high = evaluate_capacity_recommendation(forecasted_mbps=750.0, threshold_mbps=700.0)
    assert rec_high["status"] == "ADD_CAPACITY"
    assert rec_high["is_high_traffic"] is True
    assert rec_high["headroom_mbps"] == 50.0
    assert "Add Capacity" in rec_high["hint"]
    assert "Advisory Notice" in rec_high["disclaimer"]

    # Test nominal traffic condition (below threshold)
    rec_low = evaluate_capacity_recommendation(forecasted_mbps=620.0, threshold_mbps=700.0)
    assert rec_low["status"] == "NOMINAL"
    assert rec_low["is_high_traffic"] is False
    assert rec_low["headroom_mbps"] == -80.0
    assert "Nominal" in rec_low["hint"]


def test_flask_web_ui_endpoints():
    """Test Flask backend routes and REST API endpoints."""
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Main dashboard route
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"AI-Based Network Data Analytics for 5G" in resp.data

    # 2. API Summary
    resp_summary = client.get("/api/summary?threshold=700")
    assert resp_summary.status_code == 200
    json_summary = resp_summary.get_json()
    assert json_summary["status"] == "success"
    assert "next_hour_forecast_mbps" in json_summary

    # 3. API Metrics
    resp_metrics = client.get("/api/metrics")
    assert resp_metrics.status_code == 200
    json_metrics = resp_metrics.get_json()
    assert json_metrics["status"] == "success"
    assert len(json_metrics["current_metrics"]) >= 4

    # 4. API Predictions
    resp_preds = client.get("/api/predictions?limit=24")
    assert resp_preds.status_code == 200
    json_preds = resp_preds.get_json()
    assert json_preds["status"] == "success"
    assert len(json_preds["timestamps"]) == 24

    # 5. API Slices
    resp_slices = client.get("/api/slices?limit=24")
    assert resp_slices.status_code == 200
    json_slices = resp_slices.get_json()
    assert json_slices["status"] == "success"
    assert "actual_embb_mbps" in json_slices

    # 6. API Recommendation
    resp_rec = client.post("/api/recommendation", json={"threshold_mbps": 750, "forecasted_mbps": 800})
    assert resp_rec.status_code == 200
    json_rec = resp_rec.get_json()
    assert json_rec["recommendation"]["status"] == "ADD_CAPACITY"

    # 7. API Run Pipeline (end-to-end trigger in worker thread/request context)
    resp_run = client.post("/api/run-pipeline", json={"num_days": 10, "seed": 42})
    assert resp_run.status_code == 200
    json_run = resp_run.get_json()
    assert json_run["status"] == "success"
    assert "metrics" in json_run
    assert len(json_run["metrics"]) >= 4
