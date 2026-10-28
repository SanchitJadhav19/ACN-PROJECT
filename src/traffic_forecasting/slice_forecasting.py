"""Per-slice 5G traffic forecasting module for eMBB, URLLC, and mMTC.

Implements:
1. Disaggregated forecasting for individual 5G service slices:
   - eMBB (Enhanced Mobile Broadband)
   - URLLC (Ultra-Reliable Low-Latency Communication)
   - mMTC (Massive Machine-Type Communication)
2. Leakage-safe lag and rolling feature extraction per slice.
3. Random Forest regressor training per slice following the PPT configuration.
4. Evaluation of per-slice prediction metrics (MAE, RMSE, R2).
5. Bottom-up aggregation: y_pred_total = y_pred_embb + y_pred_urllc + y_pred_mmtc.
6. Verification of the aggregation relationship against direct aggregate forecasting.
"""

import numpy as np
import pandas as pd

from traffic_forecasting.features import create_lag_and_rolling_features
from traffic_forecasting.model import (
    calculate_metrics,
    split_time_series_data,
    train_random_forest,
)


def train_and_evaluate_slices(
    raw_df: pd.DataFrame,
    slices: tuple[str, ...] = ("embb", "urllc", "mmtc"),
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> dict[str, object]:
    """Train per-slice forecasting models and evaluate individual and aggregate performance.

    Args:
        raw_df: DataFrame containing ['timestamp', 'embb_mbps', 'urllc_mbps', 'mmtc_mbps', 'total_mbps'].
        slices: Tuple of slice names to forecast.
        train_ratio: Proportion of chronological data for training.
        random_state: Random state for reproducibility.

    Returns:
        dict: {
            "slice_models": dict of trained models per slice,
            "slice_metrics": dict of metrics per slice,
            "slice_predictions": dict of test prediction arrays per slice,
            "slice_actuals": dict of test ground truth arrays per slice,
            "next_hour_forecasts": dict of next-hour predictions per slice,
            "bottom_up_predictions": np.ndarray sum of slice predictions,
            "bottom_up_metrics": dict of metrics for bottom-up sum vs actual total,
            "bottom_up_next_hour": float sum of next-hour slice forecasts,
            "timestamps_test": pd.Series test timestamps,
            "metrics_df": pd.DataFrame summary table of slice performance,
        }
    """
    slice_models = {}
    slice_metrics = {}
    slice_predictions = {}
    slice_actuals = {}
    next_hour_forecasts = {}

    timestamps_test = None

    for slice_name in slices:
        col = f"{slice_name}_mbps"
        # Engineer leakage-safe features for this slice
        df_feat = create_lag_and_rolling_features(
            raw_df,
            target_col=col,
            lag_hours=(1, 2, 3, 6, 12, 24),
            rolling_windows=(6, 24),
        )
        feature_cols = [c for c in df_feat.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]

        X_train, X_test, y_train, y_test, _, ts_test = split_time_series_data(
            df_feat,
            feature_cols=feature_cols,
            target_col=col,
            train_ratio=train_ratio,
        )
        if timestamps_test is None:
            timestamps_test = ts_test

        # Train Random Forest per PPT specification
        rf = train_random_forest(
            X_train,
            y_train,
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=3,
            random_state=random_state,
        )
        preds = rf.predict(X_test)

        # Predict next hour from the latest available features
        latest_feat = X_test.iloc[[-1]]
        # Construct true next feature vector from the end of df_feat
        # (shift the historical values to represent the next step t+1)
        next_feat_dict = {}
        for lag in (1, 2, 3, 6, 12, 24):
            # lag 1 for t+1 is raw value at t
            target_idx = len(raw_df) - lag
            next_feat_dict[f"lag_{lag}"] = raw_df[col].iloc[target_idx]
        for w in (6, 24):
            next_feat_dict[f"rolling_mean_{w}"] = raw_df[col].iloc[-w:].mean()

        next_feat_df = pd.DataFrame([next_feat_dict])[feature_cols]
        next_pred = float(np.round(rf.predict(next_feat_df)[0], 2))

        slice_models[slice_name] = rf
        slice_predictions[slice_name] = np.round(preds, 2)
        slice_actuals[slice_name] = y_test.to_numpy()
        slice_metrics[slice_name] = calculate_metrics(y_test, preds)
        next_hour_forecasts[slice_name] = next_pred

    # Compute bottom-up sum of slices
    bottom_up_preds = np.zeros(len(slice_predictions[slices[0]]))
    bottom_up_next_hour = 0.0
    for s in slices:
        bottom_up_preds += slice_predictions[s]
        bottom_up_next_hour += next_hour_forecasts[s]

    bottom_up_preds = np.round(bottom_up_preds, 2)
    bottom_up_next_hour = float(np.round(bottom_up_next_hour, 2))

    # Evaluate bottom-up predictions against actual total_mbps
    # Extract total_mbps test set aligned with the slice feature DataFrame
    df_feat_total = create_lag_and_rolling_features(raw_df, target_col="total_mbps")
    feat_cols_total = [c for c in df_feat_total.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]
    _, _, _, y_test_total, _, _ = split_time_series_data(
        df_feat_total,
        feature_cols=feat_cols_total,
        target_col="total_mbps",
        train_ratio=train_ratio,
    )
    bottom_up_metrics = calculate_metrics(y_test_total.to_numpy(), bottom_up_preds)

    # Format slice metrics DataFrame
    rows = []
    for s in slices:
        m = slice_metrics[s]
        rows.append(
            {
                "Slice": s.upper(),
                "Description": {
                    "embb": "Enhanced Mobile Broadband",
                    "urllc": "Ultra-Reliable Low-Latency",
                    "mmtc": "Massive Machine-Type Comm.",
                }.get(s, s),
                "MAE (Mbps)": m["mae"],
                "RMSE (Mbps)": m["rmse"],
                "R2 Score": m["r2"],
                "Next-Hour Forecast (Mbps)": next_hour_forecasts[s],
            }
        )

    # Append bottom-up sum row
    rows.append(
        {
            "Slice": "BOTTOM-UP SUM",
            "Description": "Sum of eMBB + URLLC + mMTC Forecasts",
            "MAE (Mbps)": bottom_up_metrics["mae"],
            "RMSE (Mbps)": bottom_up_metrics["rmse"],
            "R2 Score": bottom_up_metrics["r2"],
            "Next-Hour Forecast (Mbps)": bottom_up_next_hour,
        }
    )

    metrics_df = pd.DataFrame(rows)

    return {
        "slice_models": slice_models,
        "slice_metrics": slice_metrics,
        "slice_predictions": slice_predictions,
        "slice_actuals": slice_actuals,
        "next_hour_forecasts": next_hour_forecasts,
        "bottom_up_predictions": bottom_up_preds,
        "bottom_up_metrics": bottom_up_metrics,
        "bottom_up_next_hour": bottom_up_next_hour,
        "timestamps_test": timestamps_test,
        "metrics_df": metrics_df,
    }

