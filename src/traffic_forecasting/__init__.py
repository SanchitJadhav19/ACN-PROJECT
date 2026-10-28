"""5G Network Data Analytics: Traffic Forecasting Package.

Predicts near-future total and per-slice 5G traffic demand using historical lag and rolling features
with Random Forest Regression, ARIMA, and LSTM sequential networks, and compares against
a naive persistence baseline. Includes rule-based capacity recommendations for network resource planning.
"""

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
from traffic_forecasting.experiment import run_full_experiment

__all__ = [
    "generate_synthetic_5g_traffic",
    "create_lag_and_rolling_features",
    "split_time_series_data",
    "train_random_forest",
    "calculate_metrics",
    "evaluate_forecasts",
    "evaluate_multiple_models",
    "extract_feature_importances",
    "train_and_evaluate_arima",
    "train_and_evaluate_lstm",
    "train_and_evaluate_slices",
    "evaluate_capacity_recommendation",
    "run_full_experiment",
]
