"""ARIMA forecasting baseline module for 5G network traffic.

Implements:
1. ARIMA(p, d, q) model fitting strictly on chronological training data.
2. Leakage-safe 1-step ahead out-of-sample forecasting across the test block.
3. 1-hour ahead future demand forecasting.
4. Metric evaluation (MAE, RMSE, R2).
"""

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from traffic_forecasting.model import calculate_metrics

logger = logging.getLogger(__name__)


def train_and_evaluate_arima(
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    order: tuple[int, int, int] = (2, 1, 2),
) -> dict[str, object]:
    """Train an ARIMA model on training data and generate leakage-safe 1-step ahead forecasts on test data.

    To ensure strict temporal causality:
    - The ARIMA parameters are fitted SOLELY on y_train.
    - Test evaluation evaluates 1-step ahead predictions using the learned parameters applied
      to the sequential history without refitting on test data.

    Args:
        y_train: Chronological training series of total traffic demand.
        y_test: Chronological test series of total traffic demand.
        order: (p, d, q) order of the ARIMA model (default: (2, 1, 2)).

    Returns:
        dict: {
            "model_fit": fitted statsmodels ARIMA model,
            "predictions": np.ndarray of 1-step ahead predictions for y_test,
            "next_hour_forecast": float for the 1-hour ahead forecast past the test set,
            "metrics": dict containing 'mae', 'rmse', 'r2',
            "order": tuple,
            "status": "SUCCESS" or error message,
        }
    """
    train_vals = np.asarray(y_train, dtype=float)
    test_vals = np.asarray(y_test, dtype=float)

    try:
        # Fit ARIMA strictly on training observations
        model = ARIMA(train_vals, order=order)
        model_fit = model.fit()

        # Out-of-sample one-step ahead forecasting:
        # Concatenate train and test, and apply the fixed fitted parameters
        # without refitting (refit=False) to obtain Kalman-filtered 1-step ahead forecasts.
        full_series = np.concatenate([train_vals, test_vals])
        applied = model_fit.apply(full_series, refit=False)

        # In applied.fittedvalues, index t corresponds to one-step forecast of observation t.
        # Test predictions correspond to indices from len(train_vals) to len(full_series).
        test_preds = applied.fittedvalues[len(train_vals):]

        # Calculate metrics against test ground truth
        metrics = calculate_metrics(test_vals, test_preds)

        # Next-hour forecast past the end of the test set
        next_forecast = float(np.round(applied.forecast(steps=1)[0], 2))

        return {
            "model_fit": model_fit,
            "predictions": np.round(test_preds, 2),
            "next_hour_forecast": next_forecast,
            "metrics": metrics,
            "order": order,
            "status": "SUCCESS",
        }
    except Exception as exc:
        logger.error(f"ARIMA training/evaluation failed: {exc}")
        # Graceful fallback: return naive lag-1 values as fallback predictions
        fallback_preds = np.roll(test_vals, 1)
        fallback_preds[0] = train_vals[-1]
        metrics = calculate_metrics(test_vals, fallback_preds)
        return {
            "model_fit": None,
            "predictions": fallback_preds,
            "next_hour_forecast": float(test_vals[-1]),
            "metrics": metrics,
            "order": order,
            "status": f"FAILED: {str(exc)}",
        }

