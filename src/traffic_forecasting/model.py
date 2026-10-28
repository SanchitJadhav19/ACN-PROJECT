"""Model training, baseline forecasting, and evaluation module.

Implements:
1. Time-aware sequential train/test split (no shuffling to preserve temporal order).
2. Random Forest Regressor for 1-hour ahead total 5G traffic demand forecasting.
3. Naive persistence baseline (prediction = lag_1 value at time t-1).
4. Evaluation metrics (MAE, RMSE, R2, and % MAE improvement).
5. Feature importance extraction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_time_series_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "total_mbps",
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split dataset chronologically into train and test sets (no shuffling).

    Args:
        df: Feature-engineered DataFrame.
        feature_cols: List of column names used as input features.
        target_col: Target column to forecast.
        train_ratio: Proportion of chronological data for training (default: 0.8).

    Returns:
        tuple: (X_train, X_test, y_train, y_test, timestamps_train, timestamps_test)
    """
    n_samples = len(df)
    split_idx = int(n_samples * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    timestamps_train = train_df["timestamp"].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()
    timestamps_test = test_df["timestamp"].copy()

    return X_train, X_test, y_train, y_test, timestamps_train, timestamps_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 180,
    max_depth: int | None = 10,
    min_samples_leaf: int = 3,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a Random Forest Regressor on training features and target.

    Follows the specification from the mid-semester presentation (acn_midsem.pdf):
    - 180 trees
    - max_depth = 10
    - min_samples_leaf = 3
    - random_state = 42

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.
        n_estimators: Number of trees in the forest (default: 180).
        max_depth: Maximum depth of trees (default: 10).
        min_samples_leaf: Minimum samples required at leaf node (default: 3).
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        RandomForestRegressor: Trained model instance.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics: MAE, RMSE, and R2.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        dict: Dictionary containing 'mae', 'rmse', 'r2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    # Using root_mean_squared_error or np.sqrt(mean_squared_error) for compatibility
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(np.round(mae, 2)),
        "rmse": float(np.round(rmse, 2)),
        "r2": float(np.round(r2, 4)),
    }


def evaluate_forecasts(
    y_test: pd.Series,
    y_pred_rf: np.ndarray,
    y_pred_baseline: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Evaluate Random Forest model against the Naive Baseline.

    Args:
        y_test: Actual ground truth values in the test set.
        y_pred_rf: Predictions from the Random Forest model.
        y_pred_baseline: Predictions from the Naive Baseline (lag_1).

    Returns:
        pd.DataFrame: Summary table comparing Baseline and Random Forest metrics.
    """
    baseline_metrics = calculate_metrics(y_test, y_pred_baseline)
    rf_metrics = calculate_metrics(y_test, y_pred_rf)

    improvement_mae = ((baseline_metrics["mae"] - rf_metrics["mae"]) / baseline_metrics["mae"]) * 100.0

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Naive Baseline (Lag-1)",
                "MAE (Mbps)": baseline_metrics["mae"],
                "RMSE (Mbps)": baseline_metrics["rmse"],
                "R2 Score": baseline_metrics["r2"],
                "MAE Improvement (%)": 0.0,
            },
            {
                "Model": "Random Forest Regressor",
                "MAE (Mbps)": rf_metrics["mae"],
                "RMSE (Mbps)": rf_metrics["rmse"],
                "R2 Score": rf_metrics["r2"],
                "MAE Improvement (%)": np.round(improvement_mae, 2),
            },
        ]
    )

    return metrics_df


def evaluate_multiple_models(
    y_test: pd.Series | np.ndarray,
    predictions_dict: dict[str, np.ndarray | pd.Series],
    baseline_name: str = "Naive Baseline (Lag-1)",
) -> pd.DataFrame:
    """Evaluate multiple forecasting models against a reference baseline.

    Args:
        y_test: Actual ground truth values in the test set.
        predictions_dict: Dictionary mapping model names to prediction arrays.
        baseline_name: Key in predictions_dict to use as the percentage improvement baseline.

    Returns:
        pd.DataFrame: Comparative metrics table for all models with MAE, RMSE, R2, and MAE Improvement.
    """
    baseline_mae = None
    if baseline_name in predictions_dict:
        base_preds = predictions_dict[baseline_name]
        baseline_mae = calculate_metrics(y_test, base_preds)["mae"]

    rows = []
    for name, preds in predictions_dict.items():
        m = calculate_metrics(y_test, preds)
        if baseline_mae is not None and baseline_mae > 0:
            imp = ((baseline_mae - m["mae"]) / baseline_mae) * 100.0
            imp_val = 0.0 if name == baseline_name else np.round(imp, 2)
        else:
            imp_val = 0.0

        rows.append(
            {
                "Model": name,
                "MAE (Mbps)": m["mae"],
                "RMSE (Mbps)": m["rmse"],
                "R2 Score": m["r2"],
                "MAE Improvement (%)": imp_val,
            }
        )

    return pd.DataFrame(rows)


def extract_feature_importances(
    model: RandomForestRegressor,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Extract and sort feature importances from a trained Random Forest model.

    Args:
        model: Fitted RandomForestRegressor.
        feature_cols: Names of input features.

    Returns:
        pd.DataFrame: DataFrame sorted by importance descending.
    """
    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": np.round(importances, 4),
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return fi_df
