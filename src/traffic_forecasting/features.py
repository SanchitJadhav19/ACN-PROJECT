"""Feature engineering module for 5G network traffic forecasting.

Builds historical lag and rolling window features for time-series modeling
while strictly preventing data leakage by shifting historical windows.
"""

import pandas as pd


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "total_mbps",
    lag_hours: tuple[int, ...] = (1, 2, 3, 6, 12, 24),
    rolling_windows: tuple[int, ...] = (6, 24),
) -> pd.DataFrame:
    """Create lag and rolling mean features from the target traffic series.

    To ensure zero data leakage:
    - Lag_k at time t represents the traffic value at time t - k.
    - Rolling_mean_w at time t represents the average of the previous w hours
      (time t-w to t-1), computed by shifting the series by 1 before calculating the mean.

    Args:
        df: Input DataFrame containing the target traffic column.
        target_col: Column name of the total traffic series.
        lag_hours: List/tuple of hourly lag intervals (default: 1, 2, 3, 6, 12, 24).
        rolling_windows: List/tuple of window sizes for rolling means (default: 6, 24).

    Returns:
        pd.DataFrame: New DataFrame containing original columns, lag features,
                      rolling mean features, with NaN rows removed.
    """
    df_feat = df.copy()

    # Create lag features for historical traffic
    for lag in lag_hours:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    # Create rolling mean features (shift by 1 first to prevent lookahead leakage)
    shifted_series = df_feat[target_col].shift(1)
    for window in rolling_windows:
        df_feat[f"rolling_mean_{window}"] = shifted_series.rolling(window=window).mean()

    # Drop initial rows that contain NaN due to shifting / rolling window
    df_clean = df_feat.dropna().reset_index(drop=True)

    return df_clean
