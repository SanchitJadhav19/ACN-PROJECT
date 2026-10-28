"""Data generation module for synthetic 5G network traffic.

Generates realistic hourly time-series traffic for three 5G service slices:
- eMBB: Enhanced Mobile Broadband (video streaming, gaming, high bandwidth)
- URLLC: Ultra-Reliable Low-Latency Communication (industrial IoT, autonomous systems)
- mMTC: Massive Machine-Type Communication (smart meters, sensors, telemetry)
- Total: Sum of eMBB, URLLC, and mMTC traffic
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_5g_traffic(
    num_days: int = 60,
    start_date: str = "2026-01-01 00:00:00",
    random_seed: int = 42,
    output_path: str | None = "results/synthetic_5g_traffic.csv",
) -> pd.DataFrame:
    """Generate synthetic hourly 5G traffic data for eMBB, URLLC, and mMTC.

    Args:
        num_days: Number of days of hourly traffic to generate (default: 60 days = 1440 hours).
        start_date: Start timestamp for the time series.
        random_seed: Random seed for reproducibility.
        output_path: File path to save the generated CSV. If None, file is not saved.

    Returns:
        pd.DataFrame: DataFrame containing [timestamp, embb_mbps, urllc_mbps, mmtc_mbps, total_mbps].
    """
    np.random.seed(random_seed)
    total_hours = num_days * 24

    # Generate hourly timestamps
    timestamps = pd.date_range(start=start_date, periods=total_hours, freq="h")
    hour_of_day = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()  # 0=Monday, 6=Sunday
    is_weekend = (day_of_week >= 5).astype(int)

    # 1. eMBB Slice: Strong diurnal variation peaking in evening (19:00 - 22:00)
    # Higher demand on weekends due to streaming and entertainment
    embb_diurnal = 250 + 150 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
    embb_evening_peak = 100 * np.exp(-0.5 * ((hour_of_day - 20) / 2.5) ** 2)
    embb_weekend_factor = 1.0 + 0.15 * is_weekend
    embb_noise = np.random.normal(loc=0, scale=20, size=total_hours)
    embb_mbps = (embb_diurnal + embb_evening_peak) * embb_weekend_factor + embb_noise
    embb_mbps = np.clip(embb_mbps, a_min=50.0, a_max=None)

    # 2. URLLC Slice: Business-hour activity (08:00 - 18:00), lower on weekends
    urllc_diurnal = 50 + 25 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    urllc_workday_boost = 20 * ((hour_of_day >= 8) & (hour_of_day <= 18)) * (1 - is_weekend)
    urllc_noise = np.random.normal(loc=0, scale=5, size=total_hours)
    urllc_mbps = urllc_diurnal + urllc_workday_boost + urllc_noise
    urllc_mbps = np.clip(urllc_mbps, a_min=15.0, a_max=None)

    # 3. mMTC Slice: Periodic sensor telemetry reporting and steady background
    mmtc_baseline = 80 + 15 * np.sin(2 * np.pi * hour_of_day / 12)
    mmtc_noise = np.random.normal(loc=0, scale=8, size=total_hours)
    mmtc_mbps = mmtc_baseline + mmtc_noise
    mmtc_mbps = np.clip(mmtc_mbps, a_min=20.0, a_max=None)

    # Round individual slices to 2 decimal places
    embb_mbps = np.round(embb_mbps, 2)
    urllc_mbps = np.round(urllc_mbps, 2)
    mmtc_mbps = np.round(mmtc_mbps, 2)

    # Total demand is the exact sum of all 3 slices
    total_mbps = np.round(embb_mbps + urllc_mbps + mmtc_mbps, 2)

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "embb_mbps": embb_mbps,
            "urllc_mbps": urllc_mbps,
            "mmtc_mbps": mmtc_mbps,
            "total_mbps": total_mbps,
        }
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
