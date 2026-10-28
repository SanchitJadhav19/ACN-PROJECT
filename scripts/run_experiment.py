#!/usr/bin/env python3
"""Main entry point CLI script to execute the 5G traffic forecasting experiment.

Defaults to 60 days of synthetic hourly traffic (1,440 hours) to strictly match
the specification in the mid-semester presentation (acn_midsem.pdf).
"""

import argparse
import os
import sys

# Ensure src is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from traffic_forecasting.experiment import run_full_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 5G Network Data Analytics Traffic Forecasting Benchmark."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days of hourly synthetic 5G traffic (default: 60, matching PPT).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(project_root, "results"),
        help="Target folder to save generated datasets, CSV metrics, and plots.",
    )

    args = parser.parse_args()

    run_full_experiment(
        num_days=args.days,
        random_seed=args.seed,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
