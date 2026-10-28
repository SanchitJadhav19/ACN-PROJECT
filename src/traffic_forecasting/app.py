"""Flask Web Application for 5G Traffic Forecasting and Network Analytics Dashboard.

Provides an academic web interface for:
1. Executive dashboard and pipeline metadata.
2. Multi-model evaluation benchmark (Naive Baseline, ARIMA, LSTM, Random Forest, Bottom-Up Sum).
3. Interactive traffic visualization (total traffic and slice breakdown: eMBB, URLLC, mMTC).
4. Actual vs Predicted model demand comparisons.
5. Random Forest feature importance inspection.
6. Rule-based proactive capacity recommendation with dynamic threshold adjustment.
7. Pipeline execution controls with explicit distinction between cached and newly executed runs.
"""

import datetime
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from traffic_forecasting.capacity import evaluate_capacity_recommendation
from traffic_forecasting.experiment import run_full_experiment

# Determine paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
HISTORICAL_DIR = os.path.join(RESULTS_DIR, "historical_midsem")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)

# In-memory tracking for pipeline execution status
state = {
    "last_run_time": None,
    "last_run_type": "Existing Saved Results (Default)",
    "is_running": False,
    "last_error": None,
}


def load_csv_safe(path: str) -> pd.DataFrame | None:
    """Safely read CSV file if it exists, returning None otherwise."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def get_latest_results() -> dict[str, object]:
    """Load latest metrics, predictions, and slice data from the results directory."""
    metrics_path = os.path.join(RESULTS_DIR, "forecast_metrics.csv")
    preds_path = os.path.join(RESULTS_DIR, "forecast_predictions.csv")
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    slices_metrics_path = os.path.join(RESULTS_DIR, "slice_metrics.csv")
    slices_preds_path = os.path.join(RESULTS_DIR, "slice_predictions.csv")
    raw_path = os.path.join(RESULTS_DIR, "synthetic_5g_traffic.csv")
    hist_metrics_path = os.path.join(HISTORICAL_DIR, "forecast_metrics.csv")

    metrics_df = load_csv_safe(metrics_path)
    preds_df = load_csv_safe(preds_path)
    fi_df = load_csv_safe(fi_path)
    slice_m_df = load_csv_safe(slices_metrics_path)
    slice_p_df = load_csv_safe(slices_preds_path)
    raw_df = load_csv_safe(raw_path)
    hist_metrics_df = load_csv_safe(hist_metrics_path)

    # If primary results don't exist yet, run experiment
    if metrics_df is None or preds_df is None or raw_df is None:
        exp_res = run_full_experiment(num_days=60, random_seed=42, results_dir=RESULTS_DIR)
        metrics_df = exp_res["metrics_df"]
        preds_df = exp_res["predictions_df"]
        fi_df = exp_res["fi_df"]
        slice_m_df = exp_res["slice_metrics_df"]
        slice_p_df = exp_res["slice_preds_df"]
        raw_df = exp_res["raw_df"]
        state["last_run_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["last_run_type"] = "Initial System Pipeline Run (60 days)"

    return {
        "metrics_df": metrics_df,
        "preds_df": preds_df,
        "fi_df": fi_df,
        "slice_m_df": slice_m_df,
        "slice_p_df": slice_p_df,
        "raw_df": raw_df,
        "hist_metrics_df": hist_metrics_df,
    }


@app.route("/")
def index():
    """Render the academic dashboard interface."""
    data = get_latest_results()
    metrics_list = data["metrics_df"].to_dict(orient="records") if data["metrics_df"] is not None else []
    slice_metrics_list = data["slice_m_df"].to_dict(orient="records") if data["slice_m_df"] is not None else []
    hist_metrics_list = data["hist_metrics_df"].to_dict(orient="records") if data["hist_metrics_df"] is not None else []

    # Compute key summary stats
    total_samples = len(data["raw_df"]) if data["raw_df"] is not None else 0
    test_samples = len(data["preds_df"]) if data["preds_df"] is not None else 0
    train_samples = total_samples - 24 - test_samples if total_samples > 24 else 0

    # Best model (Random Forest) next forecast
    rf_next = 250.41
    if data["preds_df"] is not None and "random_forest_pred_mbps" in data["preds_df"].columns:
        rf_next = float(data["preds_df"]["random_forest_pred_mbps"].iloc[-1])

    initial_rec = evaluate_capacity_recommendation(forecasted_mbps=rf_next, threshold_mbps=700.0)

    context = {
        "project_title": "AI-Based Network Data Analytics for 5G",
        "subtitle": "Forecasting Total Traffic Demand from Recent History",
        "institution": "National Institute of Technology Karnataka (NITK), Surathkal",
        "course": "Advanced Computer Networks",
        "team_members": [
            {"name": "Sanchit Sanjay Jadhav", "roll": "231IT059"},
            {"name": "Sambhav Jetendra Purohit", "roll": "231IT053"},
            {"name": "Kunal Laxmikant Katkar", "roll": "231IT033"},
        ],
        "metrics": metrics_list,
        "slice_metrics": slice_metrics_list,
        "historical_metrics": hist_metrics_list,
        "total_samples": total_samples,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "rf_next_forecast": rf_next,
        "initial_rec": initial_rec,
        "last_run_type": state["last_run_type"],
        "last_run_time": state["last_run_time"] or "Loaded from saved experiment run",
    }
    return render_template("dashboard.html", **context)


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """Return executive summary metadata and latest forecast indicators."""
    data = get_latest_results()
    preds_df = data["preds_df"]
    rf_next = 250.41
    if preds_df is not None and "random_forest_pred_mbps" in preds_df.columns:
        rf_next = float(preds_df["random_forest_pred_mbps"].iloc[-1])

    threshold = float(request.args.get("threshold", 700.0))
    rec = evaluate_capacity_recommendation(forecasted_mbps=rf_next, threshold_mbps=threshold)

    return jsonify(
        {
            "status": "success",
            "next_hour_forecast_mbps": rf_next,
            "threshold_mbps": threshold,
            "capacity_recommendation": rec,
            "last_run_type": state["last_run_type"],
            "last_run_time": state["last_run_time"] or "From saved disk cache",
        }
    )


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return model benchmarking comparison metrics."""
    data = get_latest_results()
    metrics = data["metrics_df"].to_dict(orient="records") if data["metrics_df"] is not None else []
    hist_metrics = data["hist_metrics_df"].to_dict(orient="records") if data["hist_metrics_df"] is not None else []
    slice_metrics = data["slice_m_df"].to_dict(orient="records") if data["slice_m_df"] is not None else []

    # Midsem presentation reported baseline for reference
    midsem_reported = [
        {"Model": "Naive lag-1 (PPT Table I)", "MAE (Mbps)": 43.73, "RMSE (Mbps)": 54.26, "R2 Score": 0.711, "MAE Improvement (%)": 0.0},
        {"Model": "Random Forest (PPT Table I)", "MAE (Mbps)": 31.52, "RMSE (Mbps)": 39.25, "R2 Score": 0.849, "MAE Improvement (%)": 27.9},
    ]

    return jsonify(
        {
            "status": "success",
            "current_metrics": metrics,
            "slice_metrics": slice_metrics,
            "historical_90day_metrics": hist_metrics,
            "midsem_ppt_reported_metrics": midsem_reported,
        }
    )


@app.route("/api/traffic-history", methods=["GET"])
def api_traffic_history():
    """Return recent historical traffic records for charting."""
    data = get_latest_results()
    raw_df = data["raw_df"]
    if raw_df is None:
        return jsonify({"status": "error", "message": "No traffic data available"}), 404

    # Downsample or limit to last 168 hours (1 week) or full series upon request
    limit = int(request.args.get("limit", 168))
    df_sub = raw_df.tail(limit)

    return jsonify(
        {
            "status": "success",
            "timestamps": df_sub["timestamp"].tolist(),
            "total_mbps": df_sub["total_mbps"].tolist(),
            "embb_mbps": df_sub["embb_mbps"].tolist(),
            "urllc_mbps": df_sub["urllc_mbps"].tolist(),
            "mmtc_mbps": df_sub["mmtc_mbps"].tolist(),
        }
    )


@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    """Return test set predictions across models."""
    data = get_latest_results()
    preds_df = data["preds_df"]
    if preds_df is None:
        return jsonify({"status": "error", "message": "No predictions available"}), 404

    limit = int(request.args.get("limit", 168))
    df_sub = preds_df.head(limit)

    return jsonify(
        {
            "status": "success",
            "timestamps": df_sub["timestamp"].tolist(),
            "actual_total_mbps": df_sub["actual_total_mbps"].tolist(),
            "baseline_pred_mbps": df_sub["baseline_pred_mbps"].tolist() if "baseline_pred_mbps" in df_sub.columns else [],
            "random_forest_pred_mbps": df_sub["random_forest_pred_mbps"].tolist() if "random_forest_pred_mbps" in df_sub.columns else [],
            "arima_pred_mbps": df_sub["arima_pred_mbps"].tolist() if "arima_pred_mbps" in df_sub.columns else [],
            "lstm_pred_mbps": df_sub["lstm_pred_mbps"].tolist() if "lstm_pred_mbps" in df_sub.columns else [],
            "bottom_up_slices_pred_mbps": df_sub["bottom_up_slices_pred_mbps"].tolist() if "bottom_up_slices_pred_mbps" in df_sub.columns else [],
        }
    )


@app.route("/api/slices", methods=["GET"])
def api_slices():
    """Return per-slice test predictions and bottom-up aggregation."""
    data = get_latest_results()
    slice_p_df = data["slice_p_df"]
    slice_m_df = data["slice_m_df"]
    if slice_p_df is None:
        return jsonify({"status": "error", "message": "No slice predictions available"}), 404

    limit = int(request.args.get("limit", 168))
    df_sub = slice_p_df.head(limit)

    return jsonify(
        {
            "status": "success",
            "timestamps": df_sub["timestamp"].tolist(),
            "actual_embb_mbps": df_sub["actual_embb_mbps"].tolist(),
            "pred_embb_mbps": df_sub["pred_embb_mbps"].tolist(),
            "actual_urllc_mbps": df_sub["actual_urllc_mbps"].tolist(),
            "pred_urllc_mbps": df_sub["pred_urllc_mbps"].tolist(),
            "actual_mmtc_mbps": df_sub["actual_mmtc_mbps"].tolist(),
            "pred_mmtc_mbps": df_sub["pred_mmtc_mbps"].tolist(),
            "bottom_up_total_mbps": df_sub["bottom_up_total_mbps"].tolist(),
            "actual_total_mbps": df_sub["actual_total_mbps"].tolist(),
            "metrics": slice_m_df.to_dict(orient="records") if slice_m_df is not None else [],
        }
    )


@app.route("/api/feature-importance", methods=["GET"])
def api_feature_importance():
    """Return Random Forest feature importance data."""
    data = get_latest_results()
    fi_df = data["fi_df"]
    if fi_df is None:
        return jsonify({"status": "error", "message": "No feature importance data"}), 404

    return jsonify(
        {
            "status": "success",
            "features": fi_df["feature"].tolist(),
            "importance": fi_df["importance"].tolist(),
        }
    )


@app.route("/api/recommendation", methods=["POST"])
def api_recommendation():
    """Compute capacity recommendation for a given threshold and forecast."""
    req_data = request.get_json(silent=True) or {}
    threshold = float(req_data.get("threshold_mbps", 700.0))
    forecast = req_data.get("forecasted_mbps")

    if forecast is None:
        # Default to latest Random Forest test prediction
        data = get_latest_results()
        if data["preds_df"] is not None and "random_forest_pred_mbps" in data["preds_df"].columns:
            forecast = float(data["preds_df"]["random_forest_pred_mbps"].iloc[-1])
        else:
            forecast = 250.41

    rec = evaluate_capacity_recommendation(forecasted_mbps=float(forecast), threshold_mbps=threshold)
    return jsonify({"status": "success", "recommendation": rec})


@app.route("/api/run-pipeline", methods=["POST"])
def api_run_pipeline():
    """Execute the forecasting pipeline on-demand."""
    if state["is_running"]:
        return jsonify({"status": "busy", "message": "Pipeline execution is currently in progress."}), 429

    req_data = request.get_json(silent=True) or {}
    num_days = int(req_data.get("num_days", 60))
    seed = int(req_data.get("seed", 42))

    state["is_running"] = True
    try:
        run_res = run_full_experiment(
            num_days=num_days,
            random_seed=seed,
            results_dir=RESULTS_DIR,
        )
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["last_run_time"] = now_str
        state["last_run_type"] = f"Live Run ({num_days} days, seed {seed})"

        return jsonify(
            {
                "status": "success",
                "message": f"Pipeline successfully executed with {num_days} days of synthetic traffic.",
                "last_run_time": now_str,
                "last_run_type": state["last_run_type"],
                "metrics": run_res["metrics_df"].to_dict(orient="records"),
                "slice_metrics": run_res["slice_metrics_df"].to_dict(orient="records"),
                "next_hour_forecasts": run_res["next_hour_forecasts"],
                "capacity_recommendation": run_res["capacity_rec"],
            }
        )
    except Exception as exc:
        state["last_error"] = str(exc)
        return jsonify({"status": "error", "message": f"Execution failed: {str(exc)}"}), 500
    finally:
        state["is_running"] = False


def create_app():
    """Application factory for deployment."""
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

