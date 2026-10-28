"""Rule-based capacity recommendation module for 5G network resource planning.

Implements the rule panel planned in the mid-semester presentation (acn_midsem.pdf Slide 8):
- Evaluates next-hour forecasted traffic demand against an operator-configurable threshold.
- Generates an "Add capacity" hint when demand exceeds threshold.
- Emits a neutral status when demand is nominal.
- Features explicit disclaimers establishing that this is a predictive advisory rule,
  not an active closed-loop network controller actuating physical radio resources.
"""

from typing import Any


DEFAULT_HIGH_TRAFFIC_THRESHOLD_MBPS = 700.0


def evaluate_capacity_recommendation(
    forecasted_mbps: float,
    threshold_mbps: float = DEFAULT_HIGH_TRAFFIC_THRESHOLD_MBPS,
) -> dict[str, Any]:
    """Evaluate whether the next-hour forecasted 5G traffic requires proactive capacity scaling.

    Args:
        forecasted_mbps: Predicted aggregate 5G traffic demand for the next hour (Mbps).
        threshold_mbps: Configured high-traffic threshold in Mbps (default: 700.0 Mbps).

    Returns:
        dict: Structured recommendation containing:
            - status: "ADD_CAPACITY" | "NOMINAL"
            - level: "ALERT" | "NORMAL"
            - badge_color: "warning" | "success"
            - hint: Concise operator headline.
            - explanation: Detailed context for network engineers.
            - forecasted_mbps: float
            - threshold_mbps: float
            - headroom_mbps: float difference (forecast - threshold)
            - disclaimer: Explicit analytical advisory statement.
    """
    forecast_val = float(np_or_float(forecasted_mbps))
    threshold_val = float(np_or_float(threshold_mbps))
    delta = forecast_val - threshold_val

    disclaimer = (
        "Advisory Notice: This recommendation is generated from predictive rule-based heuristics "
        "designed for 5G Non-RT RIC / NWDAF proactive planning. It serves as decision-support guidance "
        "and does not actuate physical network equipment or modify active radio resource controllers."
    )

    if forecast_val >= threshold_val:
        return {
            "status": "ADD_CAPACITY",
            "level": "ALERT",
            "badge_color": "warning",
            "hint": f"Add Capacity: Next-hour forecast ({forecast_val:.1f} Mbps) exceeds threshold ({threshold_val:.1f} Mbps)",
            "explanation": (
                f"Predicted demand of {forecast_val:.1f} Mbps exceeds the operating threshold by {delta:.1f} Mbps. "
                "Recommend proactively expanding Physical Resource Block (PRB) allocation, "
                "initiating carrier aggregation on high-demand sectors, or pre-warming virtualized DU/CU instances."
            ),
            "forecasted_mbps": round(forecast_val, 2),
            "threshold_mbps": round(threshold_val, 2),
            "headroom_mbps": round(delta, 2),
            "is_high_traffic": True,
            "disclaimer": disclaimer,
        }
    else:
        headroom = threshold_val - forecast_val
        return {
            "status": "NOMINAL",
            "level": "NORMAL",
            "badge_color": "success",
            "hint": f"Nominal: Next-hour forecast ({forecast_val:.1f} Mbps) within threshold ({threshold_val:.1f} Mbps)",
            "explanation": (
                f"Predicted demand of {forecast_val:.1f} Mbps remains within normal bounds with {headroom:.1f} Mbps "
                "headroom below the configured threshold. Maintain standard PRB allocation; "
                "no proactive capacity scale-out required for the upcoming hour."
            ),
            "forecasted_mbps": round(forecast_val, 2),
            "threshold_mbps": round(threshold_val, 2),
            "headroom_mbps": round(-headroom, 2),
            "is_high_traffic": False,
            "disclaimer": disclaimer,
        }


def np_or_float(val: Any) -> float:
    """Helper to safely extract float from numpy scalars or python numbers."""
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)

