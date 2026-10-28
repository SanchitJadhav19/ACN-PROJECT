#!/usr/bin/env python3
"""Launcher script for the 5G Network Traffic Analytics & Forecasting Web UI Dashboard."""

import argparse
import os
import socket
import sys

# Ensure src is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from traffic_forecasting.app import app


def is_port_available(host: str, port: int) -> bool:
    """Check if the specified host and port can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(host: str, preferred_port: int) -> int:
    """Return preferred_port if free, or search common fallback ports."""
    candidate_ports = [preferred_port, 5050, 8080, 5005, 5001]
    for p in candidate_ports:
        if is_port_available(host, p):
            return p
    return preferred_port


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch 5G Network Data Analytics Web UI Dashboard."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to listen on (default: 5050).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in Flask debug mode.",
    )

    args = parser.parse_args()
    port = find_available_port(args.host, args.port)

    print("=" * 70)
    print("5G NETWORK DATA ANALYTICS — WEB UI DASHBOARD")
    print("=" * 70)
    print(f"Launching dashboard at: http://{args.host}:{port}")
    print("Press CTRL+C to terminate the web server.")
    print("=" * 70)

    app.run(host=args.host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
