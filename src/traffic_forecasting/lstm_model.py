"""LSTM deep learning forecasting baseline module for 5G network traffic.

Implements:
1. PyTorch-based sequential LSTM network for 1-hour ahead traffic demand.
2. Leakage-safe sequence formulation (sliding historical window, seq_len=24).
3. Preprocessing scaling fitted ONLY on the training split.
4. Metric evaluation (MAE, RMSE, R2) and 1-hour ahead future prediction.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from traffic_forecasting.model import calculate_metrics

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TrafficLSTM(nn.Module):
        """Single/multi-layer LSTM with linear output projection head."""

        def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            # Take final time step hidden state
            return self.fc(out[:, -1, :])
else:
    class TrafficLSTM:  # type: ignore
        pass


def train_and_evaluate_lstm(
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    seq_len: int = 24,
    hidden_dim: int = 32,
    num_layers: int = 1,
    epochs: int = 25,
    lr: float = 0.01,
    random_seed: int = 42,
) -> dict[str, object]:
    """Train an LSTM network on historical sequences and evaluate on unseen test data.

    To ensure strict temporal separation and zero data leakage:
    1. StandardScaler is fitted strictly on y_train.
    2. y_test is transformed using the train-fitted scaler.
    3. Test sequences for observation t strictly use prior history [t - seq_len : t].
    4. Predictions are inverse-transformed back to the original Mbps scale.

    Args:
        y_train: Chronological training series of total traffic demand.
        y_test: Chronological test series of total traffic demand.
        seq_len: Number of historical hours in the input sequence (default: 24).
        hidden_dim: Hidden dimension of LSTM cells (default: 32).
        num_layers: Number of stacked LSTM layers (default: 1).
        epochs: Number of training epochs (default: 25).
        lr: Adam learning rate (default: 0.01).
        random_seed: Random seed for reproducibility.

    Returns:
        dict: {
            "model": trained TrafficLSTM instance,
            "predictions": np.ndarray of 1-step ahead test predictions,
            "next_hour_forecast": float for the 1-hour ahead forecast,
            "metrics": dict with 'mae', 'rmse', 'r2',
            "seq_len": int,
            "epochs": int,
            "status": "SUCCESS" or error message,
        }
    """
    if not TORCH_AVAILABLE:
        return {
            "model": None,
            "predictions": np.asarray(y_test, dtype=float),
            "next_hour_forecast": float(y_test.iloc[-1] if hasattr(y_test, "iloc") else y_test[-1]),
            "metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0},
            "seq_len": seq_len,
            "epochs": epochs,
            "status": "FAILED: PyTorch is not installed in the environment.",
        }

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    train_arr = np.asarray(y_train, dtype=float).reshape(-1, 1)
    test_arr = np.asarray(y_test, dtype=float).reshape(-1, 1)

    # 1. Scaling fitted ONLY on training data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_arr).flatten()
    test_scaled = scaler.transform(test_arr).flatten()

    # 2. Build training sequences
    X_train_list, y_train_list = [], []
    for i in range(seq_len, len(train_scaled)):
        X_train_list.append(train_scaled[i - seq_len : i])
        y_train_list.append(train_scaled[i])

    X_train_t = torch.tensor(np.array(X_train_list), dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(np.array(y_train_list), dtype=torch.float32).unsqueeze(-1)

    # 3. Build test sequences using preceding historical context (zero leakage)
    full_scaled = np.concatenate([train_scaled, test_scaled])
    split_idx = len(train_scaled)
    X_test_list = []
    for i in range(split_idx, len(full_scaled)):
        X_test_list.append(full_scaled[i - seq_len : i])

    X_test_t = torch.tensor(np.array(X_test_list), dtype=torch.float32).unsqueeze(-1)

    # 4. Initialize model, optimizer, loss
    model = TrafficLSTM(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 5. Training loop
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

    # 6. Out-of-sample inference on test set
    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_t).numpy()
    test_preds = scaler.inverse_transform(test_preds_scaled).flatten()

    # 7. Next-hour forecast
    last_window = full_scaled[-seq_len:]
    last_window_t = torch.tensor(last_window, dtype=torch.float32).reshape(1, seq_len, 1)
    with torch.no_grad():
        next_scaled = model(last_window_t).numpy()
    next_hour_forecast = float(np.round(scaler.inverse_transform(next_scaled)[0, 0], 2))

    metrics = calculate_metrics(test_arr.flatten(), test_preds)

    return {
        "model": model,
        "predictions": np.round(test_preds, 2),
        "next_hour_forecast": next_hour_forecast,
        "metrics": metrics,
        "seq_len": seq_len,
        "epochs": epochs,
        "status": "SUCCESS",
    }

