"""
FlowCast v2 — Deep Learning Models
LSTM with PSO, Physics-Informed LSTM (Paper 1), Adapted-GRU (Paper 3)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
#  Standard LSTM
# ─────────────────────────────────────────────────────────────
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  Physics-Informed LSTM (Paper 1: Xie et al. 2021 approach)
# ─────────────────────────────────────────────────────────────
class PhysicsInformedLSTM(nn.Module):
    """
    LSTM with physics constraints in the loss function:
    1. Water balance: P - ET - Q ≈ ΔS
    2. Monotonicity: higher rainfall → higher discharge
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class PhysicsLoss(nn.Module):
    """
    Combined loss = MSE + λ_wb * water_balance_penalty + λ_mono * monotonicity_penalty
    """
    def __init__(self, lambda_wb: float = 0.1, lambda_mono: float = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_wb = lambda_wb
        self.lambda_mono = lambda_mono

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        precip_sum: Optional[torch.Tensor] = None,
        et_sum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_mse = self.mse(pred, target)

        # Water balance constraint: P - ET - Q should be non-negative (storage)
        loss_wb = torch.tensor(0.0, device=pred.device)
        if precip_sum is not None and et_sum is not None:
            water_balance_residual = precip_sum - et_sum - pred
            # Penalize large negative residuals (violates mass conservation)
            loss_wb = torch.mean(torch.relu(-water_balance_residual) ** 2)

        # Monotonicity constraint: if precip_a > precip_b, then Q_a >= Q_b
        loss_mono = torch.tensor(0.0, device=pred.device)
        if precip_sum is not None and len(pred) > 1:
            # Sample random pairs
            n = min(len(pred), 128)
            idx_a = torch.randint(0, len(pred), (n,))
            idx_b = torch.randint(0, len(pred), (n,))
            p_diff = precip_sum[idx_a] - precip_sum[idx_b]
            q_diff = pred[idx_a] - pred[idx_b]
            # If precip_a > precip_b but Q_a < Q_b, penalize
            violations = torch.relu(-q_diff * torch.sign(p_diff))
            loss_mono = torch.mean(violations ** 2)

        return loss_mse + self.lambda_wb * loss_wb + self.lambda_mono * loss_mono


# ─────────────────────────────────────────────────────────────
#  Adapted GRU (Paper 3: Dhakal et al. 2020)
# ─────────────────────────────────────────────────────────────
class AdaptedGRUCell(nn.Module):
    """
    A-GRU cell that processes static (x_s) and dynamic (x_d) inputs separately.
    Equations from Paper 3:
        i = σ(W_i · x_s + b_i)               — input gate from static features
        r[t] = σ(W_r · x_d[t] + U_r · h[t-1] + b_r)   — reset gate
        g[t] = tanh(W_g · x_d[t] + U_g · h[t-1] + b_g) — candidate
        c[t] = (1 - r[t]) * c[t-1] + r[t] * i * g[t]   — cell state
        h[t] = c[t]
    """
    def __init__(self, dynamic_size: int, static_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Input gate (static only)
        self.W_i = nn.Linear(static_size, hidden_size)

        # Reset gate (dynamic + recurrent)
        self.W_r = nn.Linear(dynamic_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate gate (dynamic + recurrent)
        self.W_g = nn.Linear(dynamic_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor, h_prev: torch.Tensor):
        """
        Args:
            x_d: (batch, dynamic_size) — dynamic input at time t
            x_s: (batch, static_size) — static basin attributes
            h_prev: (batch, hidden_size)
        """
        i = torch.sigmoid(self.W_i(x_s))
        r = torch.sigmoid(self.W_r(x_d) + self.U_r(h_prev))
        g = torch.tanh(self.W_g(x_d) + self.U_g(h_prev))
        c = (1 - r) * h_prev + r * i * g
        h = c
        return h


class AdaptedGRU(nn.Module):
    """
    Full A-GRU model with separate static/dynamic input processing.
    Paper 3 architecture adapted for single-basin prediction.
    """
    def __init__(
        self,
        dynamic_size: int,
        static_size: int,
        hidden_size: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = AdaptedGRUCell(dynamic_size, static_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor):
        """
        Args:
            x_d: (batch, seq_len, dynamic_size)
            x_s: (batch, static_size)
        """
        batch_size, seq_len, _ = x_d.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x_d.device)

        for t in range(seq_len):
            h = self.cell(x_d[:, t, :], x_s, h)
            h = self.dropout(h)

        out = self.fc(h)
        return out.squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  Training utilities
# ─────────────────────────────────────────────────────────────
class DeepModelTrainer:
    """Unified trainer for all deep learning models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        patience: int = 15,
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.patience = patience
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        loss_fn: nn.Module = None,
        physics_mode: bool = False,
        precip_idx: int = 0,
        et_idx: int = -1,
    ) -> Dict:
        """Train with early stopping."""
        loss_fn = loss_fn or nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Training {self.model_name} | device={device}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                self.optimizer.zero_grad()

                if len(batch) == 3:  # A-GRU: (x_d, x_s, y)
                    x_d, x_s, y = [b.to(device) for b in batch]
                    pred = self.model(x_d, x_s)
                elif len(batch) == 2:  # Standard LSTM: (x, y)
                    x, y = [b.to(device) for b in batch]
                    pred = self.model(x)
                    x_d = x  # for physics loss

                if physics_mode and isinstance(loss_fn, PhysicsLoss):
                    precip_sum = x_d[:, :, precip_idx].sum(dim=1)
                    if et_idx >= 0:
                        et_sum = x_d[:, :, et_idx].sum(dim=1)
                    else:
                        et_sum = None
                    loss = loss_fn(pred, y, precip_sum, et_sum)
                else:
                    loss = loss_fn(pred, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            self.train_losses.append(train_loss)

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x_d, x_s, y = [b.to(device) for b in batch]
                        pred = self.model(x_d, x_s)
                    else:
                        x, y = [b.to(device) for b in batch]
                        pred = self.model(x)
                    val_loss += nn.MSELoss()(pred, y).item()
                    n_val += 1
            val_loss /= max(n_val, 1)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f} "
                      f"val_loss={val_loss:.6f} lr={lr:.6f}")

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
        }

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Predict on a DataLoader."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    x_d, x_s, _ = [b.to(device) for b in batch]
                    pred = self.model(x_d, x_s)
                elif len(batch) == 2:
                    x, _ = [b.to(device) for b in batch]
                    pred = self.model(x)
                else:
                    x = batch[0].to(device)
                    pred = self.model(x)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    def save(self, path: Path):
        torch.save({
            "model_state": self.model.state_dict(),
            "model_name": self.model_name,
        }, path)


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader from numpy arrays."""
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_agru_dataloader(
    X_d: np.ndarray,
    X_s: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for A-GRU with static features."""
    Xd_t = torch.FloatTensor(X_d)
    Xs_t = torch.FloatTensor(X_s)
    y_t = torch.FloatTensor(y)
    dataset = TensorDataset(Xd_t, Xs_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)