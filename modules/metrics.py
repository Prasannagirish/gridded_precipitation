"""
FlowCast v2 — Evaluation Metrics
NSE, KGE, RMSE, MAE, R², PBIAS — standard hydrological evaluation suite.
"""
import numpy as np
from typing import Dict


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency. Perfect = 1.0"""
    num = np.sum((observed - predicted) ** 2)
    den = np.sum((observed - np.mean(observed)) ** 2)
    return 1.0 - num / (den + 1e-10)


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Kling-Gupta Efficiency. Perfect = 1.0"""
    r = np.corrcoef(observed, predicted)[0, 1]
    alpha = np.std(predicted) / (np.std(observed) + 1e-10)
    beta = np.mean(predicted) / (np.mean(observed) + 1e-10)
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Square Error."""
    return np.sqrt(np.mean((observed - predicted) ** 2))


def mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(observed - predicted))


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def pbias(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Percent Bias. Perfect = 0. Positive = overestimation."""
    return 100.0 * np.sum(predicted - observed) / (np.sum(observed) + 1e-10)


def pearson_r(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return np.corrcoef(observed, predicted)[0, 1]


def evaluate_all(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute all metrics."""
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()
    return {
        "NSE": round(nse(obs, pred), 4),
        "KGE": round(kge(obs, pred), 4),
        "RMSE": round(rmse(obs, pred), 4),
        "MAE": round(mae(obs, pred), 4),
        "R2": round(r_squared(obs, pred), 4),
        "R": round(pearson_r(obs, pred), 4),
        "PBIAS": round(pbias(obs, pred), 2),
    }


def evaluate_flow_regimes(
    observed: np.ndarray,
    predicted: np.ndarray,
    low_threshold: float = 0.2,
    high_threshold: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model performance across flow regimes."""
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()

    low_q = np.quantile(obs, low_threshold)
    high_q = np.quantile(obs, high_threshold)

    low_mask = obs <= low_q
    mid_mask = (obs > low_q) & (obs <= high_q)
    high_mask = obs > high_q

    results = {}
    for name, mask in [("low_flow", low_mask), ("mid_flow", mid_mask), ("high_flow", high_mask)]:
        if mask.sum() > 10:
            results[name] = evaluate_all(obs[mask], pred[mask])
        else:
            results[name] = {"NSE": np.nan, "KGE": np.nan}
    return results