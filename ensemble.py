"""
FlowCast v2 — Ensemble Methods
Stacking, weighted averaging, and adaptive weighting (Paper 2: Yao et al. approach)
"""
import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, List
from metrics import nse, kge


class WeightedEnsemble:
    """Optimized weighted average of model predictions."""

    def __init__(self):
        self.weights = None
        self.model_names = []

    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        observed: np.ndarray,
        method: str = "nse_weighted",
    ):
        """
        Compute optimal weights based on validation performance.

        Methods:
            - "equal": uniform weights
            - "nse_weighted": weight proportional to NSE
            - "optimize": grid search for best NSE
        """
        self.model_names = list(predictions.keys())
        n_models = len(self.model_names)
        pred_matrix = np.column_stack([predictions[m] for m in self.model_names])

        if method == "equal":
            self.weights = np.ones(n_models) / n_models

        elif method == "nse_weighted":
            scores = []
            for m in self.model_names:
                s = max(nse(observed, predictions[m]), 0.0)
                scores.append(s)
            total = sum(scores) + 1e-10
            self.weights = np.array(scores) / total

        elif method == "optimize":
            # Grid search over weight space
            best_nse = -np.inf
            best_w = np.ones(n_models) / n_models
            # Simplex sampling
            for _ in range(5000):
                w = np.random.dirichlet(np.ones(n_models))
                ensemble_pred = pred_matrix @ w
                score = nse(observed, ensemble_pred)
                if score > best_nse:
                    best_nse = score
                    best_w = w
            self.weights = best_w

        print(f"\n  Ensemble weights ({method}):")
        for name, w in zip(self.model_names, self.weights):
            print(f"    {name}: {w:.4f}")

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate ensemble prediction."""
        pred_matrix = np.column_stack([predictions[m] for m in self.model_names])
        return pred_matrix @ self.weights


class StackingEnsemble:
    """Stacking ensemble with Ridge meta-learner."""

    def __init__(self, alpha: float = 1.0):
        self.meta_model = Ridge(alpha=alpha)
        self.model_names = []

    def fit(self, predictions: Dict[str, np.ndarray], observed: np.ndarray):
        self.model_names = list(predictions.keys())
        X_meta = np.column_stack([predictions[m] for m in self.model_names])
        self.meta_model.fit(X_meta, observed)
        print(f"\n  Stacking coefficients:")
        for name, coef in zip(self.model_names, self.meta_model.coef_):
            print(f"    {name}: {coef:.4f}")

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        X_meta = np.column_stack([predictions[m] for m in self.model_names])
        return self.meta_model.predict(X_meta)