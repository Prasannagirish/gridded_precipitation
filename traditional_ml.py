"""
FlowCast v2 — Traditional ML Models with PSO Optimization
RF-PSO, SVR-PSO, XGBoost (Paper 2: Eatesam et al.)
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Optional, Tuple
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pso import PSOOptimizer
from metrics import nse


class BaseMLModel:
    """Base class for traditional ML models."""

    def __init__(self, name: str, scaler_X=None, scaler_y=None):
        self.name = name
        self.model = None
        self.scaler_X = scaler_X or StandardScaler()
        self.scaler_y = scaler_y or StandardScaler()
        self.is_fitted = False
        self.best_params = {}

    def fit(self, X_train, y_train):
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def save(self, path: Path):
        joblib.dump({
            "model": self.model,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "best_params": self.best_params,
        }, path)

    def load(self, path: Path):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler_X = data["scaler_X"]
        self.scaler_y = data["scaler_y"]
        self.best_params = data["best_params"]
        self.is_fitted = True


class RFModel(BaseMLModel):
    """Random Forest with optional PSO optimization."""

    def __init__(self, random_state: int = 42):
        super().__init__("RF-PSO")
        self.random_state = random_state

    def optimize_and_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_bounds: Dict = None,
        n_particles: int = 15,
        n_iterations: int = 30,
        n_cv_folds: int = 3,
    ) -> Dict:
        """PSO optimization with time-series cross-validation."""
        param_bounds = param_bounds or {
            "n_estimators": (50, 400),
            "max_depth": (3, 25),
            "min_samples_split": (2, 15),
            "min_samples_leaf": (1, 8),
            "max_features": (0.3, 1.0),
        }
        param_types = {
            "n_estimators": "int",
            "max_depth": "int",
            "min_samples_split": "int",
            "min_samples_leaf": "int",
        }

        X_s = self.scaler_X.fit_transform(X_train)
        y_s = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        tscv = TimeSeriesSplit(n_splits=n_cv_folds)

        def objective(params):
            scores = []
            for train_idx, val_idx in tscv.split(X_s):
                rf = RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    max_features=params["max_features"],
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_s[train_idx], y_s[train_idx])
                pred = rf.predict(X_s[val_idx])
                # Inverse transform for NSE calculation
                pred_real = self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
                obs_real = self.scaler_y.inverse_transform(y_s[val_idx].reshape(-1, 1)).ravel()
                scores.append(nse(obs_real, pred_real))
            return np.mean(scores)

        print(f"\n{'='*60}")
        print(f"Optimizing RF with PSO ({n_particles} particles, {n_iterations} iters)")
        print(f"{'='*60}")

        pso = PSOOptimizer(
            n_particles=n_particles,
            n_iterations=n_iterations,
            verbose=True,
        )
        best_params, best_score, history = pso.optimize(
            objective, param_bounds, param_types
        )

        self.best_params = best_params
        self.model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_s, y_s)
        self.is_fitted = True

        print(f"\n  Best RF params: {best_params}")
        print(f"  Best CV NSE: {best_score:.4f}")
        return {"params": best_params, "cv_score": best_score, "history": history}


class SVRModel(BaseMLModel):
    """Support Vector Regression with PSO optimization."""

    def __init__(self):
        super().__init__("SVR-PSO")

    def optimize_and_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_bounds: Dict = None,
        n_particles: int = 15,
        n_iterations: int = 30,
        n_cv_folds: int = 3,
    ) -> Dict:
        param_bounds = param_bounds or {
            "C": (0.1, 500.0),
            "epsilon": (0.001, 0.5),
            "gamma": (0.0001, 0.5),
        }

        X_s = self.scaler_X.fit_transform(X_train)
        y_s = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        tscv = TimeSeriesSplit(n_splits=n_cv_folds)

        def objective(params):
            scores = []
            for train_idx, val_idx in tscv.split(X_s):
                svr = SVR(
                    kernel="rbf",
                    C=params["C"],
                    epsilon=params["epsilon"],
                    gamma=params["gamma"],
                )
                svr.fit(X_s[train_idx], y_s[train_idx])
                pred = svr.predict(X_s[val_idx])
                pred_real = self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
                obs_real = self.scaler_y.inverse_transform(y_s[val_idx].reshape(-1, 1)).ravel()
                scores.append(nse(obs_real, pred_real))
            return np.mean(scores)

        print(f"\n{'='*60}")
        print(f"Optimizing SVR with PSO ({n_particles} particles, {n_iterations} iters)")
        print(f"{'='*60}")

        pso = PSOOptimizer(
            n_particles=n_particles,
            n_iterations=n_iterations,
            verbose=True,
        )
        best_params, best_score, history = pso.optimize(param_bounds=param_bounds, objective_fn=objective)

        self.best_params = best_params
        self.model = SVR(
            kernel="rbf",
            C=best_params["C"],
            epsilon=best_params["epsilon"],
            gamma=best_params["gamma"],
        )
        self.model.fit(X_s, y_s)
        self.is_fitted = True

        print(f"\n  Best SVR params: {best_params}")
        print(f"  Best CV NSE: {best_score:.4f}")
        return {"params": best_params, "cv_score": best_score, "history": history}


class XGBoostModel(BaseMLModel):
    """XGBoost with PSO optimization (from original FlowCast)."""

    def __init__(self, random_state: int = 42):
        super().__init__("XGBoost-PSO")
        self.random_state = random_state

    def optimize_and_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_particles: int = 15,
        n_iterations: int = 30,
        n_cv_folds: int = 3,
    ) -> Dict:
        param_bounds = {
            "n_estimators": (50, 500),
            "max_depth": (3, 15),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.5, 1.0),
            "min_samples_split": (2, 20),
        }
        param_types = {
            "n_estimators": "int",
            "max_depth": "int",
            "min_samples_split": "int",
        }

        X_s = self.scaler_X.fit_transform(X_train)
        y_s = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        tscv = TimeSeriesSplit(n_splits=n_cv_folds)

        def objective(params):
            scores = []
            for train_idx, val_idx in tscv.split(X_s):
                model = GradientBoostingRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    subsample=params["subsample"],
                    min_samples_split=params["min_samples_split"],
                    random_state=self.random_state,
                )
                model.fit(X_s[train_idx], y_s[train_idx])
                pred = model.predict(X_s[val_idx])
                pred_real = self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
                obs_real = self.scaler_y.inverse_transform(y_s[val_idx].reshape(-1, 1)).ravel()
                scores.append(nse(obs_real, pred_real))
            return np.mean(scores)

        print(f"\n{'='*60}")
        print(f"Optimizing XGBoost with PSO")
        print(f"{'='*60}")

        pso = PSOOptimizer(n_particles=n_particles, n_iterations=n_iterations, verbose=True)
        best_params, best_score, history = pso.optimize(objective, param_bounds, param_types)

        self.best_params = best_params
        self.model = GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            min_samples_split=best_params["min_samples_split"],
            random_state=self.random_state,
        )
        self.model.fit(X_s, y_s)
        self.is_fitted = True

        return {"params": best_params, "cv_score": best_score, "history": history}