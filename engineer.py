"""
FlowCast v2 — Feature Engineering
Computes lag features, rolling statistics, SPI, temporal encodings,
and antecedent precipitation indices.
"""
import numpy as np
import pandas as pd
from typing import List, Optional


class HydroFeatureEngineer:
    """Compute hydrological features for the Kabini basin."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def build_all_features(self) -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        self._lag_features()
        self._rolling_features()
        self._temporal_encodings()
        self._antecedent_precipitation()
        self._spi_features()
        self._temperature_features()
        self._et_estimate()
        self.df = self.df.dropna()
        return self.df

    def _lag_features(self):
        """Precipitation and discharge lag features."""
        for lag in [1, 2, 3, 5, 7, 14]:
            self.df[f"precip_lag{lag}"] = self.df["precip"].shift(lag)
        for lag in [1, 2, 3, 5, 7]:
            self.df[f"discharge_lag{lag}"] = self.df["discharge"].shift(lag)

    def _rolling_features(self):
        """Rolling window statistics."""
        for window in [3, 7, 14, 30]:
            self.df[f"precip_rolling{window}"] = (
                self.df["precip"].rolling(window).mean()
            )
        for window in [3, 7, 14]:
            self.df[f"precip_std{window}"] = (
                self.df["precip"].rolling(window).std()
            )
        for window in [7, 14, 30]:
            self.df[f"discharge_rolling{window}"] = (
                self.df["discharge"].rolling(window).mean()
            )

    def _temporal_encodings(self):
        """Cyclical encoding of time features."""
        dates = self.df.index
        doy = dates.dayofyear
        month = dates.month

        self.df["month_sin"] = np.sin(2 * np.pi * month / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * month / 12)
        self.df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        self.df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

        # Monsoon indicator (June-September)
        self.df["is_monsoon"] = ((month >= 6) & (month <= 9)).astype(float)

    def _antecedent_precipitation(self):
        """Antecedent Precipitation Index (API)."""
        for window in [5, 10, 20, 30]:
            self.df[f"antecedent_precip_{window}"] = (
                self.df["precip"].rolling(window).sum()
            )

    def _spi_features(self):
        """Standardized Precipitation Index (simplified)."""
        for window in [30, 60, 90]:
            rolling_sum = self.df["precip"].rolling(window).sum()
            mean = rolling_sum.rolling(365, min_periods=30).mean()
            std = rolling_sum.rolling(365, min_periods=30).std()
            self.df[f"spi_{window}"] = (rolling_sum - mean) / (std + 1e-8)

    def _temperature_features(self):
        """Temperature-derived features."""
        self.df["temp_range"] = self.df["tmax"] - self.df["tmin"]
        self.df["temp_mean"] = (self.df["tmax"] + self.df["tmin"]) / 2

    def _et_estimate(self):
        """Hargreaves ET estimate for water balance constraint."""
        tmean = (self.df["tmax"] + self.df["tmin"]) / 2
        trange = self.df["tmax"] - self.df["tmin"]
        # Simplified Hargreaves
        doy = self.df.index.dayofyear
        # Extraterrestrial radiation approximation (for ~12°N latitude)
        ra = 15.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365)
        self.df["et_hargreaves"] = (
            0.0023 * ra * (tmean + 17.8) * np.sqrt(np.maximum(trange, 0))
        )


def select_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Select feature subset, handling missing columns gracefully."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  Warning: features not found: {missing}")
    return df[available]


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int = 30,
) -> tuple:
    """Create sequences for RNN models."""
    Xs, ys = [], []
    for i in range(seq_length, len(X)):
        Xs.append(X[i - seq_length : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def create_sequences_with_static(
    X_dynamic: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    seq_length: int = 30,
) -> tuple:
    """Create sequences for A-GRU with separate static/dynamic inputs (Paper 3)."""
    Xd, Xs, ys = [], [], []
    for i in range(seq_length, len(X_dynamic)):
        Xd.append(X_dynamic[i - seq_length : i])
        Xs.append(X_static)  # static features repeated per sample
        ys.append(y[i])
    return np.array(Xd), np.array(Xs), np.array(ys)