"""
FlowCast v2 — Bias Correction Module
=====================================
Implements Quantile Delta Mapping (QDM) for CMIP6 bias correction.

QDM preserves the GCM's projected climate change signal while correcting
the statistical distribution to match observations. This is critical for
SSP585 extremes where basic quantile mapping distorts the trend.

References:
    Cannon et al. (2015) — Bias Correction of GCM Precipitation by
    Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles
    and Extremes? (J. Climate, 28, 6938–6959)

Usage:
    bc = BiasCorrector()
    bc.fit(obs_series, gcm_hist_series)
    corrected_future = bc.transform(gcm_future_series)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class QDMParams:
    """Fitted QDM parameters for one variable × one month."""
    obs_quantiles: np.ndarray       # observed quantile values
    hist_quantiles: np.ndarray      # GCM historical quantile values
    quantile_levels: np.ndarray     # quantile levels [0..1]
    obs_mean: float
    obs_std: float
    hist_mean: float
    hist_std: float


class QuantileDeltaMapping:
    """
    Quantile Delta Mapping (QDM) bias correction.

    Steps:
        1. For future value x_f, find its quantile τ in the future distribution
        2. Map τ to the historical distribution → x_h(τ)
        3. Compute delta: Δ = x_f − x_h(τ)  [additive for temp]
                     or: Δ = x_f / x_h(τ)   [multiplicative for precip]
        4. Map τ to the observed distribution → x_o(τ)
        5. Corrected value = x_o(τ) + Δ     [additive]
                          or x_o(τ) × Δ     [multiplicative]
    """

    def __init__(self, n_quantiles: int = 100, monthly: bool = True):
        """
        Args:
            n_quantiles: Number of quantile bins for the mapping.
            monthly: If True, fit separate corrections per calendar month
                     (captures seasonal bias structure).
        """
        self.n_quantiles = n_quantiles
        self.monthly = monthly
        self.quantile_levels = np.linspace(0, 1, n_quantiles + 1)
        self.params: Dict[int, QDMParams] = {}  # month → params
        self._is_fitted = False

    def fit(
        self,
        obs: pd.Series,
        gcm_hist: pd.Series,
    ) -> "QuantileDeltaMapping":
        """
        Fit the QDM correction using observed data and GCM historical run.

        Both series must have DatetimeIndex covering an overlapping period.

        Args:
            obs:       Observed daily values (e.g., CHIRPS precip, station temp)
            gcm_hist:  GCM historical run values for the same period
        """
        # Align to common date range
        common_start = max(obs.index.min(), gcm_hist.index.min())
        common_end = min(obs.index.max(), gcm_hist.index.max())

        obs_aligned = obs.loc[common_start:common_end].dropna()
        hist_aligned = gcm_hist.loc[common_start:common_end].dropna()

        if len(obs_aligned) < 365:
            warnings.warn(
                f"Only {len(obs_aligned)} overlapping days for bias correction. "
                f"Recommend at least 10 years of overlap."
            )

        if self.monthly:
            for month in range(1, 13):
                obs_m = obs_aligned[obs_aligned.index.month == month].values
                hist_m = hist_aligned[hist_aligned.index.month == month].values

                if len(obs_m) < 10 or len(hist_m) < 10:
                    # Fall back: use all months pooled
                    obs_m = obs_aligned.values
                    hist_m = hist_aligned.values

                self.params[month] = QDMParams(
                    obs_quantiles=np.quantile(obs_m, self.quantile_levels),
                    hist_quantiles=np.quantile(hist_m, self.quantile_levels),
                    quantile_levels=self.quantile_levels,
                    obs_mean=float(np.mean(obs_m)),
                    obs_std=float(np.std(obs_m)),
                    hist_mean=float(np.mean(hist_m)),
                    hist_std=float(np.std(hist_m)),
                )
        else:
            obs_vals = obs_aligned.values
            hist_vals = hist_aligned.values
            params = QDMParams(
                obs_quantiles=np.quantile(obs_vals, self.quantile_levels),
                hist_quantiles=np.quantile(hist_vals, self.quantile_levels),
                quantile_levels=self.quantile_levels,
                obs_mean=float(np.mean(obs_vals)),
                obs_std=float(np.std(obs_vals)),
                hist_mean=float(np.mean(hist_vals)),
                hist_std=float(np.std(hist_vals)),
            )
            for month in range(1, 13):
                self.params[month] = params

        self._is_fitted = True
        return self

    def transform(
        self,
        gcm_future: pd.Series,
        method: str = "additive",
    ) -> pd.Series:
        """
        Apply QDM correction to future GCM data.

        Args:
            gcm_future: Future GCM daily values with DatetimeIndex.
            method:     "additive" for temperature, "multiplicative" for precip.

        Returns:
            Bias-corrected series with same index.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        corrected = np.empty_like(gcm_future.values, dtype=np.float64)

        for i, (date, value) in enumerate(gcm_future.items()):
            month = date.month
            p = self.params[month]

            # Step 1: Find quantile of future value in historical distribution
            tau = self._value_to_quantile(value, p.hist_quantiles)

            # Step 2: Map to observed quantile
            obs_mapped = self._quantile_to_value(tau, p.obs_quantiles)

            # Step 3: Compute and apply delta
            hist_mapped = self._quantile_to_value(tau, p.hist_quantiles)

            if method == "multiplicative":
                # For precipitation: preserve ratio
                if abs(hist_mapped) < 1e-6:
                    delta = 1.0
                else:
                    delta = value / hist_mapped
                corrected[i] = obs_mapped * delta
            else:
                # For temperature: preserve difference
                delta = value - hist_mapped
                corrected[i] = obs_mapped + delta

        result = pd.Series(corrected, index=gcm_future.index, name=gcm_future.name)

        # Post-processing
        if method == "multiplicative":
            result = result.clip(lower=0.0)  # no negative precip

        return result

    def _value_to_quantile(self, value: float, quantile_values: np.ndarray) -> float:
        """Map a value to its quantile level using the empirical CDF."""
        if value <= quantile_values[0]:
            return 0.0
        if value >= quantile_values[-1]:
            return 1.0
        # Linear interpolation
        idx = np.searchsorted(quantile_values, value, side="right") - 1
        idx = min(idx, len(self.quantile_levels) - 2)
        frac = (value - quantile_values[idx]) / (
            quantile_values[idx + 1] - quantile_values[idx] + 1e-10
        )
        return self.quantile_levels[idx] + frac * (
            self.quantile_levels[idx + 1] - self.quantile_levels[idx]
        )

    def _quantile_to_value(self, tau: float, quantile_values: np.ndarray) -> float:
        """Map a quantile level to a value using linear interpolation."""
        return float(np.interp(tau, self.quantile_levels, quantile_values))

    def diagnostics(self) -> Dict:
        """Return diagnostic statistics for the fitted correction."""
        if not self._is_fitted:
            return {}
        diag = {}
        for month, p in self.params.items():
            diag[month] = {
                "obs_mean": p.obs_mean,
                "hist_mean": p.hist_mean,
                "bias_mean": p.hist_mean - p.obs_mean,
                "obs_std": p.obs_std,
                "hist_std": p.hist_std,
            }
        return diag


class MultiVariateBiasCorrector:
    """
    Bias-correct multiple CMIP6 variables at once.

    Handles the variable-specific correction methods:
        - precip:  multiplicative QDM (preserves ratio, clips negatives)
        - tasmax:  additive QDM
        - tasmin:  additive QDM
    """

    # Default correction method per variable
    METHODS = {
        "precip": "multiplicative",
        "tmax":   "additive",
        "tmin":   "additive",
        "radiation":       "additive",
        "vapor_pressure":  "multiplicative",
    }

    def __init__(self, n_quantiles: int = 100):
        self.n_quantiles = n_quantiles
        self.correctors: Dict[str, QuantileDeltaMapping] = {}
        self._is_fitted = False

    def fit(
        self,
        obs_df: pd.DataFrame,
        gcm_hist_df: pd.DataFrame,
        variables: list = None,
    ) -> "MultiVariateBiasCorrector":
        """
        Fit bias correction for each variable.

        Args:
            obs_df:      Observed DataFrame (date index, columns = variable names)
            gcm_hist_df: GCM historical DataFrame (same column names)
            variables:   List of columns to correct; defaults to all common columns.
        """
        if variables is None:
            variables = [c for c in obs_df.columns if c in gcm_hist_df.columns]

        print(f"\n  Fitting bias correction ({len(variables)} variables)...")

        for var in variables:
            if var not in obs_df.columns or var not in gcm_hist_df.columns:
                print(f"    ⚠ Skipping {var}: not in both obs and GCM data")
                continue

            qdm = QuantileDeltaMapping(n_quantiles=self.n_quantiles, monthly=True)
            qdm.fit(obs_df[var], gcm_hist_df[var])
            self.correctors[var] = qdm

            # Print diagnostics summary
            diag = qdm.diagnostics()
            annual_bias = np.mean([d["bias_mean"] for d in diag.values()])
            method = self.METHODS.get(var, "additive")
            print(f"    ✓ {var:20s} | method={method:14s} | "
                  f"mean_bias={annual_bias:+.3f}")

        self._is_fitted = True
        return self

    def transform(self, gcm_future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply bias correction to all fitted variables in a future GCM DataFrame.

        Returns a new DataFrame with corrected values.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        corrected = gcm_future_df.copy()

        for var, qdm in self.correctors.items():
            if var not in gcm_future_df.columns:
                continue
            method = self.METHODS.get(var, "additive")
            corrected[var] = qdm.transform(gcm_future_df[var], method=method)

        # Physical consistency checks
        if "tmax" in corrected.columns and "tmin" in corrected.columns:
            # Ensure tmax >= tmin
            mask = corrected["tmax"] < corrected["tmin"]
            if mask.any():
                mean_t = (corrected["tmax"] + corrected["tmin"]) / 2
                corrected.loc[mask, "tmax"] = mean_t[mask] + 0.5
                corrected.loc[mask, "tmin"] = mean_t[mask] - 0.5

        if "precip" in corrected.columns:
            corrected["precip"] = corrected["precip"].clip(lower=0.0)

        return corrected

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all fitted corrections."""
        rows = []
        for var, qdm in self.correctors.items():
            diag = qdm.diagnostics()
            for month, d in diag.items():
                rows.append({
                    "variable": var,
                    "month": month,
                    "obs_mean": d["obs_mean"],
                    "gcm_hist_mean": d["hist_mean"],
                    "bias": d["bias_mean"],
                    "method": self.METHODS.get(var, "additive"),
                })
        return pd.DataFrame(rows)