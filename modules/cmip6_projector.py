"""
FlowCast v2 — CMIP6 Real-Data Projector
=========================================
Replaces synthetic delta-change with actual bias-corrected CMIP6 data.

Pipeline:
    1. Load downloaded CMIP6 CSVs (from cmip6.py downloader)
    2. Load observed data for bias correction calibration
    3. Fit Quantile Delta Mapping per variable per GCM
    4. Apply correction to future SSP data
    5. Engineer features from corrected climate data
    6. Predict future discharge using recursive Q-lag infilling

This module is imported by main.py as a drop-in replacement for the
old synthetic CMIP6Projector.

Usage:
    projector = CMIP6RealProjector(
        cmip6_dir="cmip6_downloads",
        obs_df=raw_df,                    # observed data with precip, tmax, tmin, discharge
        feature_columns=feature_list,     # feature columns used by the trained model
    )
    projector.fit_bias_correction()
    future_scenarios = projector.generate_all_scenarios(trained_model)
    summary = projector.summarize_projections(future_scenarios)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import json

from modules.bias_correction import MultiVariateBiasCorrector


class CMIP6RealProjector:
    """
    Climate projection engine using real CMIP6 GCM data.

    Replaces the synthetic delta-change approach with:
        - Actual downloaded CMIP6 NetCDF → CSV data
        - Quantile Delta Mapping bias correction
        - Full feature engineering on corrected climate
        - Recursive prediction for discharge lag features
    """

    # Variables that we bias-correct
    CLIMATE_VARS = ["precip", "tmax", "tmin"]
    # Optional extra variables (used if available in both obs and GCM)
    OPTIONAL_VARS = ["radiation", "vapor_pressure"]

    def __init__(
        self,
        cmip6_dir: str,
        obs_df: pd.DataFrame,
        feature_columns: List[str],
        gcm_models: List[str] = None,
        scenarios: List[str] = None,
        future_periods: List[Tuple[int, int]] = None,
        baseline_period: Tuple[str, str] = ("2001-01-01", "2020-12-31"),
        n_quantiles: int = 100,
    ):
        """
        Args:
            cmip6_dir:        Path to downloaded CMIP6 CSVs (from cmip6.py)
            obs_df:           Observed DataFrame with precip, tmax, tmin, discharge
            feature_columns:  List of feature column names the ML model expects
            gcm_models:       GCM model IDs (default: auto-detect from directory)
            scenarios:        SSP scenarios (default: ["ssp245", "ssp585"])
            future_periods:   List of (start_year, end_year) tuples
            baseline_period:  Tuple of (start_date, end_date) for historical baseline
            n_quantiles:      Number of quantile bins for bias correction
        """
        self.cmip6_dir = Path(cmip6_dir)
        self.obs_df = obs_df.copy()
        self.feature_columns = feature_columns
        self.scenarios = scenarios or ["ssp245", "ssp585"]
        self.future_periods = future_periods or [
            (2031, 2050), (2051, 2070), (2071, 2090),
        ]
        self.baseline_start, self.baseline_end = baseline_period
        self.n_quantiles = n_quantiles

        # Auto-detect GCM models from directory
        if gcm_models is None:
            self.gcm_models = self._detect_gcm_models()
        else:
            self.gcm_models = gcm_models

        # Storage for fitted bias correctors {gcm_name: MultiVariateBiasCorrector}
        self.bias_correctors: Dict[str, MultiVariateBiasCorrector] = {}

        # Storage for corrected future data
        self.corrected_futures: Dict[str, pd.DataFrame] = {}

        print(f"\n  CMIP6RealProjector initialized:")
        print(f"    Directory:  {self.cmip6_dir}")
        print(f"    GCMs:       {self.gcm_models}")
        print(f"    Scenarios:  {self.scenarios}")
        print(f"    Periods:    {self.future_periods}")

    def _detect_gcm_models(self) -> List[str]:
        """Auto-detect available GCM models from the historical directory."""
        hist_dir = self.cmip6_dir / "historical"
        if not hist_dir.exists():
            warnings.warn(f"Historical directory not found: {hist_dir}")
            return []
        models = []
        for f in hist_dir.glob("*_historical_daily.csv"):
            model_name = f.name.replace("_historical_daily.csv", "")
            models.append(model_name)
        return sorted(models)

    def _load_gcm_csv(self, model: str, scenario: str) -> Optional[pd.DataFrame]:
        """Load a downloaded GCM CSV file."""
        path = self.cmip6_dir / scenario / f"{model}_{scenario}_daily.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        df.index.name = "date"
        return df

    # ─────────────────────────────────────────────────────────────
    #  STEP 1: Bias Correction Fitting
    # ─────────────────────────────────────────────────────────────

    def fit_bias_correction(self) -> Dict[str, MultiVariateBiasCorrector]:
        """
        Fit QDM bias correction for each GCM using the historical run
        vs observed data overlap period.

        Returns:
            Dict mapping GCM name → fitted MultiVariateBiasCorrector
        """
        print("\n" + "─" * 65)
        print("  BIAS CORRECTION: Fitting Quantile Delta Mapping")
        print("─" * 65)

        obs_subset = self.obs_df.loc[self.baseline_start:self.baseline_end]

        for model in self.gcm_models:
            print(f"\n  ── {model} ──")
            gcm_hist = self._load_gcm_csv(model, "historical")

            if gcm_hist is None:
                print(f"    ✗ No historical CSV found, skipping {model}")
                continue

            # Determine which variables we can correct
            variables = [v for v in self.CLIMATE_VARS
                         if v in obs_subset.columns and v in gcm_hist.columns]
            variables += [v for v in self.OPTIONAL_VARS
                          if v in obs_subset.columns and v in gcm_hist.columns]

            if not variables:
                print(f"    ✗ No common variables between obs and GCM")
                continue

            bc = MultiVariateBiasCorrector(n_quantiles=self.n_quantiles)
            bc.fit(obs_subset, gcm_hist, variables=variables)
            self.bias_correctors[model] = bc

            # Print overlap period
            common_start = max(obs_subset.index.min(), gcm_hist.index.min())
            common_end = min(obs_subset.index.max(), gcm_hist.index.max())
            n_overlap = len(obs_subset.loc[common_start:common_end])
            print(f"    Overlap: {common_start.date()} → {common_end.date()} "
                  f"({n_overlap} days)")

        print(f"\n  ✓ Bias correction fitted for {len(self.bias_correctors)} GCMs")
        return self.bias_correctors

    # ─────────────────────────────────────────────────────────────
    #  STEP 2: Generate Bias-Corrected Future Climate
    # ─────────────────────────────────────────────────────────────

    def generate_corrected_futures(self) -> Dict[str, pd.DataFrame]:
        """
        Apply bias correction to all future GCM × SSP combinations.

        Returns:
            Dict mapping "GCM_SSP_startYr_endYr" → corrected DataFrame
        """
        print("\n" + "─" * 65)
        print("  BIAS CORRECTION: Applying to future scenarios")
        print("─" * 65)

        for model in self.gcm_models:
            if model not in self.bias_correctors:
                continue

            bc = self.bias_correctors[model]

            for ssp in self.scenarios:
                gcm_future = self._load_gcm_csv(model, ssp)
                if gcm_future is None:
                    print(f"    ✗ No future CSV: {model}/{ssp}")
                    continue

                # Apply bias correction
                corrected = bc.transform(gcm_future)

                # Slice into future periods
                for yr_start, yr_end in self.future_periods:
                    period_data = corrected.loc[
                        f"{yr_start}-01-01":f"{yr_end}-12-31"
                    ]
                    if period_data.empty:
                        continue

                    key = f"{model}_{ssp}_{yr_start}_{yr_end}"
                    self.corrected_futures[key] = period_data

                    print(f"    ✓ {key}: {len(period_data)} days  "
                          f"precip_mean={period_data['precip'].mean():.2f} mm/d")

        print(f"\n  ✓ Generated {len(self.corrected_futures)} corrected future periods")
        return self.corrected_futures

    # ─────────────────────────────────────────────────────────────
    #  STEP 3: Feature Engineering for Future Climate
    # ─────────────────────────────────────────────────────────────

    def _engineer_future_features(
        self,
        climate_df: pd.DataFrame,
        model,
        scaler_X=None,
    ) -> pd.DataFrame:
        """
        Engineer the full feature set from bias-corrected climate data,
        using recursive prediction for discharge lag features.

        This mirrors HydroFeatureEngineer but handles the absence of
        observed discharge by predicting it step-by-step.

        Args:
            climate_df: Bias-corrected future climate DataFrame
                        (columns: precip, tmax, tmin, [radiation, vapor_pressure])
            model:      Trained ML model with .predict(X) method
            scaler_X:   Optional scaler (if model expects scaled input)

        Returns:
            DataFrame with all features matching self.feature_columns
        """
        df = climate_df.copy()

        # ── Temperature-derived features ──
        if "tmax" in df.columns and "tmin" in df.columns:
            df["temp_range"] = df["tmax"] - df["tmin"]
            df["temp_mean"] = (df["tmax"] + df["tmin"]) / 2

        # ── Hargreaves ET estimate ──
        if "tmax" in df.columns and "tmin" in df.columns:
            tmean = (df["tmax"] + df["tmin"]) / 2
            trange = (df["tmax"] - df["tmin"]).clip(lower=0)
            doy = df.index.dayofyear
            # Extraterrestrial radiation for ~12°N (Kabini latitude)
            ra = 15.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365)
            df["et_hargreaves"] = 0.0023 * ra * (tmean + 17.8) * np.sqrt(trange)

        # ── Precipitation lag features ──
        for lag in [1, 2, 3, 5, 7, 14]:
            col = f"precip_lag{lag}"
            if col in self.feature_columns:
                df[col] = df["precip"].shift(lag)

        # ── Precipitation rolling features ──
        for window in [3, 7, 14, 30]:
            col = f"precip_rolling{window}"
            if col in self.feature_columns:
                df[col] = df["precip"].rolling(window).mean()

        for window in [3, 7, 14]:
            col = f"precip_std{window}"
            if col in self.feature_columns:
                df[col] = df["precip"].rolling(window).std()

        # ── Antecedent precipitation ──
        for window in [5, 10, 20, 30]:
            col = f"antecedent_precip_{window}"
            if col in self.feature_columns:
                df[col] = df["precip"].rolling(window).sum()

        # ── SPI features (simplified) ──
        for window in [30, 60, 90]:
            col = f"spi_{window}"
            if col in self.feature_columns:
                rolling_sum = df["precip"].rolling(window).sum()
                mean = rolling_sum.rolling(365, min_periods=30).mean()
                std = rolling_sum.rolling(365, min_periods=30).std()
                df[col] = (rolling_sum - mean) / (std + 1e-8)

        # ── Temporal encodings ──
        dates = df.index
        month = dates.month
        doy = dates.dayofyear
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
        df["is_monsoon"] = ((month >= 6) & (month <= 9)).astype(float)

        # ── Recursive discharge prediction ──
        # Initialize discharge with a climatological estimate
        # (use observed mean for the corresponding month as seed)
        obs_monthly_mean = self.obs_df["discharge"].groupby(
            self.obs_df.index.month
        ).mean()
        df["discharge"] = df.index.month.map(obs_monthly_mean)

        # Discharge lag columns needed
        discharge_lag_cols = [c for c in self.feature_columns if c.startswith("discharge_lag")]
        discharge_rolling_cols = [c for c in self.feature_columns if c.startswith("discharge_rolling")]

        if discharge_lag_cols or discharge_rolling_cols:
            df = self._recursive_predict(
                df, model, scaler_X,
                discharge_lag_cols, discharge_rolling_cols,
            )

        # ── Drop warmup period and select final features ──
        df = df.dropna(subset=[c for c in self.feature_columns if c in df.columns])

        # Ensure all required feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0  # fill missing features with 0

        return df

    def _recursive_predict(
        self,
        df: pd.DataFrame,
        model,
        scaler_X,
        lag_cols: List[str],
        rolling_cols: List[str],
    ) -> pd.DataFrame:
        """
        Recursively predict discharge day-by-day, filling in lag features
        as we go. This handles the chicken-and-egg problem: we need
        discharge_lag1..7 to predict Q, but Q is what we're predicting.

        Strategy:
            1. Seed with climatological discharge for warmup (first 30 days)
            2. For each subsequent day:
                a. Compute discharge lags from previously predicted values
                b. Assemble full feature vector
                c. Predict Q_t using the trained model
                d. Store Q_t for use in future lags
        """
        discharge_pred = df["discharge"].copy()

        # Parse lag values needed
        lags_needed = set()
        for col in lag_cols:
            lag_num = int(col.replace("discharge_lag", ""))
            lags_needed.add(lag_num)

        rolling_windows = set()
        for col in rolling_cols:
            window = int(col.replace("discharge_rolling", ""))
            rolling_windows.add(window)

        warmup = max(max(lags_needed, default=0), max(rolling_windows, default=0), 30)
        feature_cols = self.feature_columns

        for i in range(warmup, len(df)):
            # Fill discharge lags
            for lag in lags_needed:
                col = f"discharge_lag{lag}"
                if i - lag >= 0:
                    df.iloc[i, df.columns.get_loc(col)] = discharge_pred.iloc[i - lag]

            # Fill discharge rolling means
            for window in rolling_windows:
                col = f"discharge_rolling{window}"
                start_idx = max(0, i - window)
                df.iloc[i, df.columns.get_loc(col)] = discharge_pred.iloc[start_idx:i].mean()

            # Assemble feature vector and predict
            row = df.iloc[i]
            X_row = np.array([[row[c] for c in feature_cols]], dtype=np.float32)

            # Handle NaN in features (replace with 0)
            X_row = np.nan_to_num(X_row, nan=0.0)

            try:
                pred = model.predict(X_row)[0]
                pred = max(pred, 0.0)  # non-negative discharge
            except Exception:
                pred = discharge_pred.iloc[i]  # fallback to climatology

            discharge_pred.iloc[i] = pred

        df["discharge"] = discharge_pred
        return df

    # ─────────────────────────────────────────────────────────────
    #  STEP 4: Full Projection Pipeline
    # ─────────────────────────────────────────────────────────────

    def generate_all_scenarios(
        self,
        trained_model,
        scaler_X=None,
    ) -> Dict[str, Dict]:
        """
        Run the complete projection pipeline:
            1. Bias-correct future climate data (if not already done)
            2. Engineer features with recursive Q prediction
            3. Predict discharge for each GCM × SSP × period

        Args:
            trained_model: Fitted model with .predict(X) method
            scaler_X:      Scaler for input features (if model needs it)

        Returns:
            Dict: { scenario_key: { "discharge": array, "dates": array,
                                     "features": DataFrame, "metadata": dict } }
        """
        if not self.corrected_futures:
            self.fit_bias_correction()
            self.generate_corrected_futures()

        print("\n" + "─" * 65)
        print("  PROJECTIONS: Generating future discharge predictions")
        print("─" * 65)

        all_projections = {}

        for key, climate_df in self.corrected_futures.items():
            print(f"\n  ── {key} ──")

            # Engineer features (with recursive discharge prediction)
            feature_df = self._engineer_future_features(
                climate_df, trained_model, scaler_X
            )

            if len(feature_df) == 0:
                print(f"    ✗ No valid rows after feature engineering")
                continue

            # Final prediction using the feature set
            X_future = feature_df[self.feature_columns].values.astype(np.float32)
            X_future = np.nan_to_num(X_future, nan=0.0)

            discharge_pred = trained_model.predict(X_future)
            discharge_pred = np.maximum(discharge_pred, 0.0)

            # Parse metadata from key
            parts = key.split("_")
            gcm = "_".join(parts[:-3]) if len(parts) > 4 else parts[0]
            ssp = parts[-3]
            yr_start, yr_end = int(parts[-2]), int(parts[-1])

            all_projections[key] = {
                "discharge": discharge_pred,
                "dates": feature_df.index,
                "features": feature_df,
                "metadata": {
                    "GCM": gcm,
                    "SSP": ssp,
                    "period_start": yr_start,
                    "period_end": yr_end,
                    "n_days": len(discharge_pred),
                    "mean_Q": float(np.mean(discharge_pred)),
                    "max_Q": float(np.max(discharge_pred)),
                    "min_Q": float(np.min(discharge_pred)),
                },
            }

            print(f"    ✓ {len(discharge_pred)} days predicted  "
                  f"mean_Q={np.mean(discharge_pred):.2f}  "
                  f"max_Q={np.max(discharge_pred):.2f}")

        return all_projections

    # ─────────────────────────────────────────────────────────────
    #  STEP 5: Summarize Projections
    # ─────────────────────────────────────────────────────────────

    def summarize_projections(
        self,
        projections: Dict[str, Dict],
        baseline_discharge: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Create a summary table of all projections.

        Args:
            projections:        Output from generate_all_scenarios()
            baseline_discharge: Observed discharge array for computing % change

        Returns:
            DataFrame with columns: GCM, SSP, Period, Mean_Q, Change_pct,
                                    Max_Q, Min_Q, Std_Q
        """
        if baseline_discharge is None:
            baseline_discharge = self.obs_df["discharge"].values

        baseline_mean = float(np.mean(baseline_discharge))

        rows = []
        for key, data in projections.items():
            meta = data["metadata"]
            q = data["discharge"]
            change_pct = ((np.mean(q) - baseline_mean) / baseline_mean) * 100

            rows.append({
                "GCM": meta["GCM"],
                "SSP": meta["SSP"],
                "Period": f"{meta['period_start']}-{meta['period_end']}",
                "Mean_Q": round(float(np.mean(q)), 2),
                "Change_pct": round(float(change_pct), 1),
                "Max_Q": round(float(np.max(q)), 2),
                "Min_Q": round(float(np.min(q)), 2),
                "Std_Q": round(float(np.std(q)), 2),
                "Q10": round(float(np.percentile(q, 10)), 2),
                "Q50": round(float(np.percentile(q, 50)), 2),
                "Q90": round(float(np.percentile(q, 90)), 2),
            })

        summary_df = pd.DataFrame(rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                ["SSP", "GCM", "Period"]
            ).reset_index(drop=True)

        return summary_df

    def generate_ensemble_summary(
        self,
        projections: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Compute multi-model ensemble statistics per SSP × period.

        Averages across GCMs to give ensemble mean, spread, and confidence
        intervals for each SSP × period combination.
        """
        # Group projections by SSP_period
        grouped = {}
        for key, data in projections.items():
            meta = data["metadata"]
            group_key = f"{meta['SSP']}_{meta['period_start']}_{meta['period_end']}"
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(data["discharge"])

        rows = []
        for group_key, discharge_list in grouped.items():
            parts = group_key.split("_")
            ssp, yr_start, yr_end = parts[0], parts[1], parts[2]

            # Compute ensemble statistics
            means = [np.mean(q) for q in discharge_list]
            maxes = [np.max(q) for q in discharge_list]

            rows.append({
                "SSP": ssp,
                "Period": f"{yr_start}-{yr_end}",
                "Ensemble_Mean_Q": round(float(np.mean(means)), 2),
                "Ensemble_Std_Q": round(float(np.std(means)), 2),
                "Ensemble_Min_Mean": round(float(np.min(means)), 2),
                "Ensemble_Max_Mean": round(float(np.max(means)), 2),
                "Ensemble_Max_Peak": round(float(np.max(maxes)), 2),
                "N_GCMs": len(discharge_list),
            })

        return pd.DataFrame(rows).sort_values(["SSP", "Period"]).reset_index(drop=True)

    def save_projections(
        self,
        projections: Dict[str, Dict],
        output_dir: str = "outputs",
    ):
        """Save projection results to JSON and individual CSVs."""
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True)

        # Save summary
        summary = self.summarize_projections(projections)
        summary.to_csv(out_path / "cmip6_projections_summary.csv", index=False)

        ensemble = self.generate_ensemble_summary(projections)
        ensemble.to_csv(out_path / "cmip6_ensemble_summary.csv", index=False)

        # Save individual time series
        ts_dir = out_path / "projection_timeseries"
        ts_dir.mkdir(exist_ok=True)

        for key, data in projections.items():
            ts_df = pd.DataFrame({
                "date": data["dates"],
                "discharge_predicted": data["discharge"],
            })
            ts_df.to_csv(ts_dir / f"{key}.csv", index=False)

        # Save metadata JSON
        meta_dict = {k: v["metadata"] for k, v in projections.items()}
        with open(out_path / "cmip6_projection_metadata.json", "w") as f:
            json.dump(meta_dict, f, indent=2)

        print(f"\n  ✓ Projections saved to {out_path}/")
        print(f"    Summary:     cmip6_projections_summary.csv")
        print(f"    Ensemble:    cmip6_ensemble_summary.csv")
        print(f"    Timeseries:  projection_timeseries/*.csv")