"""
FlowCast v2 — Data Loader
Handles CHIRPS rainfall, CWC discharge, ERA5 reanalysis, and static basin attributes.
Includes synthetic data generator for testing the pipeline.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class KabiniDataLoader:
    """Load and preprocess Kabini River Basin data."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")

    def load_real_data(
        self,
        precip_path: str,
        discharge_path: str,
        temp_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load real observation data from CSV files.
        """
        precip = pd.read_csv(precip_path, parse_dates=["date"])
        discharge = pd.read_csv(discharge_path, parse_dates=["date"])

        df = precip.merge(discharge, on="date", how="inner")
        
        # --- NEW: Map your specific file columns to pipeline columns ---
        rename_map = {
            "rainfall_mean_mm": "precip",   # From chirps_kabini_daily.csv
            "precip_mm": "precip",          # Fallback
            "q_upstream_mk": "discharge",   # From discharge_daily_observed.csv
            "discharge_cumecs": "discharge" # Fallback
        }
        df = df.rename(columns=rename_map)

        if temp_path:
            temp = pd.read_csv(temp_path, parse_dates=["date"])
            df = df.merge(temp, on="date", how="left")
        else:
            # Estimate temperature from seasonal cycle if not available
            doy = df["date"].dt.dayofyear
            df["tmax"] = 30 + 5 * np.sin(2 * np.pi * (doy - 120) / 365)
            df["tmin"] = 18 + 4 * np.sin(2 * np.pi * (doy - 120) / 365)

        # Derive additional meteorological fields
        df["radiation"] = 200 + 100 * np.sin(
            2 * np.pi * (df["date"].dt.dayofyear - 80) / 365
        )
        df["vapor_pressure"] = 1.5 + 0.8 * np.sin(
            2 * np.pi * (df["date"].dt.dayofyear - 180) / 365
        )

        df = df.set_index("date").sort_index()
        df = df.dropna(subset=["precip", "discharge"])
        return df
    
    def generate_synthetic_data(
        self,
        start_date: str = "2001-01-01",
        end_date: str = "2023-12-31",
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate physically plausible synthetic data for Kabini basin.
        Mimics Indian monsoon patterns for pipeline testing.
        """
        np.random.seed(seed)
        dates = pd.date_range(start_date, end_date, freq="D")
        n = len(dates)
        doy = dates.dayofyear.values.astype(float)

        # --- Precipitation: monsoon-dominated ---
        # Monsoon season: June-September (DOY ~152-273)
        monsoon_signal = np.exp(
            -0.5 * ((doy - 210) / 40) ** 2
        )  # Gaussian centered on late July
        base_precip_prob = 0.05 + 0.7 * monsoon_signal
        rain_occurs = np.random.rand(n) < base_precip_prob

        precip_intensity = np.random.exponential(scale=8 + 25 * monsoon_signal)
        precip = rain_occurs * precip_intensity
        # Add occasional extreme events
        extreme_mask = np.random.rand(n) < 0.005
        precip[extreme_mask] *= np.random.uniform(3, 8, size=extreme_mask.sum())
        precip = np.maximum(precip, 0)

        # --- Temperature ---
        tmax = 30 + 5 * np.sin(2 * np.pi * (doy - 120) / 365) + np.random.normal(0, 1.5, n)
        tmin = 18 + 4 * np.sin(2 * np.pi * (doy - 120) / 365) + np.random.normal(0, 1.2, n)
        tmin = np.minimum(tmin, tmax - 2)

        # --- Radiation & vapor pressure ---
        radiation = (
            200 + 100 * np.sin(2 * np.pi * (doy - 80) / 365)
            + np.random.normal(0, 20, n)
        )
        vapor_pressure = (
            1.5 + 0.8 * np.sin(2 * np.pi * (doy - 180) / 365)
            + np.random.normal(0, 0.15, n)
        )

        # --- Discharge: rainfall-runoff with memory ---
        discharge = np.zeros(n)
        soil_moisture = 0.3  # normalized
        for i in range(n):
            # Simple conceptual model
            pet = 3.0 + 2.5 * np.sin(2 * np.pi * (doy[i] - 120) / 365)
            aet = pet * min(soil_moisture / 0.5, 1.0)

            # Runoff generation
            if soil_moisture > 0.6:
                runoff_coeff = 0.3 + 0.5 * (soil_moisture - 0.6) / 0.4
            else:
                runoff_coeff = 0.05 + 0.25 * soil_moisture

            direct_runoff = precip[i] * runoff_coeff
            baseflow = max(0, 15 * (soil_moisture - 0.2))

            discharge[i] = direct_runoff + baseflow + np.random.normal(0, 2)
            discharge[i] = max(0, discharge[i])

            # Update soil moisture
            soil_moisture += (precip[i] * 0.8 - aet - direct_runoff * 0.5) / 500
            soil_moisture = np.clip(soil_moisture, 0.05, 0.95)

        # Apply routing delay (convolution with unit hydrograph)
        uh = np.array([0.05, 0.15, 0.3, 0.25, 0.15, 0.07, 0.03])
        discharge_routed = np.convolve(discharge, uh, mode="same")
        discharge_routed = np.maximum(discharge_routed, 0)

        df = pd.DataFrame({
            "precip": precip,
            "tmax": tmax,
            "tmin": tmin,
            "radiation": np.maximum(radiation, 0),
            "vapor_pressure": np.maximum(vapor_pressure, 0.1),
            "discharge": discharge_routed,
        }, index=dates)
        df.index.name = "date"

        return df

    def get_static_features(self) -> dict:
        """Return static basin attributes for A-GRU (Paper 3 style)."""
        from config import basin_cfg
        return {
            "area_km2": basin_cfg.area_km2,
            "mean_elevation_m": basin_cfg.mean_elevation_m,
            "mean_slope": basin_cfg.mean_slope,
            "forest_fraction": basin_cfg.forest_fraction,
            "soil_porosity": basin_cfg.soil_porosity,
            "soil_conductivity": basin_cfg.soil_conductivity,
            "clay_fraction": basin_cfg.clay_fraction,
            "sand_fraction": basin_cfg.sand_fraction,
            "silt_fraction": basin_cfg.silt_fraction,
            "aridity_index": basin_cfg.aridity_index,
            "precip_seasonality": basin_cfg.precip_seasonality,
            "mean_annual_precip_mm": basin_cfg.mean_annual_precip_mm,
            "mean_annual_temp_c": basin_cfg.mean_annual_temp_c,
            "baseflow_index": basin_cfg.baseflow_index,
            "lai_max": basin_cfg.lai_max,
            "geological_permeability": basin_cfg.geological_permeability,
        }


def train_val_test_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split — no shuffling to preserve time-series integrity."""
    train = df.loc[:train_end]
    val = df.loc[train_end:val_end].iloc[1:]  # exclude boundary
    test = df.loc[val_end:].iloc[1:]
    return train, val, test