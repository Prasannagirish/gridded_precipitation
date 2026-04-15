"""
FlowCast v2 — Configuration
Kabini River Basin Discharge Prediction Pipeline
Techniques from: Yan et al. (2026), Eatesam et al. (2025), Dhakal et al. (2020)
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "saved_models"

for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True)


@dataclass
class BasinConfig:
    """Kabini River Basin static attributes."""
    name: str = "Kabini"
    area_km2: float = 7040.0
    mean_elevation_m: float = 850.0
    mean_slope: float = 5.2
    forest_fraction: float = 0.35
    soil_porosity: float = 0.42
    soil_conductivity: float = 12.5        # mm/hr
    clay_fraction: float = 0.28
    sand_fraction: float = 0.35
    silt_fraction: float = 0.37
    aridity_index: float = 0.65
    precip_seasonality: float = 0.8        # high monsoon dominance
    mean_annual_precip_mm: float = 1400.0
    mean_annual_temp_c: float = 24.5
    baseflow_index: float = 0.45
    lai_max: float = 4.2
    geological_permeability: float = -13.0  # log10


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    # Temporal settings adjusted for 1990-2020 dataset
    train_start: str = "1990-01-01"
    train_end: str = "2015-12-31"
    val_start: str = "2016-01-01"
    val_end: str = "2018-12-31"
    test_start: str = "2019-01-01"
    test_end: str = "2020-12-31"

    # Feature families updated to exclude temperature/radiation data
    feature_combinations: Dict[str, List[str]] = field(default_factory=lambda: {
        "M1_full": [
            "precip", 
            "precip_lag1", "precip_lag2", "precip_lag3", "precip_lag7",
            "precip_rolling7", "precip_rolling14", "precip_rolling30",
            "discharge_lag1", "discharge_lag2", "discharge_lag3", "discharge_lag7",
            "month_sin", "month_cos", "doy_sin", "doy_cos",
            "antecedent_precip_5", "antecedent_precip_10",
            "spi_30", "spi_90",
        ],
        "M2_meteo_only": [
            "precip", 
            "precip_lag1", "precip_lag2", "precip_lag3",
            "precip_rolling7", "precip_rolling14",
            "month_sin", "month_cos",
        ],
        "M3_hydro_focus": [
            "precip", 
            "discharge_lag1", "discharge_lag2", "discharge_lag3", "discharge_lag7",
            "precip_lag1", "precip_lag2", "precip_lag3",
            "precip_rolling7", "precip_rolling30",
            "antecedent_precip_5", "antecedent_precip_10",
        ],
        "M4_temporal": [
            "precip", 
            "discharge_lag1", "discharge_lag2",
            "precip_lag1", "precip_lag2",
            "month_sin", "month_cos", "doy_sin", "doy_cos",
            "precip_rolling7",
        ],
        "M5_minimal": [
            "precip", 
            "discharge_lag1",
            "precip_lag1",
            "precip_rolling7",
            "month_sin", "month_cos",
        ],
    })

    # K-fold cross-validation
    n_folds: int = 5
    sequence_length: int = 30  # lookback window for RNNs


@dataclass
class ModelConfig:
    """Model hyperparameter configuration."""
    random_state: int = 42

    # PSO settings (Paper 2)
    pso_n_particles: int = 20
    pso_n_iterations: int = 50
    pso_w: float = 0.7        # inertia weight
    pso_c1: float = 1.5       # cognitive parameter
    pso_c2: float = 1.5       # social parameter

    # Random Forest
    rf_param_bounds: Dict = field(default_factory=lambda: {
        "n_estimators": (50, 500),
        "max_depth": (3, 30),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": (0.3, 1.0),
    })

    # SVR
    svr_param_bounds: Dict = field(default_factory=lambda: {
        "C": (0.1, 1000.0),
        "epsilon": (0.001, 1.0),
        "gamma": (0.0001, 1.0),
    })

    # LSTM
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 64
    lstm_patience: int = 15

    # A-GRU (Paper 3)
    agru_hidden_size: int = 256
    agru_dropout: float = 0.4
    agru_learning_rate: float = 0.0005
    agru_epochs: int = 100
    agru_batch_size: int = 64
    agru_patience: int = 15
    agru_seq_length: int = 270

    # Physics-informed LSTM (Paper 1 — Xie et al. approach)
    pi_lambda_water_balance: float = 0.1
    pi_lambda_monotonicity: float = 0.05


@dataclass
class ClimateConfig:
    """CMIP6 climate projection configuration."""
    scenarios: List[str] = field(default_factory=lambda: ["ssp245", "ssp585"])
    future_periods: List[tuple] = field(default_factory=lambda: [
        (2031, 2050),
        (2051, 2070),
        (2071, 2090),
    ])
    baseline_period: tuple = (2001, 2020)

    # GCM models to evaluate
    gcm_models: List[str] = field(default_factory=lambda: [
        "MRI-ESM2-0",
        "CMCC-ESM2",
        "MPI-ESM1-2-LR",
        "INM-CM5-0",
        "ACCESS-CM2",
    ])

    # ── CMIP6 Real Data Paths ──
    # Path to downloaded CMIP6 CSVs (output of cmip6.py downloader)
    cmip6_download_dir: str = "cmip6_downloads"

    # Path to Cauvery basin shapefile (for spatial clipping)
    basin_shapefile: str = "data/cauvery_basin.shp"

    # Bias correction settings
    n_quantiles: int = 100          # quantile bins for QDM
    bias_correct_monthly: bool = True  # fit per calendar month

    # Whether to use real CMIP6 data (True) or synthetic delta-change (False)
    use_real_cmip6: bool = True

    # Variables to download and bias-correct (Removed tasmax and tasmin)
    cmip6_variables: List[str] = field(default_factory=lambda: [
        "pr",       # precipitation → precip (mm/day)
    ])

    # Cleared optional variables since we don't have baseline observations for them
    cmip6_optional_variables: List[str] = field(default_factory=lambda: [])


# Global config instances
basin_cfg = BasinConfig()
data_cfg = DataConfig()
model_cfg = ModelConfig()
climate_cfg = ClimateConfig()