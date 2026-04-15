"""
FlowCast v2 — Configuration
Kabini River Basin Discharge Prediction Pipeline
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
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
    # Temporal settings adjusted for 1990-2014 baseline
    train_start: str = "1990-01-01"
    train_end: str = "2005-12-31"
    val_start: str = "2006-01-01"
    val_end: str = "2010-12-31"
    test_start: str = "2011-01-01"
    test_end: str = "2014-12-31"

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
    })
    n_folds: int = 5
    sequence_length: int = 30


@dataclass
class ModelConfig:
    """Model hyperparameter configuration."""
    random_state: int = 42

    pso_n_particles: int = 20
    pso_n_iterations: int = 50
    pso_w: float = 0.7        
    pso_c1: float = 1.5       
    pso_c2: float = 1.5       

    rf_param_bounds: Dict = field(default_factory=lambda: {
        "n_estimators": (50, 500),
        "max_depth": (3, 30),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": (0.3, 1.0),
    })

    svr_param_bounds: Dict = field(default_factory=lambda: {
        "C": (0.1, 1000.0),
        "epsilon": (0.001, 1.0),
        "gamma": (0.0001, 1.0),
    })

    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 64
    lstm_patience: int = 15

    agru_hidden_size: int = 256
    agru_dropout: float = 0.4
    agru_learning_rate: float = 0.0005
    agru_epochs: int = 100
    agru_batch_size: int = 64
    agru_patience: int = 15
    agru_seq_length: int = 270

    pi_lambda_water_balance: float = 0.1
    pi_lambda_monotonicity: float = 0.05


@dataclass
class ClimateConfig:
    """CMIP6 climate projection configuration."""
    # Updated to match the requested scenario ranges
    scenarios: List[str] = field(default_factory=lambda: ["ssp245", "ssp585"])
    
    baseline_period: tuple = (1990, 2014)
    future_periods: List[tuple] = field(default_factory=lambda: [
        (2015, 2050)
    ])

    # GCM models updated to requested list
    gcm_models: List[str] = field(default_factory=lambda: [
        "EC-Earth3",
        "EC-Earth3-Veg",
        "MIROC6"
    ])

    cmip6_download_dir: str = "cmip6_downloads"

    # Updated shapefile extension to .kml
    basin_shapefile: str = "data/cauvery_basin.kml"

    n_quantiles: int = 100          
    bias_correct_monthly: bool = True  
    use_real_cmip6: bool = True

    cmip6_variables: List[str] = field(default_factory=lambda: ["pr"])
    cmip6_optional_variables: List[str] = field(default_factory=lambda: [])

# Global config instances
basin_cfg = BasinConfig()
data_cfg = DataConfig()
model_cfg = ModelConfig()
climate_cfg = ClimateConfig()