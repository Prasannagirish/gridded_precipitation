# 🌊 FlowCast v2 — Kabini River Basin Discharge Forecasting

> **End-to-end machine learning pipeline for daily streamflow prediction using CHIRPS rainfall, CWC discharge records, and CMIP6 climate projections (SSP2-4.5 / SSP5-8.5, 2015–2050)**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Data Sources](#data-sources)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [CMIP6 Climate Projections](#cmip6-climate-projections)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [References](#references)

---

## Overview

FlowCast v2 is a research-grade, production-ready pipeline that forecasts daily river discharge at the **Kabini River Basin** (7,040 km², Karnataka, India). It spans five phases:

1. **Exploratory Data Analysis** — lag correlation, seasonality, extreme events
2. **Classical Hydrological Analysis** — Eckhardt baseflow, FDC, flood frequency, Snyder UH
3. **Feature Engineering** — 35 hydrologically-informed features
4. **Multi-Model Training with PSO** — 7 models, Particle Swarm Optimisation
5. **CMIP6 Climate Projections** — QDM bias correction, 3 GCMs, 2 SSP scenarios, 2015–2050

All results are served through a **7-tab interactive Streamlit dashboard** with Plotly visualisations.

---

## Results Summary

### Best Model: SVR-PSO — Test Set (2011–2014)

| Metric | Value |
|--------|-------|
| **NSE** | **0.9093** |
| **KGE** | **0.8866** |
| RMSE | 51.16 m³/s |
| MAE | 17.79 m³/s |
| R² | 0.9093 |
| PBIAS | −4.21% |

### All Models — Test NSE Comparison

| Model | NSE | KGE | RMSE | Notes |
|-------|-----|-----|------|-------|
| **SVR-PSO** ⭐ | **0.9093** | 0.8866 | 51.16 | Best overall |
| XGBoost-PSO | 0.8606 | 0.8050 | 63.42 | Strong high-flow |
| RF-PSO | 0.8446 | 0.8021 | 66.96 | Best low-flow (tree) |
| A-GRU | 0.7504 | 0.7230 | 85.16 | Best deep learning |
| Weighted Ensemble | 0.7464 | 0.7124 | 85.85 | A-GRU weight: 0.731 |
| PI-LSTM | 0.7373 | 0.7252 | 87.38 | Physics constraints |
| LSTM | 0.7326 | 0.7246 | 88.16 | Baseline RNN |

### CMIP6 Projections (2015–2050, vs 1990–2014 baseline ~78.7 m³/s)

| GCM | SSP | Mean Q (m³/s) | Change |
|-----|-----|--------------|--------|
| EC-Earth3 | SSP2-4.5 | 68.04 | −13.6% |
| EC-Earth3-Veg | SSP2-4.5 | 67.41 | −14.4% |
| MIROC6 | SSP2-4.5 | 64.09 | −18.6% |
| EC-Earth3 | SSP5-8.5 | 69.70 | −11.5% |
| EC-Earth3-Veg | SSP5-8.5 | 69.43 | −11.9% |
| MIROC6 | SSP5-8.5 | 61.67 | **−21.7%** |

---

## Project Structure

```
flowcast-v2/
├── app.py                    # Streamlit dashboard (7 tabs)
├── main.py                   # Unified pipeline orchestrator
├── config.py                 # Basin, data, model, and climate config
│
├── modules/
│   ├── loader.py             # CHIRPS + CWC data loading & synthetic gen
│   ├── eda.py                # Exploratory data analysis
│   ├── hydro.py              # Hydrological analysis (Eckhardt, FDC, UH)
│   ├── engineer.py           # Feature engineering (35 features)
│   ├── traditional_ml.py     # RF-PSO, SVR-PSO, XGBoost-PSO
│   ├── deep_learning.py      # LSTM, PI-LSTM, A-GRU architectures
│   ├── ensemble.py           # Weighted ensemble + stacking
│   ├── pso.py                # Particle Swarm Optimisation
│   ├── metrics.py            # NSE, KGE, RMSE, MAE, R², PBIAS
│   ├── bias_correction.py    # Quantile Delta Mapping (QDM)
│   ├── cmip6.py              # CMIP6 data loading & processing
│   ├── cmip6_projector.py    # Future discharge projection
│   └── visualization.py      # Plot generation
│
├── data/
│   ├── chirps_kabini_daily.csv       # CHIRPS basin-average rainfall
│   ├── discharge_daily_observed.csv  # CWC gauge data (Muthankera)
│   └── cauvery_basin.kml             # Basin boundary shapefile
│
├── outputs/                  # Generated plots and results
│   ├── pipeline_results.json         # All model metrics
│   ├── error_violin_plot.png
│   ├── hydrograph_*.png
│   ├── scatter_*.png
│   ├── cmip6_regime_hydrograph*.png
│   ├── cmip6_projections.png
│   ├── cmip6_yearly_forecasts/
│   └── cmip6_detailed_forecasts/
│
├── hydrological_outputs/     # Hydrology phase outputs
│   ├── eckhardt_separation.png
│   ├── recession_curve.png
│   ├── snyder_unit_hydrograph.png
│   ├── annual_water_balance.png
│   └── hydrology_parameters.csv
│
├── eda_outputs/              # EDA phase outputs
│   ├── rainfall_timeseries.png
│   ├── discharge_timeseries.png
│   ├── lag_correlation.png
│   └── correlation_heatmap.png
│
├── saved_models/             # Serialised model checkpoints
└── requirements.txt
```

---

## Pipeline Architecture

```
Raw Data (CHIRPS + CWC)
        │
        ▼
┌─────────────────┐
│  Phase 1: EDA   │  Lag correlation (1–14d), seasonality,
│                 │  extreme events, distribution analysis
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Phase 2: Hydrology  │  Eckhardt baseflow (BFI=0.45),
│                      │  Flow Duration Curve, Gumbel flood
│                      │  frequency, Snyder UH, SCS CN~72
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────┐
│  Phase 3: Feature Eng.  │  35 features — lags, rolling,
│                         │  SPI, API, ET, temporal encodings
└───────────┬─────────────┘
            │
            ▼
┌────────────────────────────────────────────────────┐
│               Phase 4: Model Training               │
│                                                    │
│  Tree-Based (PSO)     Deep Learning               │
│  ├── RF-PSO           ├── LSTM (2-layer, 128u)    │
│  ├── SVR-PSO ⭐       ├── PI-LSTM (physics loss)  │
│  └── XGBoost-PSO      ├── A-GRU (static/dynamic)  │
│                       └── Weighted Ensemble        │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────┐
│      Phase 5: CMIP6 Projections      │
│                                      │
│  GCMs: EC-Earth3, EC-Earth3-Veg,    │
│         MIROC6                       │
│  Scenarios: SSP2-4.5, SSP5-8.5      │
│  Period: 2015–2050                   │
│  Bias Correction: QDM (monthly)      │
└──────────────────────────────────────┘
```

---

## Data Sources

| Dataset | Variable | Source | Period |
|---------|----------|--------|--------|
| CHIRPS v2.0 | Daily basin rainfall (mm) | CHG / Google Earth Engine | 1990–2014 |
| CWC Gauge | Daily discharge at Muthankera (m³/s) | Central Water Commission | 1990–2014 |
| NASA NEX-GDDP | Daily precipitation (pr) | NASA / CMIP6 | 2015–2050 |
| ERA5 (estimated) | Tmax, Tmin (seasonal climatology) | ECMWF / derived | 1990–2014 |

### Data Files Required

```
data/
├── chirps_kabini_daily.csv
│   columns: date, rainfall_mean_mm
│
└── discharge_daily_observed.csv
    columns: date, q_upstream_mk
```

---

## Models

### Particle Swarm Optimisation (PSO)

Custom PSO implementation in `pso.py`:

```python
PSOOptimizer(
    n_particles = 20,
    n_iterations = 50,
    w = 0.7,         # inertia (decays × 0.99/iter)
    c1 = 1.5,        # cognitive coefficient
    c2 = 1.5,        # social coefficient
)
# Objective: NSE via TimeSeriesSplit (3 folds)
```

### SVR-PSO (Best Model)

```python
SVR(kernel='rbf',
    C=<PSO-optimised>,
    epsilon=<PSO-optimised>,
    gamma=<PSO-optimised>)
# Search space: C∈[0.1,500], ε∈[0.001,0.5], γ∈[0.0001,0.5]
```

### Physics-Informed LSTM

Loss function combining MSE with physical penalties:

```
L = MSE + λ_wb × WaterBalancePenalty + λ_mono × MonotonicityPenalty
```

- **Water balance**: penalises negative storage residuals (P − ET − Q < 0)
- **Monotonicity**: penalises inverse Q response to positive ΔP

### Adapted-GRU (A-GRU)

Separate input pathways for static basin attributes (16 features) and dynamic meteorological inputs. Custom gating:

```
i[t]  = σ(W_i · x_static)              # static input gate
r[t]  = σ(W_r · x_dyn[t] + U_r · h[t-1])
g[t]  = tanh(W_g · x_dyn[t] + U_g · h[t-1])
c[t]  = (1 - r[t]) · c[t-1] + r[t] · i[t] · g[t]
```

---

## Feature Engineering

```python
# Lag features
precip_lag{1,2,3,5,7,14}
discharge_lag{1,2,3,5,7}

# Rolling statistics
precip_rolling{3,7,14,30}       # mean
precip_std{3,7,14}              # std dev
discharge_rolling{7,14,30}

# Antecedent Precipitation Index
antecedent_precip_{5,10,20,30}  # cumulative sum

# Standardised Precipitation Index
spi_{30,60,90}                  # z-score of rolling sum

# Temporal encodings (cyclical)
month_sin, month_cos
doy_sin, doy_cos
is_monsoon                      # Jun–Sep indicator

# Temperature-derived
temp_range, temp_mean, et_hargreaves
```

Primary feature set `M1_full` selects 20 features for model training.

---

## CMIP6 Climate Projections

### Bias Correction: Quantile Delta Mapping (QDM)

```python
# Cannon et al. (2015) — preserves GCM climate change signal
bc = QuantileDeltaMapping(n_quantiles=100, monthly=True)
bc.fit(obs_chirps, gcm_historical)          # fit on 1990–2014 overlap
corrected = bc.transform(gcm_future, method='multiplicative')  # precip
```

Monthly-stratified fitting captures seasonal bias structure. Multiplicative method for precipitation preserves ratios and clips negatives.

### GCM Configuration

```python
gcm_models  = ["EC-Earth3", "EC-Earth3-Veg", "MIROC6"]
scenarios   = ["ssp245", "ssp585"]
period      = (2015, 2050)
variable    = "pr"   # daily precipitation (kg m⁻² s⁻¹)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/flowcast-v2.git
cd flowcast-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
streamlit>=1.32.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
torch>=2.0
xgboost>=2.0
plotly>=5.18
matplotlib>=3.7
scipy>=1.11
joblib>=1.3
optuna>=3.4            # optional: for Optuna sweeps
```

---

## Usage

### Run the Full Pipeline

```bash
python main.py
```

This executes all 5 phases sequentially (~20–30 min on CPU):
- Loads / generates data
- Runs hydrological analysis
- Engineers features
- Trains and evaluates all 7 models with PSO
- Generates CMIP6 projections and all plots
- Saves `outputs/pipeline_results.json`

### Launch the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Run Individual Modules

```python
from modules.loader import KabiniDataLoader
from modules.engineer import HydroFeatureEngineer
from modules.traditional_ml import SVRModel
from modules.metrics import evaluate_all

loader = KabiniDataLoader()
df = loader.load_real_data("data/chirps_kabini_daily.csv",
                            "data/discharge_daily_observed.csv")
engineer = HydroFeatureEngineer(df)
df_features = engineer.build_all_features()
```

---

## Dashboard

The Streamlit dashboard (`app.py`) provides 7 interactive tabs:

| Tab | Content |
|-----|---------|
| ⚙️ Control Center | Basin map, pipeline execution, high-level KPI tiles |
| 💧 Catchment Physics | Eckhardt separation, FDC, flood frequency, recession curve, Snyder UH |
| 📊 ML Performance | Error violin plot, feature importance, hydrographs, scatter plots |
| 🌍 Climate Projections | CMIP6 summary table, regime hydrograph, yearly trends, detailed forecasts |

---

## Configuration

All settings are centralised in `config.py`:

```python
# Temporal splits
DataConfig(
    train_start = "1990-01-01",
    train_end   = "2005-12-31",
    val_start   = "2006-01-01",
    val_end     = "2010-12-31",
    test_start  = "2011-01-01",
    test_end    = "2014-12-31",
    sequence_length = 30,           # LSTM lookback window
)

# PSO settings
ModelConfig(
    pso_n_particles  = 20,
    pso_n_iterations = 50,
    pso_w  = 0.7,
    pso_c1 = 1.5,
    pso_c2 = 1.5,
)

# Climate scenarios
ClimateConfig(
    gcm_models = ["EC-Earth3", "EC-Earth3-Veg", "MIROC6"],
    scenarios  = ["ssp245", "ssp585"],
)
```

---

## References

1. Eckhardt, K. (2005). Recursive digital baseflow separation filters. *Hydrological Processes*, 19, 507–515.
2. Cannon, A.J. et al. (2015). QDM bias correction of GCM precipitation. *Journal of Climate*, 28, 6938–6959.
3. Eatesam, S. et al. (2022). PSO-optimized SVR/RF for streamflow forecasting. *Water Resources Management*.
4. Xie, K. et al. (2021). Physics-Informed LSTM for hydrological modelling. *Journal of Hydrology*.
5. Dhakal, N. et al. (2020). Adapted-GRU for multi-basin discharge prediction. *HESS*.
6. Funk, C. et al. (2015). CHIRPS. *Scientific Data*, 2, 150066.
7. Eyring, V. et al. (2016). Overview of CMIP6. *Geoscientific Model Development*, 9, 1937–1958.
8. Nash, J.E. & Sutcliffe, J.V. (1970). River flow forecasting through conceptual models. *Journal of Hydrology*, 10, 282–290.
9. Kling, H. et al. (2012). KGE criterion. *Journal of Hydrology*, 424–425, 264–277.
10. Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization. *IEEE ICNN*, 4, 1942–1948.

---

<div align="center">
  <strong>FlowCast v2</strong> · Kabini River Basin · Batch 2026<br>
  Best Model: SVR-PSO · NSE = 0.9093 · KGE = 0.8866
</div>