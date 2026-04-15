"""
FlowCast v2 — CMIP6 Scenario Forecasting (2025–2030)
Rainfall-Only Model + Physical Post-Processing

Loads pre-trained models from:  saved_models/
Saves forecast outputs to:      outputs/forecast/

Model mapping (matches main.py save names):
  rf_pso.joblib       — RF-PSO traditional model
  xgb_pso.joblib      — XGBoost-PSO traditional model
  svr_pso.joblib      — SVR-PSO traditional model
  lstm.pt             — Standard LSTM
  pi_lstm.pt          — Physics-Informed LSTM
  agru.pt             — Adapted GRU
  dl_scalers.joblib   — StandardScaler for DL input/output
  static_scaler.joblib— StandardScaler for A-GRU static features
"""

import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

print("=" * 60)
print("  PHASE 5: CMIP6 SCENARIO FORECASTING (2025–2030)")
print("  ─── Rainfall-Only Model + Physical Post-Processing ───")
print("=" * 60)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
MODEL_DIR    = BASE_DIR / "saved_models"          # ← where main.py saves models
OUTPUT_DIR   = BASE_DIR / "outputs"               # ← main.py OUTPUT_DIR
FORECAST_DIR = OUTPUT_DIR / "forecast"            # ← forecast-specific outputs
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR  = BASE_DIR / "data"
CMIP6_DIR = BASE_DIR / "cmip6_downloads" / "processed"

features_path = DATA_DIR / "features_dataset.csv"
master_path   = DATA_DIR / "master_dataset.csv"

# --------------------------------------------------
# LOAD SAVED MODELS
#
# Priority order for rainfall-only forecasting:
#   1. XGBoost-PSO  (best traditional model in most runs)
#   2. RF-PSO
#   3. SVR-PSO
#
# DL models (LSTM, PI-LSTM, A-GRU) require torch;
# we attempt to load them as a bonus ensemble if available.
# --------------------------------------------------
print("\n📦 Loading saved model assets...")

# ── Traditional models ──────────────────────────────────────
_trad_candidates = [
    ("XGBoost-PSO", MODEL_DIR / "xgb_pso.joblib"),
    ("RF-PSO",      MODEL_DIR / "rf_pso.joblib"),
    ("SVR-PSO",     MODEL_DIR / "svr_pso.joblib"),
]

ro_model       = None
ro_model_name  = None
for name, path in _trad_candidates:
    if path.exists():
        ro_model      = joblib.load(path)
        ro_model_name = name
        print(f"   ✅ {name} loaded from {path.name}")
        break

if ro_model is None:
    print("❌  No trained traditional model found in saved_models/")
    print("    Run main.py (the full pipeline) first.")
    raise SystemExit(1)

# ── DL scalers ──────────────────────────────────────────────
dl_scalers_path = MODEL_DIR / "dl_scalers.joblib"
scaler_X, scaler_y = None, None
if dl_scalers_path.exists():
    scalers  = joblib.load(dl_scalers_path)
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]
    print(f"   ✅ DL scalers loaded")

# ── DL models (optional) ────────────────────────────────────
dl_models = {}
try:
    import torch
    ROOT = BASE_DIR
    sys.path.insert(0, str(ROOT))
    from deep_learning import LSTMNetwork, PhysicsInformedLSTM, AdaptedGRU, DeepModelTrainer

    for dl_name, dl_path in [
        ("LSTM",    MODEL_DIR / "lstm.pt"),
        ("PI-LSTM", MODEL_DIR / "pi_lstm.pt"),
        ("A-GRU",   MODEL_DIR / "agru.pt"),
    ]:
        if dl_path.exists():
            checkpoint = torch.load(dl_path, map_location="cpu")
            dl_models[dl_name] = checkpoint
            print(f"   ✅ {dl_name} checkpoint found (path: {dl_path.name})")
except ImportError:
    print("   ⚠️  torch not available — DL models skipped")

# ── Hydrological parameters ─────────────────────────────────
hydro_json = MODEL_DIR / "hydro_params.json"
if hydro_json.exists():
    hydro_params       = json.load(open(hydro_json))
    recession_k        = hydro_params["recession_constant"]
    clim_monthly_mean  = {int(k): v for k, v in hydro_params["clim_monthly_mean"].items()}
    clim_monthly_max   = {int(k): v for k, v in hydro_params["clim_monthly_max"].items()}
    mean_annual_q      = hydro_params["mean_annual_q"]
    correction_factor  = hydro_params.get("correction_factor", 0.85)
    p90_threshold      = hydro_params.get("p90_threshold_m3s", 200.0)
    print(f"   ✅ hydro_params.json  (recession k={recession_k:.4f})")
else:
    print("   ⚠️  hydro_params.json not found — estimating from data")
    recession_k       = 0.975
    clim_monthly_mean = None
    clim_monthly_max  = None
    mean_annual_q     = None
    correction_factor = 0.85
    p90_threshold     = 200.0

# Fallback: load forecast_meta.json written by older train_model.py runs
forecast_meta_path = MODEL_DIR / "forecast_meta.json"
if forecast_meta_path.exists():
    meta = json.load(open(forecast_meta_path))
    correction_factor = meta.get("correction_factor", correction_factor)
    p90_threshold     = meta.get("p90_threshold_m3s", p90_threshold)
    print(f"   ✅ forecast_meta.json  (correction={correction_factor:.3f}, p90={p90_threshold:.0f})")

print(f"\n   Active forecast model : {ro_model_name}")
print(f"   Peak correction       : ×{correction_factor:.3f} above {p90_threshold:.0f} m³/s")
print(f"   Recession constant    : k={recession_k:.4f}")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if not features_path.exists():
    print(f"\n❌  features_dataset.csv not found at {features_path}")
    print("    Run feature_engineering.py first.")
    raise SystemExit(1)

df_feat = pd.read_csv(features_path)
df_feat["date"] = pd.to_datetime(df_feat["date"])
df_feat = df_feat.sort_values("date").reset_index(drop=True)

df_master = pd.read_csv(master_path)
df_master["date"] = pd.to_datetime(df_master["date"])
df_master = df_master.sort_values("date").reset_index(drop=True)

print(f"\nHistorical features : {len(df_feat)} rows "
      f"({df_feat['date'].min().date()} → {df_feat['date'].max().date()})")

# Climatology fallback
if clim_monthly_mean is None:
    df_master["month"]  = df_master["date"].dt.month
    clim_monthly_mean   = df_master.groupby("month")["q_upstream_mk"].mean().to_dict()
    clim_monthly_max    = df_master.groupby("month")["q_upstream_mk"].max().to_dict()
    mean_annual_q       = float(df_master["q_upstream_mk"].mean())


# ══════════════════════════════════════════════════════════
#  ROLLING CONVENTION DETECTION
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 1: Detecting rolling-window convention")
print("=" * 60)

RAIN_STD_COL = next((c for c in ["rainfall_std_mm", "rain_std_mm"]
                     if c in df_master.columns), None)

DIAG_START = pd.Timestamp("2005-01-01")
DIAG_END   = pd.Timestamp("2011-12-31")
fdiag = df_feat[(df_feat["date"] >= DIAG_START) & (df_feat["date"] <= DIAG_END)].reset_index(drop=True)
mdiag = df_master[(df_master["date"] >= DIAG_START) & (df_master["date"] <= DIAG_END)].reset_index(drop=True)
mdiag = mdiag[mdiag["date"].isin(fdiag["date"])].reset_index(drop=True)
fdiag = fdiag[fdiag["date"].isin(mdiag["date"])].reset_index(drop=True)

# Feature columns expected by the loaded model
# (model objects from joblib expose feature_names_in_ for tree models)
ro_feature_cols = getattr(ro_model, "feature_names_in_", None)
if ro_feature_cols is None:
    # Fall back: use all numeric columns except date / target
    ro_feature_cols = [c for c in df_feat.columns if c not in ("date", "log_q", "q_upstream_mk")]
ro_feature_cols = list(ro_feature_cols)

check_cols  = [c for c in fdiag.columns
               if c.startswith("log_rain_roll_") and "rollstd" not in c and c in ro_feature_cols]
diag_rain   = mdiag["rainfall_max_mm"].astype(float)

best_rs, best_ss, best_r = 1, 1, -1.0
for rs in [0, 1]:
    for ss in [0, 1]:
        r_roll = diag_rain.shift(rs) if rs else diag_rain
        corrs  = []
        for col in check_cols:
            w    = int(col.split("_roll_")[1].replace("d", ""))
            pred = np.log1p(r_roll.rolling(w, min_periods=1).sum().fillna(0).clip(lower=0))
            act  = fdiag[col].values.astype(float)
            if np.std(pred) > 0:
                corrs.append(float(np.corrcoef(act, pred.values)[0, 1]))
        mean_r = float(np.mean(corrs)) if corrs else 0.0
        marker = ""
        if mean_r > best_r:
            best_r, best_rs, best_ss = mean_r, rs, ss
            marker = " ← best"
        print(f"   roll_shift={rs}  std_shift={ss}  mean_r={mean_r:.4f}{marker}")

print(f"\n   ✅ roll_shift={best_rs}, std_shift={best_ss}  (mean_r={best_r:.4f})")


# ══════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def precompute_rain_features(dates_series, rain_series, rain_std_series=None):
    """Build all rainfall-derived features (no discharge lags)."""
    rain  = pd.Series(rain_series.values.astype(float))
    dates = pd.to_datetime(dates_series.values)
    df_c  = pd.DataFrame()

    for lag in range(1, 8):
        df_c[f"log_rain_lag_{lag}d"] = np.log1p(rain.shift(lag).fillna(0).clip(lower=0))

    r_roll = rain.shift(best_rs) if best_rs else rain
    for w in [3, 7, 14, 30]:
        df_c[f"log_rain_roll_{w}d"] = np.log1p(
            r_roll.rolling(w, min_periods=1).sum().fillna(0).clip(lower=0))

    r_std = rain.shift(best_ss) if best_ss else rain
    for w in [7, 14]:
        df_c[f"log_rain_rollstd_{w}d"] = np.log1p(
            r_std.rolling(w, min_periods=1).std().fillna(0).clip(lower=0))

    if rain_std_series is not None:
        df_c["log_rainfall_std"] = np.log1p(
            pd.Series(rain_std_series.values.astype(float)).fillna(0).clip(lower=0).values)
    else:
        df_c["log_rainfall_std"] = 0.0

    rain_vals = rain.values.astype(float)
    for k_val, k_name in [(0.85, "fast"), (0.92, "med"), (0.97, "slow")]:
        api = np.zeros(len(rain_vals))
        api[0] = rain_vals[0] if not np.isnan(rain_vals[0]) else 0.0
        for t in range(1, len(rain_vals)):
            rv = rain_vals[t] if not np.isnan(rain_vals[t]) else 0.0
            api[t] = k_val * api[t - 1] + rv
        df_c[f"log_api_{k_name}"] = np.log1p(
            pd.Series(api).shift(1).fillna(0).clip(lower=0).values)

    dry_spell = np.zeros(len(rain_vals))
    for t in range(1, len(rain_vals)):
        rv = rain_vals[t] if not np.isnan(rain_vals[t]) else 0.0
        dry_spell[t] = dry_spell[t - 1] + 1 if rv < 1.0 else 0
    df_c["log_dry_spell"] = np.log1p(dry_spell)

    cum_monsoon = np.zeros(len(rain_vals))
    for t in range(1, len(rain_vals)):
        d = dates[t]; m = d.month; rv = rain_vals[t] if not np.isnan(rain_vals[t]) else 0.0
        if m == 6 and d.day == 1:
            cum_monsoon[t] = rv
        elif 6 <= m <= 11:
            cum_monsoon[t] = cum_monsoon[t - 1] + rv
    df_c["log_cum_monsoon"] = np.log1p(
        pd.Series(cum_monsoon).shift(1).fillna(0).clip(lower=0).values)

    ds = pd.Series(dates)
    df_c["month_sin"]  = np.sin(2 * np.pi * ds.dt.month     / 12).values
    df_c["month_cos"]  = np.cos(2 * np.pi * ds.dt.month     / 12).values
    df_c["doy_sin"]    = np.sin(2 * np.pi * ds.dt.dayofyear / 365).values
    df_c["doy_cos"]    = np.cos(2 * np.pi * ds.dt.dayofyear / 365).values
    df_c["is_monsoon"] = ds.dt.month.between(6, 9).astype(int).values

    df_c["api_slow_x_rain7d"]    = df_c["log_api_slow"]  * df_c["log_rain_roll_7d"]
    df_c["api_med_x_rain3d"]     = df_c["log_api_med"]   * df_c["log_rain_roll_3d"]
    df_c["api_slow_x_monsoon"]   = df_c["log_api_slow"]  * df_c["is_monsoon"]
    df_c["cum_monsoon_x_rain7d"] = df_c["log_cum_monsoon"] * df_c["log_rain_roll_7d"]

    return df_c.reset_index(drop=True)


def build_feat_with_seed(seed_df, target_df):
    """Build rainfall features with historical warm-up seed."""
    s_rain  = seed_df["rainfall_max_mm"].reset_index(drop=True) \
              if "rainfall_max_mm" in seed_df.columns else pd.Series([0.0] * len(seed_df))
    t_rain  = target_df["rainfall_max_mm"].reset_index(drop=True)
    full_rain  = pd.concat([s_rain,  t_rain], ignore_index=True)
    full_dates = pd.concat([seed_df["date"].reset_index(drop=True),
                             target_df["date"].reset_index(drop=True)], ignore_index=True)
    s_std = seed_df[RAIN_STD_COL].reset_index(drop=True) \
            if RAIN_STD_COL and RAIN_STD_COL in seed_df.columns else None
    t_std = target_df[RAIN_STD_COL].reset_index(drop=True) \
            if RAIN_STD_COL and RAIN_STD_COL in target_df.columns else None
    full_std = pd.concat([s_std, t_std], ignore_index=True) \
               if (s_std is not None and t_std is not None) else None
    feat_full = precompute_rain_features(full_dates, full_rain, full_std)
    return feat_full.iloc[len(seed_df):].reset_index(drop=True)


# ══════════════════════════════════════════════════════════
#  PHYSICAL POST-PROCESSOR
# ══════════════════════════════════════════════════════════

def physical_postprocess(raw_predictions, dates, rainfall,
                         recession_k, clim_monthly_mean,
                         clim_monthly_max, correction_factor,
                         p90_threshold, seed_q=None):
    n = len(raw_predictions)
    processed = np.zeros(n)
    dates = pd.DatetimeIndex(dates)

    if seed_q is None:
        first_month = dates[0].month
        seed_q = clim_monthly_mean.get(first_month, np.mean(list(clim_monthly_mean.values())))

    prev_q = seed_q
    for i in range(n):
        month   = dates[i].month
        pred_q  = max(float(raw_predictions[i]), 0.0)
        rain_r  = np.mean(rainfall[max(0, i - 2):i + 1]) if i < len(rainfall) else 0.0
        rec_q   = prev_q * recession_k

        if rain_r < 2.0:
            pred_q = min(pred_q, max(rec_q, pred_q * 0.3))
        else:
            pred_q = max(pred_q, rec_q * 0.8)

        m_max  = clim_monthly_max.get(month, max(clim_monthly_mean.values()) * 3)
        pred_q = min(pred_q, m_max * 1.5)
        if pred_q > p90_threshold:
            pred_q *= correction_factor
        pred_q = max(pred_q, 0.0)
        processed[i] = pred_q
        prev_q       = pred_q

    result = pd.Series(processed)
    monsoon_mask = pd.DatetimeIndex(dates).month.isin([6, 7, 8, 9])
    smooth = result.copy()
    for i in range(1, n):
        alpha = 2.0 / (3 + 1) if monsoon_mask[i] else 2.0 / (7 + 1)
        smooth.iloc[i] = alpha * result.iloc[i] + (1 - alpha) * smooth.iloc[i - 1]
    return smooth.values


def predict_with_ro_model(rain_feat_df, feature_cols, model):
    """Direct prediction from a scikit-learn-compatible model."""
    X = pd.DataFrame([
        {col: rain_feat_df.iloc[i].get(col, 0.0) for col in feature_cols}
        for i in range(len(rain_feat_df))
    ])
    raw = np.expm1(model.predict(X))
    return np.maximum(raw, 0.0)


# ══════════════════════════════════════════════════════════
#  STEP 2: HINDCAST VALIDATION (2015–2020)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: Hindcast validation (2015–2020)")
print("=" * 60)

TEST_START = pd.Timestamp("2015-01-01")
df_test    = df_feat[df_feat["date"] >= TEST_START].copy().reset_index(drop=True)
y_obs      = np.expm1(df_test["log_q"].values)
test_dates = pd.DatetimeIndex(df_test["date"])

df_master["month"] = df_master["date"].dt.month
clim_monthly_q     = df_master.groupby("month")["q_upstream_mk"].median()
ANNUAL_RESET_Q     = float(clim_monthly_q[1])

seed_hc     = df_master[df_master["date"] < TEST_START].tail(30)
test_master = df_master[(df_master["date"] >= TEST_START) &
                        (df_master["date"].isin(df_test["date"]))].reset_index(drop=True)

# Build rainfall features for hindcast window
ro_feat_hc = build_feat_with_seed(seed_hc, test_master)
raw_ro_hc  = predict_with_ro_model(ro_feat_hc, ro_feature_cols, ro_model)

seed_q_val = float(seed_hc["q_upstream_mk"].iloc[-1])
ro_post_hc = physical_postprocess(
    raw_ro_hc, test_dates, test_master["rainfall_max_mm"].values,
    recession_k, clim_monthly_mean, clim_monthly_max,
    correction_factor, p90_threshold, seed_q=seed_q_val,
)

nse_ro  = 1 - np.sum((y_obs - ro_post_hc) ** 2) / np.sum((y_obs - y_obs.mean()) ** 2)
rmse_ro = np.sqrt(mean_squared_error(y_obs, ro_post_hc))
corr_ro = float(np.corrcoef(ro_post_hc, y_obs)[0, 1])

print(f"   Rainfall-only + post-process:  NSE={nse_ro:.4f}  RMSE={rmse_ro:.2f}  r={corr_ro:.4f}")

THRESHOLD = 0.40
if nse_ro >= THRESHOLD:
    print(f"   ✅ Hindcast NSE={nse_ro:.4f} ≥ {THRESHOLD} — validated for CMIP6 forecasting.")
else:
    print(f"   ⚠️  Hindcast NSE={nse_ro:.4f} < {THRESHOLD} — directional projections only.")

# Per-year NSE
annual_nse = []
for yr in sorted(set(df_test["date"].dt.year)):
    mask = (df_test["date"].dt.year == yr).values
    if mask.sum() < 30: continue
    obs_yr, pred_yr = y_obs[mask], ro_post_hc[mask]
    nse_yr = 1 - np.sum((obs_yr - pred_yr) ** 2) / np.sum((obs_yr - obs_yr.mean()) ** 2)
    annual_nse.append((yr, nse_yr))

print(f"\n   Per-year NSE:")
for yr, nse_yr in annual_nse:
    print(f"     {'✅' if nse_yr >= 0.3 else '⚠️'} {yr}: NSE={nse_yr:.4f}")

# Hindcast plot
if annual_nse:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
    td = df_test["date"].values
    axes[0].plot(td, y_obs,       color="#1a1a1a", lw=0.8, alpha=0.8, label="Observed")
    axes[0].plot(td, ro_post_hc,  color="#2E86C1", lw=0.9, alpha=0.85, linestyle="--",
                 label=f"Rainfall-Only + PP  NSE={nse_ro:.3f}")
    axes[0].set_ylabel("Discharge (m³/s)")
    axes[0].set_title(f"Hindcast Validation — {ro_model_name}")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.15)

    yr_vals, nse_vals = zip(*annual_nse)
    axes[1].bar(yr_vals, nse_vals,
                color=["#2ECC71" if n >= 0.3 else "#E74C3C" for n in nse_vals],
                alpha=0.8, edgecolor="white")
    axes[1].axhline(0.3, color="gray", linestyle="--", lw=0.8)
    axes[1].set_ylabel("NSE"); axes[1].set_xlabel("Year")
    axes[1].set_title("Per-Year NSE — Rainfall-Only + Post-Processing")
    axes[1].grid(True, alpha=0.15, axis="y")
    plt.tight_layout()
    plt.savefig(FORECAST_DIR / "hindcast_validation.png", dpi=150)
    plt.close()
    print(f"\n   📸 hindcast_validation.png")

pd.DataFrame({
    "date":          df_test["date"].values,
    "observed":      y_obs,
    "rainfall_only": ro_post_hc,
}).to_csv(FORECAST_DIR / "hindcast_validation.csv", index=False)
print(f"   💾 hindcast_validation.csv")


# ══════════════════════════════════════════════════════════
#  STEP 3: CMIP6 SCENARIO FORECASTS
# ══════════════════════════════════════════════════════════
SSPS       = ["ssp245", "ssp585"]
SSP_LABELS = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
SSP_COLORS = {"ssp245": "#8E44AD",  "ssp585": "#C0392B"}

print("\n" + "=" * 60)
print("  STEP 3: Checking for CMIP6 data")
print("=" * 60)

# Accept both processed/ subfolder and direct seed CSVs from cmip6_downloads/
cmip6_available = {}
for ssp in SSPS:
    candidates = [
        CMIP6_DIR / f"forecast_input_{ssp}.csv",
        BASE_DIR / "cmip6_downloads" / f"forecast_seed_{ssp}.csv",
    ]
    for fpath in candidates:
        if fpath.exists():
            cmip6_available[ssp] = fpath
            print(f"   ✅ {fpath.name}  ({fpath})")
            break
    else:
        print(f"   ❌ Missing CMIP6 data for {ssp}")

if not cmip6_available:
    print("\n   No CMIP6 data found.")
    print(f"   Expected: {CMIP6_DIR}/forecast_input_ssp*.csv")
    print(f"         or: {BASE_DIR}/cmip6_downloads/forecast_seed_ssp*.csv")
    raise SystemExit(1)

print("\n" + "=" * 60)
print(f"  STEP 4: CMIP6 forecasts  [hindcast NSE={nse_ro:.3f}]")
print("=" * 60)

seed_fc   = df_master.tail(30)
seed_q_fc = float(df_master["q_upstream_mk"].iloc[-1])

scenario_results = {}

for ssp, fpath in cmip6_available.items():
    label = SSP_LABELS[ssp]
    print(f"\n   🌍 {label}...")

    df_cmip = pd.read_csv(fpath)
    df_cmip["date"] = pd.to_datetime(df_cmip["date"])
    df_cmip = df_cmip.sort_values("date").reset_index(drop=True)
    cmip_dates = pd.DatetimeIndex(df_cmip["date"])
    print(f"      {cmip_dates[0].date()} → {cmip_dates[-1].date()} ({len(cmip_dates)} days)")

    rain_feat_cmip = build_feat_with_seed(seed_fc, df_cmip)
    raw_pred       = predict_with_ro_model(rain_feat_cmip, ro_feature_cols, ro_model)

    q_pred = physical_postprocess(
        raw_pred, cmip_dates, df_cmip["rainfall_max_mm"].values,
        recession_k, clim_monthly_mean, clim_monthly_max,
        correction_factor, p90_threshold, seed_q=seed_q_fc,
    )

    future_df         = pd.DataFrame({"date": cmip_dates, "Q": q_pred,
                                      "P": df_cmip["rainfall_max_mm"].values})
    future_df["year"] = future_df["date"].dt.year
    annual = future_df.groupby("year").agg(
        mean_Q=("Q", "mean"), max_Q=("Q", "max"), total_P=("P", "sum"))

    scenario_results[label] = {
        "ssp": ssp, "dates": cmip_dates,
        "rainfall": df_cmip["rainfall_max_mm"].values,
        "discharge": q_pred, "annual": annual,
    }

    print(f"      {'Year':>6}  {'Mean Q':>10}  {'Peak Q':>10}  {'Rain mm':>10}")
    for yr, r in annual.iterrows():
        print(f"      {yr:>6}  {r['mean_Q']:>10.1f}  {r['max_Q']:>10.1f}  {r['total_P']:>10.0f}")


# --------------------------------------------------
# PLOTS
# --------------------------------------------------
print("\n📸 Generating plots...")
reliability = f"Hindcast NSE={nse_ro:.3f} (rainfall-only, {ro_model_name})"

# 1 — Full forecast time-series
fig, ax = plt.subplots(figsize=(16, 6))
ht = df_master.tail(1095)
ax.plot(ht["date"], ht["q_upstream_mk"],
        color="#1a1a1a", lw=0.7, alpha=0.5, label="Historical observed")
for label, data in scenario_results.items():
    ax.plot(data["dates"], data["discharge"],
            color=SSP_COLORS[data["ssp"]], lw=0.8, alpha=0.85, label=label)
fs = min(data["dates"][0] for data in scenario_results.values())
ax.axvline(fs, color="gray", linestyle="--", lw=0.8, alpha=0.5)
ax.text(fs, ax.get_ylim()[1] * 0.92, "  Forecast →", fontsize=9, color="gray")
ax.set_title(f"Kabini Discharge — CMIP6 Forecasts  [{reliability}]")
ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_forecast_full.png", dpi=150)
plt.close()
print(f"   📸 cmip6_forecast_full.png")

# 2 — 2026 monsoon zoom
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 2.5]})
z0, z1 = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-11-30")
fl = list(scenario_results.keys())[0]
fd = scenario_results[fl]
zm = (fd["dates"] >= z0) & (fd["dates"] <= z1)
axes[0].bar(fd["dates"][zm], fd["rainfall"][zm], color="#3B8BD4", alpha=0.6, width=1)
axes[0].invert_yaxis(); axes[0].set_ylabel("Rainfall (mm)")
axes[0].set_title("2026 Monsoon Zoom"); axes[0].set_xlim(z0, z1)
for label, data in scenario_results.items():
    zm2 = (data["dates"] >= z0) & (data["dates"] <= z1)
    axes[1].plot(data["dates"][zm2], data["discharge"][zm2],
                 color=SSP_COLORS[data["ssp"]], lw=1.3, alpha=0.85, label=label)
axes[1].set_xlabel("Date"); axes[1].set_ylabel("Discharge (m³/s)")
axes[1].legend(fontsize=10); axes[1].set_xlim(z0, z1); axes[1].grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_monsoon_2026.png", dpi=150)
plt.close()
print(f"   📸 cmip6_monsoon_2026.png")

# 3 — Annual peak discharge
fig, ax = plt.subplots(figsize=(12, 5))
ham = df_master.groupby(df_master["date"].dt.year)["q_upstream_mk"].max()
ax.bar(ham.index, ham.values, color="#BDC3C7", alpha=0.6,
       edgecolor="white", lw=0.5, label="Historical")
bw = 0.35
for i, (label, data) in enumerate(scenario_results.items()):
    yrs = sorted(data["annual"].index)
    ax.bar([y + (i - 0.5) * bw for y in yrs],
           [data["annual"].loc[y, "max_Q"] for y in yrs],
           bw, color=SSP_COLORS[data["ssp"]], alpha=0.85, edgecolor="white", lw=0.5, label=label)
ax.set_title("Annual Peak Discharge — Historical vs CMIP6")
ax.set_xlabel("Year"); ax.set_ylabel("Peak Discharge (m³/s)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.15, axis="y")
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_annual_peaks.png", dpi=150)
plt.close()
print(f"   📸 cmip6_annual_peaks.png")

# 4 — Forecast envelope (if 2+ scenarios)
if len(scenario_results) >= 2:
    fig, ax = plt.subplots(figsize=(14, 5))
    lbls  = list(scenario_results.keys())
    cd    = scenario_results[lbls[0]]["dates"]
    all_q = np.column_stack([scenario_results[l]["discharge"] for l in lbls])
    ax.fill_between(cd,
                    pd.Series(all_q.min(axis=1)).rolling(30, min_periods=1).mean(),
                    pd.Series(all_q.max(axis=1)).rolling(30, min_periods=1).mean(),
                    alpha=0.25, color="#8E44AD", label="SSP range (30-day smooth)")
    ax.plot(cd, pd.Series(all_q.mean(axis=1)).rolling(30, min_periods=1).mean(),
            color="#8E44AD", lw=1.5, label="Ensemble mean")
    ax.set_title("CMIP6 Forecast Envelope — SSP2-4.5 to SSP5-8.5")
    ax.set_xlabel("Date"); ax.set_ylabel("Discharge (m³/s)")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(FORECAST_DIR / "cmip6_forecast_envelope.png", dpi=150)
    plt.close()
    print(f"   📸 cmip6_forecast_envelope.png")

# 5 — Annual mean
fig, ax = plt.subplots(figsize=(10, 5))
for i, (label, data) in enumerate(scenario_results.items()):
    yrs = sorted(data["annual"].index)
    ax.bar([y + (i - 0.5) * 0.35 for y in yrs],
           [data["annual"].loc[y, "mean_Q"] for y in yrs],
           0.35, color=SSP_COLORS[data["ssp"]], alpha=0.85,
           edgecolor="white", lw=0.5, label=label)
ax.set_title("Projected Annual Mean Discharge (2025–2030)")
ax.set_xlabel("Year"); ax.set_ylabel("Mean Discharge (m³/s)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.15, axis="y")
plt.tight_layout()
plt.savefig(FORECAST_DIR / "cmip6_annual_mean.png", dpi=150)
plt.close()
print(f"   📸 cmip6_annual_mean.png")


# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------
print("\n💾 Saving forecast data...")

# Per-scenario daily CSV (matches app.py expectations)
for label, data in scenario_results.items():
    out_path = FORECAST_DIR / f"forecast_{data['ssp']}.csv"
    pd.DataFrame({
        "date":           data["dates"],
        "rainfall_mm":    data["rainfall"],
        "discharge_m3s":  data["discharge"],
    }).to_csv(out_path, index=False)
    print(f"   💾 {out_path.name}")

# Annual summary CSV (matches app.py tab 7)
summary = []
for label, data in scenario_results.items():
    for yr, r in data["annual"].iterrows():
        summary.append({
            "scenario":             label,
            "year":                 yr,
            "mean_discharge_m3s":   round(r["mean_Q"], 2),
            "peak_discharge_m3s":   round(r["max_Q"],  2),
            "total_rainfall_mm":    round(r["total_P"], 1),
        })
annual_summary_path = FORECAST_DIR / "forecast_annual_summary.csv"
pd.DataFrame(summary).to_csv(annual_summary_path, index=False)
print(f"   💾 forecast_annual_summary.csv")

# Reliability JSON for dashboard
json.dump({
    "hindcast_nse":     round(nse_ro, 4),
    "hindcast_rmse":    round(rmse_ro, 2),
    "model_used":       ro_model_name,
    "validated":        bool(nse_ro >= THRESHOLD),
}, open(FORECAST_DIR / "forecast_reliability.json", "w"), indent=2)
print(f"   💾 forecast_reliability.json")

print(f"\n{'='*60}")
print(f"  FORECAST RELIABILITY SUMMARY")
print(f"{'='*60}")
print(f"  Model used                       : {ro_model_name}")
print(f"  Rainfall-only + post-processing  : NSE={nse_ro:.4f}  RMSE={rmse_ro:.2f}")
print(f"  Rolling convention               : roll_shift={best_rs}, std_shift={best_ss}")
verdict = "✅ Validated for scenario projections." if nse_ro >= THRESHOLD \
          else "⚠️  Directional projections only — interpret relative differences."
print(f"  Verdict: {verdict}")
print(f"\n✅ Forecasting complete. Outputs in: {FORECAST_DIR}")
print("=" * 60)