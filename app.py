"""
FlowCast v2 — Production Streamlit Dashboard
=============================================
Kabini River Basin Discharge Prediction & Climate Projection App

Features:
  • Live model training with progress feedback
  • Observed-vs-predicted hydrograph viewer
  • Interactive metric tables per model
  • CMIP6 climate projection browser (SSP245 / SSP585)
  • Future discharge forecast viewer (per period, per GCM)
  • Seasonal breakdown charts
  • Model comparison leaderboard
  • Data upload for custom CHIRPS / discharge CSVs
  • Download buttons for all outputs

Run:
    streamlit run app.py
"""

import io
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlowCast v2 — Kabini Basin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f0f4f8;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card h4 { margin: 0 0 0.25rem; color: #1e3a5f; font-size: 0.85rem; }
    .metric-card .val { font-size: 1.5rem; font-weight: 700; color: #1f4e79; }
    .good  { border-left-color: #2ca02c; }
    .warn  { border-left-color: #ff7f0e; }
    .bad   { border-left-color: #d62728; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CACHED RESOURCE LOADERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_observed_data(use_synthetic: bool = False) -> pd.DataFrame:
    from modules.loader import KabiniDataLoader
    loader = KabiniDataLoader(data_dir=ROOT / "data")
    chirps_path    = ROOT / "data" / "chirps_kabini_daily.csv"
    discharge_path = ROOT / "data" / "discharge_daily_observed.csv"

    if use_synthetic or not chirps_path.exists():
        return loader.generate_synthetic_data()
    return loader.load_real_data(
        precip_path=str(chirps_path),
        discharge_path=str(discharge_path),
    )


@st.cache_data(show_spinner=False)
def load_feature_engineered(use_synthetic: bool = False) -> tuple:
    from modules.config import data_cfg
    from modules.engineer import HydroFeatureEngineer
    df = load_observed_data(use_synthetic)
    eng = HydroFeatureEngineer(df)
    df_feat = eng.build_all_features()
    feat_set  = "M1_full"
    feat_list = [f for f in data_cfg.feature_combinations[feat_set] if f in df_feat.columns]
    return df, df_feat, feat_list


@st.cache_resource(show_spinner=False)
def load_saved_models(n_features: int) -> Dict:
    try:
        from modules.forecast import _load_all_models
        return _load_all_models(n_features)
    except Exception as e:
        return {}


@st.cache_data(show_spinner=False)
def load_metrics_json() -> Optional[Dict]:
    p = ROOT / "outputs" / "model_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_forecast_results() -> Optional[Dict]:
    p = ROOT / "outputs" / "forecast_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_forecast_summary() -> Optional[pd.DataFrame]:
    p = ROOT / "outputs" / "forecast_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    p2 = ROOT / "outputs" / "cmip6_projections_summary.csv"
    if p2.exists():
        return pd.read_csv(p2)
    return None


@st.cache_data(show_spinner=False)
def load_ensemble_csv() -> Optional[pd.DataFrame]:
    p = ROOT / "outputs" / "ensemble_predictions.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


# ════════════════════════════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ════════════════════════════════════════════════════════════════════════════

PALETTE = px.colors.qualitative.Plotly

def _color(i): return PALETTE[i % len(PALETTE)]


def hydrograph_fig(dates, obs, pred, model_name, metrics=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=obs, name="Observed",
        line=dict(color="#1f1f1f", width=1.5), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pred, name=f"Predicted ({model_name})",
        line=dict(color="#1f77b4", width=1.5), opacity=0.85,
    ))
    # Shaded over/under areas
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates)[::-1],
        y=list(np.where(pred > obs, pred, obs)) + list(np.where(pred > obs, obs, obs))[::-1],
        fill="toself", fillcolor="rgba(214,39,40,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="Overestimate", showlegend=False,
    ))

    title = f"Hydrograph — {model_name}"
    if metrics:
        title += f"  |  NSE={metrics['NSE']:.4f}  KGE={metrics['KGE']:.4f}"

    fig.update_layout(
        title=title,
        xaxis_title="Date", yaxis_title="Discharge (cumecs)",
        hovermode="x unified",
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


def scatter_fig(obs, pred, model_name, r2=None, rmse=None):
    mn, mx = min(np.min(obs), np.min(pred)), max(np.max(obs), np.max(pred))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color="#888", dash="dash"), name="1:1 Line",
    ))
    fig.add_trace(go.Scatter(
        x=obs, y=pred, mode="markers",
        marker=dict(color="#1f77b4", size=4, opacity=0.5),
        name="Data points",
    ))
    title = f"Observed vs Predicted — {model_name}"
    if r2 is not None:
        title += f"  |  R²={r2:.4f}  RMSE={rmse:.2f}"
    fig.update_layout(
        title=title,
        xaxis_title="Observed (cumecs)", yaxis_title="Predicted (cumecs)",
        plot_bgcolor="white", height=420,
    )
    return fig


def metrics_color(val: float, metric: str) -> str:
    if metric in ("NSE", "KGE", "R2", "R"):
        if val >= 0.75: return "good"
        if val >= 0.5:  return "warn"
        return "bad"
    if metric == "PBIAS":
        if abs(val) <= 10: return "good"
        if abs(val) <= 25: return "warn"
        return "bad"
    return ""


def metric_card(label: str, value, metric: str = ""):
    cls = metrics_color(value, metric) if isinstance(value, float) else ""
    val_str = f"{value:.4f}" if isinstance(value, float) and metric in ("NSE","KGE","R2","R") else \
              f"{value:.2f}" if isinstance(value, float) else str(value)
    return f"""
    <div class="metric-card {cls}">
      <h4>{label}</h4>
      <div class="val">{val_str}</div>
    </div>"""


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/water.png", width=60)
        st.title("FlowCast v2")
        st.caption("Kabini River Basin\nDischarge Prediction")
        st.divider()

        use_synthetic = st.toggle("Use synthetic data", value=False,
                                   help="Disable if real CHIRPS/discharge CSVs exist")

        st.markdown("### ⚙️ Pipeline Controls")
        run_pipeline = st.button("▶ Run Full Pipeline", use_container_width=True, type="primary")
        run_forecast  = st.button("📈 Run Forecast Only", use_container_width=True)

        st.divider()
        st.markdown("### 🎯 Model Selection")
        avail_models = ["RF", "SVR", "XGB", "LSTM", "PI-LSTM", "A-GRU"]
        selected_models = st.multiselect("Models to train/display",
                                          avail_models, default=avail_models)

        st.divider()
        st.markdown("### 🌏 CMIP6 Settings")
        scenarios = st.multiselect("Scenarios", ["ssp245", "ssp585"],
                                    default=["ssp245", "ssp585"])
        skip_cmip6 = st.toggle("Skip CMIP6 projections", value=False)

        st.divider()
        st.markdown("### 📁 Custom Data Upload")
        chirps_upload    = st.file_uploader("CHIRPS CSV (precip)", type=["csv"])
        discharge_upload = st.file_uploader("Discharge CSV", type=["csv"])

        if chirps_upload and discharge_upload:
            # Save uploaded files temporarily
            (ROOT / "data").mkdir(exist_ok=True)
            with open(ROOT / "data" / "chirps_kabini_daily.csv", "wb") as f:
                f.write(chirps_upload.read())
            with open(ROOT / "data" / "discharge_daily_observed.csv", "wb") as f:
                f.write(discharge_upload.read())
            st.success("✓ Files saved — refresh to reload data.")
            load_observed_data.clear()
            load_feature_engineered.clear()

        st.divider()
        st.markdown("### ℹ️ About")
        st.caption(
            "FlowCast v2 implements RF-PSO, SVR-PSO, XGBoost, "
            "Physics-Informed LSTM, and A-GRU for hydrological "
            "forecasting under CMIP6 SSP245/SSP585 scenarios."
        )

    return {
        "use_synthetic": use_synthetic,
        "run_pipeline": run_pipeline,
        "run_forecast": run_forecast,
        "selected_models": selected_models,
        "scenarios": scenarios,
        "skip_cmip6": skip_cmip6,
    }


# ════════════════════════════════════════════════════════════════════════════
#  TAB RENDERERS
# ════════════════════════════════════════════════════════════════════════════

# ── Tab 1: Overview ─────────────────────────────────────────────────────────

def render_overview(obs_df: pd.DataFrame):
    st.markdown('<div class="section-header">📊 Basin Data Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record Length", f"{len(obs_df):,} days")
    c2.metric("Date Range",
              f"{obs_df.index.min().date()} → {obs_df.index.max().date()}")
    c3.metric("Mean Discharge", f"{obs_df['discharge'].mean():.2f} cumecs")
    c4.metric("Mean Precip", f"{obs_df['precip'].mean():.2f} mm/day")

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**Observed Precipitation & Discharge (Full Record)**")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.35, 0.65], vertical_spacing=0.05)

        fig.add_trace(go.Bar(
            x=obs_df.index, y=obs_df["precip"],
            marker_color="#7fc7ff", name="Precipitation",
            opacity=0.8,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=obs_df.index, y=obs_df["discharge"],
            line=dict(color="#1f77b4", width=1.0),
            name="Discharge",
        ), row=2, col=1)

        fig.update_yaxes(title_text="Precip (mm)", row=1, col=1, autorange="reversed")
        fig.update_yaxes(title_text="Discharge (cumecs)", row=2, col=1)
        fig.update_layout(height=420, showlegend=False, plot_bgcolor="white",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Monthly Climatology**")
        monthly = obs_df.groupby(obs_df.index.month).agg(
            precip=("precip", "mean"),
            discharge=("discharge", "mean"),
        )
        monthly.index = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=monthly.index, y=monthly["precip"],
                               name="Precip", marker_color="#aec7e8", yaxis="y1"))
        fig2.add_trace(go.Scatter(x=monthly.index, y=monthly["discharge"],
                                   name="Discharge", line=dict(color="#1f77b4"),
                                   yaxis="y2"))
        fig2.update_layout(
            yaxis=dict(title="Precip (mm)", side="left"),
            yaxis2=dict(title="Discharge (cumecs)", side="right", overlaying="y"),
            legend=dict(orientation="h", y=-0.25),
            height=420, plot_bgcolor="white", margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("**Summary Statistics**")
    desc = obs_df[["precip", "discharge"]].describe().T
    st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)


# ── Tab 2: Model Performance ─────────────────────────────────────────────────

def render_model_performance(
    obs_df: pd.DataFrame,
    df_feat: pd.DataFrame,
    feat_list: list,
    saved_models: dict,
    metrics_json: Optional[dict],
    selected_models: list,
):
    st.markdown('<div class="section-header">🏆 Model Performance</div>', unsafe_allow_html=True)

    if not saved_models and not metrics_json:
        st.info("ℹ️ No saved models found. Use the sidebar to run the pipeline first.")
        return

    # ── Metric leaderboard ──────────────────────────────────────────────────
    if metrics_json:
        rows = []
        for name, m in metrics_json.items():
            if name not in selected_models:
                continue
            overall = m.get("overall", m)
            rows.append({
                "Model": name,
                "NSE":   overall.get("NSE", np.nan),
                "KGE":   overall.get("KGE", np.nan),
                "RMSE":  overall.get("RMSE", np.nan),
                "MAE":   overall.get("MAE",  np.nan),
                "R²":    overall.get("R2",   np.nan),
                "PBIAS": overall.get("PBIAS",np.nan),
            })

        if rows:
            lb_df = pd.DataFrame(rows).sort_values("NSE", ascending=False)
            st.markdown("**Leaderboard (sorted by NSE)**")

            # Colour-coded table
            def colour_nse(v):
                if v >= 0.75: return "background-color: #d4edda"
                if v >= 0.5:  return "background-color: #fff3cd"
                return "background-color: #f8d7da"

            st.dataframe(
                lb_df.style
                     .applymap(colour_nse, subset=["NSE", "KGE"])
                     .format({c: "{:.4f}" for c in ["NSE","KGE","R²"]}
                              | {"RMSE": "{:.2f}", "MAE": "{:.2f}", "PBIAS": "{:.1f}"}),
                use_container_width=True,
                height=min(300, 50 + 35 * len(lb_df)),
            )

            # Radar chart
            st.markdown("**Radar Comparison**")
            categories = ["NSE", "KGE", "R²"]
            fig_radar = go.Figure()
            for i, (_, row) in enumerate(lb_df.iterrows()):
                vals = [max(row[c], 0) for c in categories]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    name=row["Model"],
                    line=dict(color=_color(i)),
                    fill="toself", opacity=0.25,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1])),
                showlegend=True, height=380,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── Per-model hydrograph ─────────────────────────────────────────────────
    if saved_models:
        st.divider()
        st.markdown('<div class="section-header">📉 Predicted vs Observed Hydrographs</div>',
                    unsafe_allow_html=True)

        from modules.config import data_cfg
        from modules.loader import train_val_test_split
        _, val_df, test_df = train_val_test_split(
            df_feat, train_end=data_cfg.train_end, val_end=data_cfg.val_end
        )
        X_test = test_df[feat_list].values.astype(np.float32)
        y_test = test_df["discharge"].values.astype(np.float32)

        model_select = st.selectbox("Select model to inspect",
                                     [m for m in saved_models if m in selected_models])

        if model_select and model_select in saved_models:
            model = saved_models[model_select]
            pred  = model.predict(X_test)
            min_l = min(len(pred), len(y_test))
            obs_a, pred_a = y_test[-min_l:], pred[-min_l:]
            test_dates = test_df.index[-min_l:]

            from modules.metrics import evaluate_all
            metrics_dict = evaluate_all(obs_a, pred_a)

            # Metric cards row
            card_cols = st.columns(6)
            for i, (key, label) in enumerate([
                ("NSE","NSE"),("KGE","KGE"),("RMSE","RMSE"),
                ("R2","R²"),("PBIAS","PBIAS%"),("MAE","MAE"),
            ]):
                val = metrics_dict[key]
                card_cols[i].markdown(metric_card(label, val, key), unsafe_allow_html=True)

            st.plotly_chart(
                hydrograph_fig(test_dates, obs_a, pred_a, model_select, metrics_dict),
                use_container_width=True,
            )
            st.plotly_chart(
                scatter_fig(obs_a, pred_a, model_select, metrics_dict.get("R2"),
                            metrics_dict.get("RMSE")),
                use_container_width=True,
            )

            # Download button
            pred_df = pd.DataFrame({
                "date":     test_dates,
                "observed": obs_a,
                "predicted": pred_a,
                "residual": pred_a - obs_a,
            })
            buf = io.BytesIO()
            pred_df.to_csv(buf, index=False)
            st.download_button(
                f"⬇ Download {model_select} predictions (CSV)",
                data=buf.getvalue(),
                file_name=f"predictions_{model_select}.csv",
                mime="text/csv",
            )

    # ── Ensemble predictions ─────────────────────────────────────────────────
    ens_df = load_ensemble_csv()
    if ens_df is not None:
        st.divider()
        st.markdown('<div class="section-header">🔀 Ensemble Predictions</div>',
                    unsafe_allow_html=True)

        fig_ens = go.Figure()
        for col in ens_df.columns:
            if col == "observed":
                fig_ens.add_trace(go.Scatter(
                    y=ens_df["observed"], name="Observed",
                    line=dict(color="#000", width=1.8),
                ))
            elif col.startswith("ensemble_"):
                fig_ens.add_trace(go.Scatter(
                    y=ens_df[col], name=col.replace("_", " ").title(),
                    line=dict(dash="dash"),
                ))
        fig_ens.update_layout(
            title="Ensemble vs Individual Models (Test Period)",
            xaxis_title="Step", yaxis_title="Discharge (cumecs)",
            height=380, plot_bgcolor="white",
        )
        st.plotly_chart(fig_ens, use_container_width=True)


# ── Tab 3: CMIP6 Projections ─────────────────────────────────────────────────

def render_cmip6(scenarios: List[str]):
    st.markdown('<div class="section-header">🌍 CMIP6 Climate Projections</div>',
                unsafe_allow_html=True)

    summary_df = load_forecast_summary()
    if summary_df is None:
        st.info("ℹ️ No projection results found. Run the pipeline (with CMIP6 enabled) first.")
        _show_cmip6_structure()
        return

    # ── Scenario filter ──────────────────────────────────────────────────────
    if "scenario" in summary_df.columns:
        ssp_col = "scenario"
    elif "SSP" in summary_df.columns:
        ssp_col = "SSP"
    else:
        ssp_col = None

    if ssp_col:
        available_ssps = summary_df[ssp_col].unique().tolist()
        ssp_filter = st.multiselect("Filter by scenario", available_ssps,
                                     default=available_ssps)
        filtered = summary_df[summary_df[ssp_col].isin(ssp_filter)]
    else:
        filtered = summary_df

    st.dataframe(filtered, use_container_width=True)

    if filtered.empty:
        return

    # ── Change from baseline bar chart ──────────────────────────────────────
    change_col = next((c for c in ["change_pct","Change_pct","change_from_baseline_pct"]
                       if c in filtered.columns), None)
    period_col = next((c for c in ["period","Period"] if c in filtered.columns), None)
    gcm_col    = next((c for c in ["gcm","GCM"] if c in filtered.columns), None)
    mean_q_col = next((c for c in ["mean_Q","Mean_Q"] if c in filtered.columns), None)

    if change_col and period_col:
        st.markdown("**Projected Discharge Change from Baseline (%)**")
        fig_bar = px.bar(
            filtered,
            x=period_col, y=change_col,
            color=ssp_col or gcm_col,
            barmode="group",
            color_discrete_map={"ssp245": "#1f77b4", "SSP245": "#1f77b4",
                                 "ssp585": "#d62728", "SSP585": "#d62728"},
            labels={change_col: "Change (%)", period_col: "Period"},
            height=400,
        )
        fig_bar.add_hline(y=0, line_dash="dash", line_color="black")
        fig_bar.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_bar, use_container_width=True)

    if mean_q_col and gcm_col and period_col:
        st.markdown("**Mean Projected Discharge by GCM × Period**")
        fig_heatmap = px.density_heatmap(
            filtered, x=period_col, y=gcm_col, z=mean_q_col,
            color_continuous_scale="Blues",
            height=350,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Download
    buf = io.BytesIO()
    filtered.to_csv(buf, index=False)
    st.download_button("⬇ Download projection summary (CSV)",
                       data=buf.getvalue(),
                       file_name="cmip6_projection_summary.csv",
                       mime="text/csv")


def _show_cmip6_structure():
    st.markdown("**Expected CMIP6 data structure:**")
    st.code("""
cmip6_downloads/
├── historical/
│   ├── MRI-ESM2-0_historical_daily.csv
│   └── ...
├── ssp245/
│   ├── MRI-ESM2-0_ssp245_daily.csv
│   └── ...
└── ssp585/
    ├── MRI-ESM2-0_ssp585_daily.csv
    └── ...
# Each CSV: columns = date, precip, tmax, tmin
""")


# ── Tab 4: Forecast Viewer ────────────────────────────────────────────────────

def render_forecast_viewer():
    st.markdown('<div class="section-header">📈 Future Discharge Forecast</div>',
                unsafe_allow_html=True)

    # Try to load per-period timeseries CSVs
    ts_root = ROOT / "outputs" / "forecast_timeseries"
    if not ts_root.exists():
        st.info("ℹ️ No forecast timeseries found. Run the forecast engine first.")
        return

    # Collect all available scenario / period / GCM combos
    ssp_dirs = [d for d in ts_root.iterdir() if d.is_dir()]
    if not ssp_dirs:
        st.info("No forecast data found.")
        return

    c1, c2, c3 = st.columns(3)
    ssp_options = [d.name for d in ssp_dirs]
    sel_ssp = c1.selectbox("Scenario", ssp_options)

    period_dirs = [d for d in (ts_root / sel_ssp).iterdir() if d.is_dir()]
    period_options = [d.name for d in period_dirs]
    if not period_options:
        st.warning(f"No period data for {sel_ssp}")
        return
    sel_period = c2.selectbox("Period", period_options)

    # Load available CSVs in that period
    period_path = ts_root / sel_ssp / sel_period
    csv_files = list(period_path.glob("*.csv"))
    gcm_options = [f.stem for f in csv_files]
    sel_gcm = c3.selectbox("GCM / Ensemble", gcm_options)

    sel_csv = period_path / f"{sel_gcm}.csv"
    if not sel_csv.exists():
        st.warning(f"File not found: {sel_csv}")
        return

    fc_df = pd.read_csv(sel_csv, parse_dates=["date"])

    # Stats cards
    discharge_cols = [c for c in fc_df.columns if "discharge" in c.lower()]
    if discharge_cols:
        main_col = discharge_cols[0]
        q = fc_df[main_col].values
        q = q[~np.isnan(q)]
        cc = st.columns(5)
        cc[0].metric("Mean Q", f"{np.mean(q):.2f} cumecs")
        cc[1].metric("Max Q",  f"{np.max(q):.2f} cumecs")
        cc[2].metric("Min Q",  f"{np.min(q):.2f} cumecs")
        cc[3].metric("Q10",    f"{np.percentile(q,10):.2f}")
        cc[4].metric("Q90",    f"{np.percentile(q,90):.2f}")

    # Timeseries plot
    fig = go.Figure()
    colors_list = px.colors.qualitative.Plotly
    for ci, col in enumerate(discharge_cols):
        dash = "solid" if "ensemble" in col else "dash"
        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df[col],
            name=col.replace("discharge_", "").replace("_", " ").title(),
            line=dict(color=colors_list[ci % len(colors_list)], dash=dash),
            opacity=0.85,
        ))

    # Add baseline reference
    summary = load_forecast_summary()
    if summary is not None and "baseline_mean_Q" in summary.columns:
        baseline_val = summary["baseline_mean_Q"].iloc[0]
        fig.add_hline(y=baseline_val, line_dash="dot", line_color="gray",
                      annotation_text=f"Obs Baseline ({baseline_val:.1f})")

    fig.update_layout(
        title=f"Discharge Forecast — {sel_ssp.upper()} | {sel_period} | {sel_gcm}",
        xaxis_title="Date", yaxis_title="Discharge (cumecs)",
        height=450, hovermode="x unified", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly aggregation
    fc_df2 = fc_df.set_index("date")
    if len(discharge_cols) > 0:
        monthly_fc = fc_df2[discharge_cols[0]].resample("ME").mean()
        fig_mon = px.bar(
            x=monthly_fc.index.strftime("%Y-%m"), y=monthly_fc.values,
            labels={"x": "Month", "y": "Mean Discharge (cumecs)"},
            title="Monthly Mean Forecast",
            color=monthly_fc.index.month.isin([6, 7, 8, 9]),
            color_discrete_map={True: "#1f77b4", False: "#aec7e8"},
            height=320,
        )
        st.plotly_chart(fig_mon, use_container_width=True)

    # Download
    buf = io.BytesIO()
    fc_df.to_csv(buf, index=False)
    st.download_button(
        f"⬇ Download {sel_gcm} forecast (CSV)",
        data=buf.getvalue(),
        file_name=f"forecast_{sel_ssp}_{sel_period}_{sel_gcm}.csv",
        mime="text/csv",
    )


# ── Tab 5: Seasonal & Extremes ───────────────────────────────────────────────

def render_seasonal_extremes(obs_df: pd.DataFrame):
    st.markdown('<div class="section-header">🌦 Seasonal Analysis & Extremes</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # ── Flow Duration Curve ──
        st.markdown("**Flow Duration Curve (Observed)**")
        Q_sorted  = np.sort(obs_df["discharge"].dropna().values)[::-1]
        n         = len(Q_sorted)
        exceedance = np.arange(1, n + 1) / n * 100

        fig_fdc = go.Figure()
        fig_fdc.add_trace(go.Scatter(
            x=exceedance, y=Q_sorted,
            fill="tozeroy", fillcolor="rgba(31,119,180,0.15)",
            line=dict(color="#1f77b4", width=2), name="FDC",
        ))
        for pct, label in [(10,"Q10"),(50,"Q50"),(90,"Q90")]:
            idx = int(pct / 100 * n)
            fig_fdc.add_annotation(
                x=pct, y=Q_sorted[idx], text=f"{label}={Q_sorted[idx]:.1f}",
                showarrow=True, arrowhead=2, font=dict(size=11),
            )
        fig_fdc.update_layout(
            xaxis_title="Exceedance Probability (%)",
            yaxis_title="Discharge (cumecs, log)",
            yaxis_type="log", height=360, plot_bgcolor="white",
        )
        st.plotly_chart(fig_fdc, use_container_width=True)

    with col_b:
        # ── Annual Max Boxplot ──
        st.markdown("**Annual Maximum Discharge**")
        obs_df2 = obs_df.copy()
        obs_df2["year"] = obs_df2.index.year
        ann_max = obs_df2.groupby("year")["discharge"].max().reset_index()
        ann_max.columns = ["Year", "Max_Q"]
        fig_box = px.bar(ann_max, x="Year", y="Max_Q",
                          color="Max_Q", color_continuous_scale="Blues",
                          height=360, labels={"Max_Q": "Max Discharge (cumecs)"})
        fig_box.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        # ── Monsoon vs Dry season ──
        st.markdown("**Monsoon vs Dry Season**")
        obs_df2["season"] = obs_df2.index.month.map(
            lambda m: "Monsoon (Jun–Sep)" if m in [6,7,8,9] else "Dry Season"
        )
        fig_vio = px.violin(
            obs_df2.reset_index(), x="season", y="discharge",
            box=True, points="outliers",
            color="season",
            color_discrete_map={"Monsoon (Jun–Sep)":"#1f77b4","Dry Season":"#ff7f0e"},
            height=360, labels={"discharge":"Discharge (cumecs)"},
        )
        fig_vio.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig_vio, use_container_width=True)

    with col_d:
        # ── Lag correlation ──
        st.markdown("**Rainfall–Discharge Lag Correlation**")
        lag_results = []
        for lag in range(1, 15):
            corr = obs_df["precip"].shift(lag).corr(obs_df["discharge"])
            lag_results.append({"Lag (days)": lag, "Correlation": round(corr, 4)})
        lag_df = pd.DataFrame(lag_results)
        fig_lag = px.line(lag_df, x="Lag (days)", y="Correlation",
                           markers=True, height=360)
        fig_lag.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_lag.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_lag, use_container_width=True)

    # ── Flood frequency (Gumbel) ──────────────────────────────────────────
    st.divider()
    st.markdown("**Flood Frequency Analysis (Gumbel)**")
    ann_max_vals = ann_max["Max_Q"].values
    if len(ann_max_vals) > 2:
        y_mean = np.mean(ann_max_vals)
        y_std  = np.std(ann_max_vals, ddof=1)
        alpha  = y_std * np.sqrt(6) / np.pi
        mu     = y_mean - 0.5772 * alpha
        return_periods = [2, 5, 10, 25, 50, 100]
        gumbel_q = [mu + alpha * (-np.log(-np.log(1 - 1/t))) for t in return_periods]

        fig_gum = go.Figure()
        fig_gum.add_trace(go.Scatter(
            x=return_periods, y=gumbel_q,
            mode="lines+markers", name="Gumbel Fit",
            line=dict(color="#D32F2F", width=2), marker=dict(size=8),
        ))
        # Observed plotting positions
        n_yr = len(ann_max_vals)
        sorted_am = np.sort(ann_max_vals)[::-1]
        obs_rp = (n_yr + 1) / np.arange(1, n_yr + 1)
        fig_gum.add_trace(go.Scatter(
            x=obs_rp, y=sorted_am,
            mode="markers", name="Observed Annual Max",
            marker=dict(color="#1f77b4", size=7),
        ))
        fig_gum.update_layout(
            xaxis_title="Return Period (years)", xaxis_type="log",
            yaxis_title="Discharge (cumecs)",
            height=380, plot_bgcolor="white",
        )
        st.plotly_chart(fig_gum, use_container_width=True)

        # Table
        rp_df = pd.DataFrame({
            "Return Period (years)": return_periods,
            "Gumbel Q (cumecs)": [round(q, 2) for q in gumbel_q],
        })
        st.dataframe(rp_df, use_container_width=True)


# ── Tab 6: Settings / Logs ────────────────────────────────────────────────────

def render_settings(obs_df: pd.DataFrame, feat_list: list):
    st.markdown('<div class="section-header">⚙️ Configuration & Logs</div>',
                unsafe_allow_html=True)

    from modules.config import basin_cfg, data_cfg, model_cfg, climate_cfg

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Basin Configuration**")
        basin_dict = {k: v for k, v in vars(basin_cfg).items() if not k.startswith("_")}
        st.json(basin_dict)

    with c2:
        st.markdown("**Climate Configuration**")
        clim_dict = {k: v for k, v in vars(climate_cfg).items() if not k.startswith("_")}
        st.json(clim_dict)

    st.divider()
    st.markdown("**Feature List (active)**")
    st.write(feat_list)
    st.caption(f"Total features: {len(feat_list)}")

    st.divider()
    st.markdown("**Output Files**")
    out_dir = ROOT / "outputs"
    if out_dir.exists():
        files = list(out_dir.rglob("*"))
        file_rows = []
        for f in sorted(files):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                file_rows.append({
                    "File": str(f.relative_to(out_dir)),
                    "Size (KB)": round(size_kb, 1),
                })
        if file_rows:
            st.dataframe(pd.DataFrame(file_rows), use_container_width=True)
        else:
            st.info("No output files yet.")
    else:
        st.info("Outputs directory not found.")


# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE EXECUTION (triggered via sidebar buttons)
# ════════════════════════════════════════════════════════════════════════════

def _run_pipeline_subprocess(controls: dict, obs_df: pd.DataFrame):
    """Run the full pipeline inline (within Streamlit process)."""
    import argparse as ap_mod
    from main import (
        phase2_engineer_features, phase3_train_or_load,
        phase4_evaluate, phase5_ensemble, phase6_cmip6, phase7_forecast,
    )
    from modules.config import data_cfg, OUTPUT_DIR

    output_dir = OUTPUT_DIR

    st.session_state["pipeline_running"] = True
    progress = st.progress(0, text="Starting pipeline...")

    class _FakeArgs:
        output_dir = OUTPUT_DIR
        skip_training   = False
        synthetic       = controls["use_synthetic"]
        feature_set     = "M1_full"
        no_cmip6        = controls["skip_cmip6"]
        output_dir      = str(output_dir)
        models          = controls["selected_models"]

    args = _FakeArgs()

    try:
        progress.progress(10, "Feature engineering...")
        df_feat, feat_list = phase2_engineer_features(obs_df, args.feature_set)

        progress.progress(25, "Training / loading models...")
        trained_models, train_df, val_df, test_df, X_test, y_test = phase3_train_or_load(
            df_feat, feat_list, args, output_dir
        )

        progress.progress(55, "Evaluating models...")
        predictions, all_metrics = phase4_evaluate(
            trained_models, X_test, y_test, test_df, output_dir
        )

        progress.progress(70, "Building ensemble...")
        phase5_ensemble(trained_models, predictions, y_test, output_dir)

        if not args.no_cmip6:
            progress.progress(80, "CMIP6 projections...")
            phase6_cmip6(trained_models, obs_df, feat_list, output_dir)

        progress.progress(90, "Running forecast...")
        phase7_forecast(trained_models, obs_df, feat_list, output_dir)

        progress.progress(100, "Complete!")
        st.success("✅ Pipeline completed successfully!")

        # Clear caches so results reload
        load_metrics_json.clear()
        load_forecast_results.clear()
        load_forecast_summary.clear()
        load_ensemble_csv.clear()

    except Exception as e:
        st.error(f"❌ Pipeline failed: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.session_state["pipeline_running"] = False


def _run_forecast_only(controls: dict, obs_df: pd.DataFrame, feat_list: list, n_features: int):
    """Load saved models and run forecast only."""
    from modules.forecast import run_forecast
    from modules.config import climate_cfg, OUTPUT_DIR

    saved_models = load_saved_models(n_features)
    if not saved_models:
        st.error("No saved models found — run the full pipeline first.")
        return

    with st.spinner("Running forecast..."):
        try:
            run_forecast(
                trained_models=saved_models,
                obs_df=obs_df,
                feature_list=feat_list,
                cmip6_dir=str(ROOT / climate_cfg.cmip6_download_dir),
                scenarios=controls["scenarios"],
                future_periods=climate_cfg.future_periods,
                output_dir=str(OUTPUT_DIR),
            )
            st.success("✅ Forecast complete!")
            load_forecast_results.clear()
            load_forecast_summary.clear()
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════════════════

def main():
    controls = render_sidebar()

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🌊 FlowCast v2 — Kabini River Basin")
    st.caption(
        "CMIP6-driven discharge prediction using RF-PSO · SVR-PSO · XGBoost · "
        "Physics-Informed LSTM · Adapted-GRU"
    )

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        obs_df, df_feat, feat_list = load_feature_engineered(controls["use_synthetic"])
        n_features = len(feat_list)
        saved_models = load_saved_models(n_features)
        metrics_json = load_metrics_json()

    # ── Pipeline execution ───────────────────────────────────────────────────
    if controls["run_pipeline"]:
        with st.spinner("Running full pipeline — this may take several minutes..."):
            _run_pipeline_subprocess(controls, obs_df)
        st.rerun()

    if controls["run_forecast"]:
        _run_forecast_only(controls, obs_df, feat_list, n_features)
        st.rerun()

    # ── Status banner ────────────────────────────────────────────────────────
    model_status = f"✅ {len(saved_models)} saved models loaded" if saved_models \
                   else "⚠️ No saved models — run the pipeline to train"
    forecast_ok  = (ROOT / "outputs" / "forecast_timeseries").exists()
    forecast_status = "✅ Forecast data available" if forecast_ok \
                      else "⚠️ No forecast data yet"

    banner_cols = st.columns(3)
    banner_cols[0].info(f"**Data:** {len(obs_df):,} days loaded")
    banner_cols[1].info(f"**Models:** {model_status}")
    banner_cols[2].info(f"**Forecast:** {forecast_status}")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "🏆 Model Performance",
        "🌍 CMIP6 Projections",
        "📈 Forecast Viewer",
        "🌦 Seasonal & Extremes",
        "⚙️ Settings",
    ])

    with tabs[0]:
        render_overview(obs_df)

    with tabs[1]:
        render_model_performance(
            obs_df, df_feat, feat_list, saved_models,
            metrics_json, controls["selected_models"],
        )

    with tabs[2]:
        render_cmip6(controls["scenarios"])

    with tabs[3]:
        render_forecast_viewer()

    with tabs[4]:
        render_seasonal_extremes(obs_df)

    with tabs[5]:
        render_settings(obs_df, feat_list)


if __name__ == "__main__":
    main()