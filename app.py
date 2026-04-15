"""
FlowCast v2 — Streamlit Dashboard
Data-driven dashboard that loads predictions, plots, and results from the pipeline.

Tabs: Overview → Model Results → Comparison → Hydrology → Climate Projections
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import glob

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "saved_models"

st.set_page_config(
    page_title="FlowCast v2 — Kabini River Basin",
    page_icon="🌊",
    layout="wide",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d5a8e);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.85; }
    .metric-card h1 { margin: 5px 0 0 0; font-size: 28px; }
    .hydro-card {
        background: linear-gradient(135deg, #1a4731, #2d7a54);
        border-radius: 10px; padding: 15px; color: white;
        text-align: center; margin-bottom: 8px;
    }
    .hydro-card h4 { margin: 0; font-size: 12px; opacity: 0.85; }
    .hydro-card h2 { margin: 3px 0 0 0; font-size: 22px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════

@st.cache_data
def load_results():
    path = OUTPUT_DIR / "pipeline_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_predictions(model_name):
    """Load prediction CSV for a specific model."""
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    path = OUTPUT_DIR / f"predictions_{safe_name}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return None


@st.cache_data
def load_raw_data():
    """Load the raw data summary from pipeline."""
    path = OUTPUT_DIR / "raw_data_summary.csv"
    if path.exists():
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        return df
    return None

@st.cache_data
def load_master_data():
    """Load the master dataset for EDA."""
    # Assuming standard project structure based on eda.py
    path = ROOT.parent / "data/master_dataset.csv" 
    if not path.exists():
        path = ROOT / "data/master_dataset.csv" # Fallback
        
    if path.exists():
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data
def get_available_models():
    """Discover which models have prediction CSVs."""
    models = []
    for f in sorted(OUTPUT_DIR.glob("predictions_*.csv")):
        name = f.stem.replace("predictions_", "")
        models.append(name)
    return models


def metric_card(label, value, color="135deg, #1e3a5f, #2d5a8e"):
    st.markdown(f"""
    <div style="background:linear-gradient({color}); border-radius:12px; padding:20px;
                color:white; text-align:center; margin-bottom:10px;">
        <h3 style="margin:0;font-size:14px;opacity:0.85;">{label}</h3>
        <h1 style="margin:5px 0 0 0;font-size:28px;">{value}</h1>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════════════════

def plotly_hydrograph(df, model_name, metrics=None):
    """Interactive hydrograph for one model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["observed"], mode="lines",
        name="Observed", line=dict(color="black", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["predicted"], mode="lines",
        name=f"Predicted ({model_name})", line=dict(width=1.5),
    ))
    title = f"Hydrograph — {model_name}"
    if metrics:
        nse = metrics.get("NSE", "?")
        kge = metrics.get("KGE", "?")
        title += f"  |  NSE={nse}  KGE={kge}"
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Discharge (cumecs)",
        height=400, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plotly_scatter(df, model_name, metrics=None):
    """Interactive scatter for one model."""
    fig = go.Figure()
    min_val = min(df["observed"].min(), df["predicted"].min())
    max_val = max(df["observed"].max(), df["predicted"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode="lines",
        name="Ideal (1:1)", line=dict(dash="dash", color="gray"),
    ))
    fig.add_trace(go.Scatter(
        x=df["observed"], y=df["predicted"], mode="markers",
        name=model_name, marker=dict(size=4, opacity=0.5),
    ))
    title = f"Scatter — {model_name}"
    if metrics:
        r2 = metrics.get("R2", "?")
        rmse = metrics.get("RMSE", "?")
        title += f"  |  R²={r2}  RMSE={rmse}"
    fig.update_layout(
        title=title,
        xaxis_title="Observed Discharge (cumecs)",
        yaxis_title="Predicted Discharge (cumecs)",
        height=450, template="plotly_white",
    )
    return fig


def plotly_violin(results, model_names):
    """Violin plot of errors across models."""
    all_data = []
    for name in model_names:
        df = load_predictions(name)
        if df is not None:
            error = df["predicted"].values - df["observed"].values
            for e in error:
                all_data.append({"Model": name, "Error": e})

    if not all_data:
        return None

    plot_df = pd.DataFrame(all_data)
    fig = px.violin(plot_df, x="Model", y="Error", box=True, points=False,
                    title="Distribution of Prediction Errors (Residuals)")
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(height=450, template="plotly_white",
                      yaxis_title="Prediction Error (Predicted − Observed)")
    return fig


def plotly_eckhardt(raw_df):
    """Interactive Eckhardt baseflow separation with inverted rainfall."""
    # Use one year slice
    if raw_df.index[-1] >= pd.Timestamp("2019-01-01"):
        s = raw_df.loc["2019-01-01":"2019-12-31"]
    else:
        end = raw_df.index[-1]
        start = end - pd.DateOffset(years=1)
        s = raw_df.loc[start:end]

    if len(s) == 0 or "baseflow" not in s.columns:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=s.index, y=s["discharge"], mode="lines",
        name="Total Discharge", line=dict(color="#1f77b4", width=1.5),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=s.index, y=s["baseflow"], mode="lines", fill="tozeroy",
        name="Baseflow (Eckhardt)", line=dict(color="#2ca02c"),
        fillcolor="rgba(44,160,44,0.3)",
    ), secondary_y=False)

    # Inverted rainfall bars
    fig.add_trace(go.Bar(
        x=s.index, y=s["precip"], name="Precipitation",
        marker_color="rgba(127,127,127,0.5)", width=86400000,
    ), secondary_y=True)

    max_precip = s["precip"].max() if s["precip"].max() > 0 else 1
    fig.update_yaxes(
        title_text="Flow (cumecs)", secondary_y=False, rangemode="tozero",
    )
    fig.update_yaxes(
        title_text="Rainfall (mm)", secondary_y=True,
        autorange="reversed",  # INVERT rainfall axis
        range=[max_precip * 3, 0],
    )
    fig.update_layout(
        title="Baseflow Separation (Eckhardt Filter) with Rainfall Hyetograph",
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plotly_fdc(raw_df):
    """Flow Duration Curve."""
    q = raw_df["discharge"].dropna().values
    sorted_q = np.sort(q)[::-1]
    n = len(sorted_q)
    exceedance = np.arange(1, n + 1) / (n + 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exceedance, y=sorted_q, mode="lines",
        fill="tozeroy", fillcolor="rgba(31,119,180,0.15)",
        line=dict(color="#1f77b4", width=2), name="FDC",
    ))

    # Annotate percentiles
    for pct, label in [(5, "Q5 (high)"), (50, "Q50 (median)"), (95, "Q95 (low)")]:
        idx = min(int(pct / 100 * n), n - 1)
        fig.add_annotation(x=pct, y=sorted_q[idx],
                           text=f"{label}<br>{sorted_q[idx]:.1f}",
                           showarrow=True, arrowhead=2, font=dict(size=10))

    fig.update_layout(
        title="Flow Duration Curve",
        xaxis_title="Exceedance Probability (%)",
        yaxis_title="Discharge (cumecs)",
        yaxis_type="log",
        height=400, template="plotly_white",
    )
    return fig


def plotly_flood_frequency(flood_freq, annual_max=None):
    """Gumbel flood frequency curve."""
    rp = [int(k.replace("T", "")) for k in flood_freq.keys()]
    qvals = list(flood_freq.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rp, y=qvals, mode="lines+markers",
        name="Gumbel Fit", marker=dict(size=10),
        line=dict(color="#D32F2F", width=2),
    ))

    if annual_max is not None:
        n = len(annual_max)
        sorted_am = np.sort(annual_max)[::-1]
        plotting_rp = [(n + 1) / i for i in range(1, n + 1)]
        fig.add_trace(go.Scatter(
            x=plotting_rp, y=sorted_am, mode="markers",
            name="Observed Annual Max",
            marker=dict(size=8, color="#1f77b4"),
        ))

    fig.update_layout(
        title="Gumbel Flood Frequency Analysis",
        xaxis_title="Return Period (years)",
        yaxis_title="Discharge (cumecs)",
        xaxis_type="log", height=400, template="plotly_white",
    )
    return fig


# ═══════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════

def main():
    st.title("🌊 FlowCast v2 — Kabini River Basin")
    st.caption("ML-Coupled Hydrological Prediction & CMIP6 Climate Projections")

    results = load_results()
    raw_df = load_raw_data()

    if results is None:
        st.warning("No results found. Run the pipeline first: `python main.py`")
        st.code("cd flowcast_v2 && python main.py", language="bash")
        st.markdown("---")
        st.subheader("Pipeline Architecture")
        st.markdown("""
        **FlowCast v2** predicts daily Kabini River discharge using 6 ML models,
        PSO hyperparameter optimization, physics-informed loss, and CMIP6 climate projections.

        **Models:** RF-PSO, SVR-PSO, XGBoost-PSO, LSTM, PI-LSTM, A-GRU

        **Ensembles:** Weighted (optimized), Stacking (Ridge)

        **CMIP6:** Real GCM data → QDM Bias Correction → Recursive Prediction
        """)
        return

    models = results.get("models", {})
    ensemble = results.get("ensemble", {})

    # ── TAB LAYOUT ──
    tab1, tab_eda, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 EDA", "🤖 Model Results", "⚖️ Comparison",
        "💧 Hydrology", "🌡️ Climate Projections"
    ])

    # ═══════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ═══════════════════════════════════════════════════════════
    with tab1:
        st.header("Pipeline Overview")

        all_results = {}
        for name, res in models.items():
            all_results[name] = res["test"]
        for name, res in ensemble.items():
            if "test" in res:
                all_results[f"Ens-{name}"] = res["test"]

        if all_results:
            best_model = max(all_results, key=lambda x: all_results[x].get("NSE", -999))
            best_metrics = all_results[best_model]

            cols = st.columns(4)
            with cols[0]:
                metric_card("Best Model", best_model)
            with cols[1]:
                metric_card("NSE", f"{best_metrics['NSE']:.4f}", "135deg, #1a6b3c, #2d9a5e")
            with cols[2]:
                metric_card("KGE", f"{best_metrics['KGE']:.4f}", "135deg, #5a1a6b, #8a2d9a")
            with cols[3]:
                metric_card("RMSE", f"{best_metrics['RMSE']:.2f}", "135deg, #6b1a1a, #9a2d2d")

            # Ranking table
            st.subheader("Model Ranking (Test Set)")
            ranking_data = [{"Model": name, **metrics} for name, metrics in all_results.items()]
            ranking_df = pd.DataFrame(ranking_data).sort_values("NSE", ascending=False)
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)

        # Saved models
        saved = results.get("saved_models", {})
        if saved:
            st.subheader("Saved Models")
            model_info = []
            for name, path in saved.items():
                p = Path(path)
                size = f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "not found"
                model_info.append({"Model": name, "Path": str(p.name), "Size": size})
            st.dataframe(pd.DataFrame(model_info), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 2: EXPLORATORY DATA ANALYSIS
    # ═══════════════════════════════════════════════════════════
    with tab_eda:
        st.header("Exploratory Data Analysis (EDA)")
        
        df_master = load_master_data()
        
        if df_master is None:
            st.warning("Master dataset not found. Ensure `data/master_dataset.csv` exists.")
        else:
            # 1. Resolve Rain Column (matching eda.py logic)
            AGREED_RAIN_COL = 'rainfall_max_mm'
            if AGREED_RAIN_COL in df_master.columns:
                rain_col = AGREED_RAIN_COL
            else:
                _candidates = [c for c in df_master.columns if any(kw in c for kw in ('precip', 'rain', 'chirps', 'prcp', 'rf')) and c != 'date']
                rain_col = _candidates[0] if _candidates else None

            if not rain_col or 'q_upstream_mk' not in df_master.columns:
                st.error("Required columns (rainfall and 'q_upstream_mk') not found in dataset.")
            else:
                st.markdown(f"**Using Precipitation Column:** `{rain_col}` | **Target Column:** `q_upstream_mk`")
                
                # 2. Basic Info & Extreme Events
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top 5 Rainfall Events")
                    top_rain = df_master.nlargest(5, rain_col)[['date', rain_col]]
                    top_rain['date'] = top_rain['date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(top_rain, hide_index=True, use_container_width=True)
                with col2:
                    st.subheader("Top 5 Flood Events (Muthankera)")
                    top_flow = df_master.nlargest(5, 'q_upstream_mk')[['date', 'q_upstream_mk']]
                    top_flow['date'] = top_flow['date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(top_flow, hide_index=True, use_container_width=True)

                st.markdown("---")

                # 3. Time Series
                st.subheader("Historical Time Series")
                fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ts.add_trace(go.Scatter(x=df_master['date'], y=df_master['q_upstream_mk'], name="Discharge (cumecs)", line=dict(color="#1f77b4")), secondary_y=False)
                fig_ts.add_trace(go.Bar(x=df_master['date'], y=df_master[rain_col], name="Rainfall (mm)", marker_color="rgba(127,127,127,0.5)"), secondary_y=True)
                
                fig_ts.update_yaxes(title_text="Discharge", secondary_y=False)
                max_precip = df_master[rain_col].max()
                fig_ts.update_yaxes(title_text="Rainfall", secondary_y=True, autorange="reversed", range=[max_precip * 3, 0])
                fig_ts.update_layout(height=450, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_ts, use_container_width=True)

                # 4. Distributions
                st.markdown("---")
                st.subheader("Data Distributions")
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    fig_hist_rain = px.histogram(df_master, x=rain_col, nbins=50, title="Rainfall Distribution")
                    st.plotly_chart(fig_hist_rain, use_container_width=True)
                with dist_col2:
                    fig_hist_flow = px.histogram(df_master, x='q_upstream_mk', nbins=50, title="Discharge Distribution", color_discrete_sequence=['#2ca02c'])
                    st.plotly_chart(fig_hist_flow, use_container_width=True)

                st.markdown("---")

                # 5. Correlation & Seasonality
                st.subheader("Correlation & Seasonality")
                corr_col1, corr_col2 = st.columns(2)
                
                with corr_col1:
                    numeric_df = df_master.select_dtypes(include='number')
                    corr = numeric_df.corr()
                    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap", color_continuous_scale="RdBu_r")
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                with corr_col2:
                    df_master['month'] = df_master['date'].dt.month
                    monthly_avg = df_master.groupby('month').mean(numeric_only=True).reset_index()
                    
                    fig_seas = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_seas.add_trace(go.Scatter(x=monthly_avg['month'], y=monthly_avg['q_upstream_mk'], mode="lines+markers", name="Avg Flow", line=dict(color="#1f77b4")), secondary_y=False)
                    fig_seas.add_trace(go.Bar(x=monthly_avg['month'], y=monthly_avg[rain_col], name="Avg Rain", marker_color="rgba(127,127,127,0.5)"), secondary_y=True)
                    fig_seas.update_layout(title="Monthly Averages", height=400, xaxis=dict(tickmode='linear', tick0=1, dtick=1))
                    fig_seas.update_yaxes(title_text="Flow (cumecs)", secondary_y=False)
                    fig_seas.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)
                    st.plotly_chart(fig_seas, use_container_width=True)

                # 6. Lag Correlation
                st.markdown("---")
                st.subheader("Lag Correlation Analysis")
                lag_results = []
                for lag in range(1, 15):
                    shifted = df_master[rain_col].shift(lag)
                    corr_val = shifted.corr(df_master['q_upstream_mk'])
                    lag_results.append({"Lag (days)": lag, "Correlation": corr_val})
                
                lag_df = pd.DataFrame(lag_results)
                fig_lag = px.line(lag_df, x="Lag (days)", y="Correlation", markers=True, title="Rainfall → Discharge Lag Correlation")
                fig_lag.update_layout(height=350)
                st.plotly_chart(fig_lag, use_container_width=True)
    # ═══════════════════════════════════════════════════════════
    # TAB 3: MODEL RESULTS (PER-MODEL HYDROGRAPHS & SCATTER)
    # ═══════════════════════════════════════════════════════════
    with tab2:
        st.header("Individual Model Performance")

        model_list = list(models.keys())
        selected_model = st.selectbox("Select Model", model_list)

        if selected_model:
            res = models[selected_model]
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Validation Metrics")
                st.dataframe(pd.DataFrame([res["val"]]), use_container_width=True, hide_index=True)
            with col2:
                st.subheader("Test Metrics")
                st.dataframe(pd.DataFrame([res["test"]]), use_container_width=True, hide_index=True)

            # Load predictions and render interactive plots
            pred_df = load_predictions(selected_model)
            if pred_df is not None:
                st.subheader(f"Hydrograph — {selected_model}")
                fig_hydro = plotly_hydrograph(pred_df, selected_model, res.get("test"))
                st.plotly_chart(fig_hydro, use_container_width=True)

                st.subheader(f"Scatter Plot — {selected_model}")
                fig_scatter = plotly_scatter(pred_df, selected_model, res.get("test"))
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                # Fall back to saved PNG
                png_hydro = OUTPUT_DIR / f"hydrograph_{selected_model}.png"
                png_scatter = OUTPUT_DIR / f"scatter_{selected_model}.png"
                if png_hydro.exists():
                    st.image(str(png_hydro), caption=f"Hydrograph — {selected_model}")
                if png_scatter.exists():
                    st.image(str(png_scatter), caption=f"Scatter — {selected_model}")

            # Flow regime analysis
            if "flow_regimes" in res:
                st.subheader("Flow Regime Performance")
                regime_data = [{"Regime": k, **v} for k, v in res["flow_regimes"].items()]
                st.dataframe(pd.DataFrame(regime_data), use_container_width=True, hide_index=True)

                fig = go.Figure()
                for regime, metrics in res["flow_regimes"].items():
                    vals = [metrics.get(m, 0) for m in ["NSE", "KGE", "R2", "R"]]
                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=["NSE", "KGE", "R²", "R", "NSE"],
                        fill="toself", name=regime,
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(range=[-0.5, 1.0])),
                    height=400, title="Flow Regime Radar",
                )
                st.plotly_chart(fig, use_container_width=True)

            if "best_params" in res:
                st.subheader("Optimized Parameters (PSO)")
                st.json(res["best_params"])

    # ═══════════════════════════════════════════════════════════
    # TAB 3: COMPARISON (SIDE-BY-SIDE + VIOLIN)
    # ═══════════════════════════════════════════════════════════
    with tab3:
        st.header("Model Comparison")

        all_results_flat = {}
        for name, res in models.items():
            all_results_flat[name] = res["test"]
        for name, res in ensemble.items():
            if "test" in res:
                all_results_flat[f"Ens-{name}"] = res["test"]

        if all_results_flat:
            # Grouped bar chart of metrics
            metrics_to_plot = ["NSE", "KGE", "R2", "R"]
            available_metrics = [m for m in metrics_to_plot if m in list(all_results_flat.values())[0]]

            fig = make_subplots(rows=1, cols=len(available_metrics),
                                subplot_titles=available_metrics)
            for i, metric in enumerate(available_metrics):
                names = list(all_results_flat.keys())
                values = [all_results_flat[n].get(metric, 0) for n in names]
                colors = px.colors.qualitative.Set2[:len(names)]
                fig.add_trace(
                    go.Bar(x=names, y=values, marker_color=colors, showlegend=False),
                    row=1, col=i + 1,
                )
            fig.update_layout(height=400, title="Test Set Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)

            # RMSE comparison
            rmse_data = {n: all_results_flat[n].get("RMSE", 0) for n in all_results_flat}
            fig2 = go.Figure(go.Bar(
                x=list(rmse_data.keys()), y=list(rmse_data.values()),
                marker_color="indianred",
            ))
            fig2.update_layout(
                title="RMSE Comparison (lower is better)",
                yaxis_title="RMSE (cumecs)", height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Violin plot
            st.subheader("Error Distribution (Violin Plot)")
            fig_violin = plotly_violin(all_results_flat, list(models.keys()))
            if fig_violin:
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                png_violin = OUTPUT_DIR / "error_violin_plot.png"
                if png_violin.exists():
                    st.image(str(png_violin))

            # Ensemble weights
            if "weighted" in ensemble and "weights" in ensemble["weighted"]:
                st.subheader("Ensemble Weights")
                weights = ensemble["weighted"]["weights"]
                fig3 = go.Figure(go.Pie(
                    labels=list(weights.keys()),
                    values=list(weights.values()),
                    hole=0.4,
                ))
                fig3.update_layout(title="Optimized Ensemble Weights", height=350)
                st.plotly_chart(fig3, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 4: HYDROLOGY
    # ═══════════════════════════════════════════════════════════
    with tab4:
        st.header("Hydrological Analysis")

        # Hydrological signatures
        hydrology = results.get("hydrology", {})
        signatures = hydrology.get("signatures", {})
        if signatures:
            st.subheader("Basin Hydrological Signatures")
            sig_cols = st.columns(5)
            sig_items = list(signatures.items())
            for i, (k, v) in enumerate(sig_items):
                with sig_cols[i % 5]:
                    label = k.replace("_", " ").title()
                    st.markdown(f"""
                    <div class="hydro-card">
                        <h4>{label}</h4>
                        <h2>{v}</h2>
                    </div>
                    """, unsafe_allow_html=True)

        # Eckhardt baseflow separation
        st.subheader("Eckhardt Baseflow Separation")
        if raw_df is not None and "baseflow" in raw_df.columns:
            fig_eck = plotly_eckhardt(raw_df)
            if fig_eck:
                st.plotly_chart(fig_eck, use_container_width=True)
        else:
            png = OUTPUT_DIR / "eckhardt_separation.png"
            if png.exists():
                st.image(str(png))
            else:
                st.info("Eckhardt plot not available. Run the pipeline first.")

        # Flow Duration Curve
        st.subheader("Flow Duration Curve")
        if raw_df is not None:
            fig_fdc = plotly_fdc(raw_df)
            st.plotly_chart(fig_fdc, use_container_width=True)
        else:
            png = OUTPUT_DIR / "flow_duration_curve.png"
            if png.exists():
                st.image(str(png))

        # Flood Frequency
        ff = results.get("flood_frequency", {})
        if ff:
            st.subheader("Gumbel Flood Frequency Analysis")
            col1, col2 = st.columns([2, 1])
            with col1:
                annual_max = hydrology.get("annual_max", None)
                if annual_max:
                    annual_max = np.array(annual_max)
                fig_ff = plotly_flood_frequency(ff, annual_max)
                st.plotly_chart(fig_ff, use_container_width=True)
            with col2:
                st.markdown("**Return Period Discharges:**")
                for k, v in ff.items():
                    st.metric(k, f"{v:.1f} cumecs")

    # ═══════════════════════════════════════════════════════════
    # TAB 5: CLIMATE PROJECTIONS
    # ═══════════════════════════════════════════════════════════
    with tab5:
        st.header("CMIP6 Climate Projections")
        climate = results.get("climate", {})

        if not climate.get("projections") and not climate.get("ensemble_summary"):
            st.info("Climate projections not available. Run pipeline with `run_climate=True`.")
            st.markdown("""
            **To generate projections:**
            1. Download CMIP6 data: `python cmip6.py --shapefile data/cauvery_basin.shp`
            2. Run pipeline: `python main.py` (with `run_climate=True`)
            """)
        else:
            # Bias Correction Diagnostics
            if "bias_correction" in climate:
                st.subheader("Bias Correction Diagnostics (QDM)")
                bc_data = climate["bias_correction"]
                selected_gcm = st.selectbox("Select GCM", list(bc_data.keys()))
                if selected_gcm:
                    bc_df = pd.DataFrame(bc_data[selected_gcm])
                    cols = st.columns(4)
                    for i, var in enumerate(bc_df["variable"].unique()):
                        var_data = bc_df[bc_df["variable"] == var]
                        mean_bias = var_data["bias"].mean()
                        method = var_data["method"].iloc[0]
                        with cols[i % 4]:
                            metric_card(f"{var} ({method})", f"{mean_bias:+.3f}",
                                       "135deg, #2d4a3e, #3a6b5a")

                    fig_bc = px.bar(
                        bc_df, x="month", y="bias", color="variable", barmode="group",
                        title=f"Monthly Bias (GCM Hist − Observed) | {selected_gcm}",
                    )
                    fig_bc.update_layout(height=350)
                    st.plotly_chart(fig_bc, use_container_width=True)
                st.markdown("---")

            # Ensemble Summary
            if "ensemble_summary" in climate:
                st.subheader("Multi-Model Ensemble Summary")
                ens_df = pd.DataFrame(climate["ensemble_summary"])
                st.dataframe(ens_df, use_container_width=True, hide_index=True)

                fig_ens = px.bar(
                    ens_df, x="Period", y="Ensemble_Mean_Q",
                    color="SSP", barmode="group",
                    error_y="Ensemble_Std_Q",
                    title="Ensemble Mean Discharge with GCM Spread",
                )
                fig_ens.update_layout(height=400)
                st.plotly_chart(fig_ens, use_container_width=True)
                st.markdown("---")

            # Per-GCM Projections
            if "projections" in climate:
                st.subheader("Projection Details")
                proj_df = pd.DataFrame(climate["projections"])
                st.dataframe(proj_df, use_container_width=True, hide_index=True)

                # Change bar chart
                fig = px.bar(
                    proj_df, x="Period", y="Change_pct",
                    color="SSP", barmode="group",
                    title="Projected Discharge Change (%)",
                )
                if "GCM" in proj_df.columns and proj_df["GCM"].nunique() > 1:
                    fig = px.bar(
                        proj_df, x="Period", y="Change_pct",
                        color="SSP", barmode="group", facet_col="GCM",
                        title="Projected Discharge Change (%)",
                    )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

                # Heatmap
                if "GCM" in proj_df.columns and proj_df["GCM"].nunique() > 1:
                    pivot = proj_df.pivot_table(
                        values="Change_pct", index=["GCM", "SSP"], columns="Period"
                    )
                    fig2 = go.Figure(go.Heatmap(
                        z=pivot.values, x=pivot.columns,
                        y=[f"{g} / {s}" for g, s in pivot.index],
                        colorscale="RdBu_r", zmid=0,
                        text=np.round(pivot.values, 1), texttemplate="%{text}%",
                    ))
                    fig2.update_layout(
                        title="Discharge Change Heatmap (% from baseline)",
                        height=400,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Flow quantiles
                if "Q10" in proj_df.columns:
                    st.subheader("Flow Quantile Projections")
                    fig_q = go.Figure()
                    for ssp in proj_df["SSP"].unique():
                        ssp_data = proj_df[proj_df["SSP"] == ssp].groupby("Period").mean(
                            numeric_only=True
                        ).reset_index()
                        fig_q.add_trace(go.Scatter(
                            x=ssp_data["Period"], y=ssp_data["Q90"],
                            mode="lines", name=f"{ssp} Q90 (high flow)",
                            line=dict(dash="dot"),
                        ))
                        fig_q.add_trace(go.Scatter(
                            x=ssp_data["Period"], y=ssp_data["Q50"],
                            mode="lines+markers", name=f"{ssp} Q50 (median)",
                        ))
                        fig_q.add_trace(go.Scatter(
                            x=ssp_data["Period"], y=ssp_data["Q10"],
                            mode="lines", name=f"{ssp} Q10 (low flow)",
                            line=dict(dash="dash"),
                        ))
                    fig_q.update_layout(
                        title="Flow Quantile Projections",
                        yaxis_title="Discharge (cumecs)", height=400,
                    )
                    st.plotly_chart(fig_q, use_container_width=True)

        # Projection model info
        if climate.get("projection_model"):
            st.info(f"Projections generated using: **{climate['projection_model']}**")

    # ── Footer ──
    st.markdown("---")
    st.caption("FlowCast v2 — Yan et al. (2026), Eatesam et al. (2025), Dhakal et al. (2020)")


if __name__ == "__main__":
    main()