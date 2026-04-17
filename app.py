import streamlit as st
import subprocess
import json
import pandas as pd
from pathlib import Path
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="FlowCast v2 · Kabini Basin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
HYDRO_DIR = BASE_DIR / "hydrological_outputs"
RESULTS_JSON = OUTPUTS_DIR / "pipeline_results.json"

# ============================================================
# CSS — dark-native
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .block-container { padding-top: 1.5rem; max-width: 1400px; }

  .hero {
    background: linear-gradient(135deg, #0c1a2e 0%, #0f2b4a 50%, #0a3550 100%);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 2.6rem 2.8rem 2.2rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
  }
  .hero::before {
    content: ''; position: absolute; inset: 0;
    background:
      radial-gradient(ellipse 50% 50% at 75% 15%, rgba(56,189,248,.09) 0%, transparent 70%),
      radial-gradient(ellipse 35% 55% at 10% 85%, rgba(14,165,233,.06) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero h1 { font-family:'DM Sans',sans-serif; font-weight:700; font-size:2.2rem; color:#e0f2fe; margin:0 0 .4rem; letter-spacing:-0.4px; }
  .hero p  { color:#8faabe; font-size:.98rem; margin:0; line-height:1.55; }
  .hero .hl { color:#38bdf8; font-weight:500; }

  .metric-row { display:flex; gap:.8rem; margin:1.2rem 0 .6rem; }
  .metric-tile {
    flex:1; background:rgba(56,189,248,.06); border:1px solid rgba(56,189,248,.18);
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
  }
  .metric-tile .label {
    font-family:'JetBrains Mono',monospace; font-size:.68rem; font-weight:600;
    color:#38bdf8; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:.25rem;
  }
  .metric-tile .value { font-weight:700; font-size:1.45rem; color:#e0f2fe; }

  .metric-tile-green {
    flex:1; background:rgba(34,197,94,.06); border:1px solid rgba(34,197,94,.18);
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
  }
  .metric-tile-green .label {
    font-family:'JetBrains Mono',monospace; font-size:.68rem; font-weight:600;
    color:#4ade80; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:.25rem;
  }
  .metric-tile-green .value { font-weight:700; font-size:1.45rem; color:#d1fae5; }
  
  .metric-tile-purple {
    flex:1; background:rgba(168,85,247,.06); border:1px solid rgba(168,85,247,.18);
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
  }
  .metric-tile-purple .label {
    font-family:'JetBrains Mono',monospace; font-size:.68rem; font-weight:600;
    color:#c084fc; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:.25rem;
  }
  .metric-tile-purple .value { font-weight:700; font-size:1.45rem; color:#f3e8ff; }

  .gallery-caption {
    font-family:'JetBrains Mono',monospace; font-size:.7rem;
    color:rgba(255,255,255,.5); text-align:center; margin-top:.5rem; margin-bottom: 2rem; letter-spacing:.4px;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def run_pipeline():
    script_path = BASE_DIR / "main.py"
    if not script_path.exists():
        st.error(f"Script not found: `main.py`")
        return False
    with st.spinner(f"Executing FlowCast v2 Pipeline (This may take ~30 mins)..."):
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True,
        )
    if result.returncode == 0:
        st.success("Pipeline Execution Complete!")
        with st.expander("View Terminal Output"):
            st.code(result.stdout, language="text")
        return True
    else:
        st.error("Pipeline Failed")
        with st.expander("Error log", expanded=True):
            st.code(result.stderr, language="text")
        return False

def show_image_gallery(directory: Path, pattern: str, cols: int = 2, caption_replace: str = "_"):
    """Neatly displays images matching a specific pattern."""
    if not directory.exists():
        return
    png_files = sorted(directory.glob(pattern))
    if not png_files:
        st.info(f"No plots found matching '{pattern}'")
        return
        
    grid = st.columns(cols)
    for i, img_path in enumerate(png_files):
        with grid[i % cols]:
            st.image(Image.open(img_path), use_container_width=True)
            caption = img_path.stem.replace(caption_replace, " ").title()
            st.markdown(f'<p class="gallery-caption">{caption}</p>', unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
  <h1>🌊 FlowCast: Kabini Basin</h1>
  <p>
    End-to-end ML pipeline — from raw <span class="hl">CHIRPS rainfall</span>
    and <span class="hl">CWC discharge</span> data through
    <span class="hl">hydrological analysis</span>, <span class="hl">XGBoost & LSTMs</span>, and 
    <span class="hl">CMIP6 SSP Projections</span>.
  </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STATE CHECK
# ============================================================
pipeline_run_done = RESULTS_JSON.exists()
results_data = {}
if pipeline_run_done:
    try:
        with open(RESULTS_JSON, "r") as f:
            results_data = json.load(f)
    except Exception:
        pass

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Control Center",
    "💧 Catchment Physics",
    "📊 ML Performance",
    "🌍 Climate Projections",
])
# ------ TAB 1: CONTROL CENTER ------
with tab1:
    st.markdown("### Basin Overview")
    
    # Check for the basin preview image and display it centered
    map_path = BASE_DIR / "cauvery_basin_preview.jpeg"
    if map_path.exists():
        # Using columns to neatly center the image so it doesn't stretch too wide on large screens
        colA, colB, colC = st.columns([1, 2, 1]) 
        with colB:
            st.image(Image.open(map_path), use_container_width=True)
    else:
        st.info("💡 Save your uploaded shapefile image as `cauvery_basin_preview.jpeg` in the main project folder to display it here.")
        
    st.divider()
    
    # Moved Execution and Metrics to the bottom
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Execution")
        st.markdown('<p style="color:gray; font-size: 0.9em;">Triggers the unified `main.py` pipeline. This processes data, trains all models via PSO, and projects CMIP6 scenarios.</p>', unsafe_allow_html=True)
        if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
            if run_pipeline():
                st.rerun()
                
    with col2:
        st.markdown("### High-Level Metrics")
        if pipeline_run_done and "models" in results_data:
            # Extract top model stats
            models = results_data["models"]
            svr_nse = models.get("SVR-PSO", {}).get("test", {}).get("NSE", 0)
            xgb_nse = models.get("XGBoost-PSO", {}).get("test", {}).get("NSE", 0)
            lstm_nse = models.get("LSTM", {}).get("test", {}).get("NSE", 0)
            agru_nse = models.get("A-GRU", {}).get("test", {}).get("NSE", 0)
            
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-tile-green">
                <div class="label">SVR-PSO NSE</div>
                <div class="value">{svr_nse:.4f}</div>
              </div>
              <div class="metric-tile">
                <div class="label">XGBoost-PSO NSE</div>
                <div class="value">{xgb_nse:.4f}</div>
              </div>
              <div class="metric-tile-purple">
                <div class="label">LSTM NSE</div>
                <div class="value">{lstm_nse:.4f}</div>
              </div>
              <div class="metric-tile-purple">
                <div class="label">A-GRU NSE</div>
                <div class="value">{agru_nse:.4f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run the pipeline to generate metrics.")
# ------ TAB 2: CATCHMENT PHYSICS ------
with tab2:
    if not pipeline_run_done:
        st.warning("Pipeline has not been run. Please run the pipeline in the Control Center.")
    else:
        st.markdown("### Fundamental Hydrology")
        show_image_gallery(HYDRO_DIR, "eckhardt_separation.png", cols=1)
        show_image_gallery(HYDRO_DIR, "*flow_duration*.png", cols=2)
        show_image_gallery(HYDRO_DIR, "*flood_frequency*.png", cols=2)
        
        st.divider()
        st.markdown("### Advanced Signatures")
        colA, colB = st.columns(2)
        with colA:
            show_image_gallery(HYDRO_DIR, "recession_curve.png", cols=1)
        with colB:
            show_image_gallery(HYDRO_DIR, "snyder_unit_hydrograph.png", cols=1)

# ------ TAB 3: ML PERFORMANCE ------
with tab3:
    if not pipeline_run_done:
        st.warning("Pipeline has not been run.")
    else:
        st.markdown("### 1. Global Model Comparison")
        colA, colB = st.columns(2)
        with colA:
            show_image_gallery(OUTPUTS_DIR, "error_violin_plot.png", cols=1)
        
        st.divider()
        st.markdown("### 2. Feature Importance")
        st.markdown("*(Permutation Importance for SVR, Native Split/Gain for Trees)*")
        show_image_gallery(OUTPUTS_DIR, "*_top5_features.png", cols=3, caption_replace="_top5_features")
        
        st.divider()
        st.markdown("### 3. Individual Hydrographs & Scatter Fits")
        show_image_gallery(OUTPUTS_DIR, "hydrograph_*.png", cols=1, caption_replace="hydrograph_")
        show_image_gallery(OUTPUTS_DIR, "scatter_*.png", cols=2, caption_replace="scatter_")

# ------ TAB 4: CLIMATE PROJECTIONS ------
with tab4:
    if not pipeline_run_done:
        st.warning("Pipeline has not been run.")
    else:
        # Display CMIP6 Projections Summary Table
        if "climate" in results_data and "projections" in results_data["climate"]:
            st.markdown("### CMIP6 Projection Summary (2015-2050)")
            df_proj = pd.DataFrame(results_data["climate"]["projections"])
            st.dataframe(df_proj, use_container_width=True)
            
        st.divider()
        st.markdown("### Long-Term Climate Shifts")
        colA, colB = st.columns([1, 1])
        with colA:
            show_image_gallery(OUTPUTS_DIR, "cmip6_regime_hydrograph*.png", cols=1)
        with colB:
            show_image_gallery(OUTPUTS_DIR, "cmip6_projections.png", cols=1)
            
        st.divider()
        st.markdown("### Yearly Trends: Discharge vs Total Rainfall")
        yearly_dir = OUTPUTS_DIR / "cmip6_yearly_forecasts"
        show_image_gallery(yearly_dir, "*.png", cols=2, caption_replace="_yearly_trend")

        st.divider()
        st.markdown("### Continuous Dual-Axis Forecasts (2015 - 2050)")
        detailed_dir = OUTPUTS_DIR / "cmip6_detailed_forecasts"
        show_image_gallery(detailed_dir, "*.png", cols=1, caption_replace="_forecast_plot")