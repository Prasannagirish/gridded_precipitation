"""
FlowCast v2 — Visualization Module
Per-model hydrographs, scatter plots, violin plots, Eckhardt filter,
flow duration curve, flood frequency, and CMIP6 projection plots.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

PALETTE = sns.color_palette("husl", 10)
MODEL_COLORS = {}


def _get_color(model_name, idx=0):
    if model_name not in MODEL_COLORS:
        MODEL_COLORS[model_name] = PALETTE[len(MODEL_COLORS) % len(PALETTE)]
    return MODEL_COLORS[model_name]


# ════════════════════════════════════════════════════════════
#  PER-MODEL HYDROGRAPHS
# ════════════════════════════════════════════════════════════
def plot_model_hydrograph(dates, obs, pred, model_name, metrics_dict=None, output_dir="outputs"):
    """Plot individual hydrograph for a single model with obs vs pred."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    color = _get_color(model_name)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, obs, color="black", lw=1.2, alpha=0.7, label="Observed")
    ax.plot(dates, pred, color=color, lw=1.2, alpha=0.85, label=f"Predicted ({model_name})")
    ax.fill_between(dates, obs, pred, where=(pred > obs), interpolate=True, color="red", alpha=0.08)
    ax.fill_between(dates, obs, pred, where=(pred <= obs), interpolate=True, color="blue", alpha=0.08)

    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (cumecs)")
    title = f"Hydrograph — {model_name}"
    if metrics_dict:
        nse_val = metrics_dict.get("NSE", None)
        kge_val = metrics_dict.get("KGE", None)
        if nse_val is not None and kge_val is not None:
            title += f"  |  NSE={nse_val:.4f}  KGE={kge_val:.4f}"
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hydrograph_{model_name}.png", dpi=200)
    plt.close()


# ════════════════════════════════════════════════════════════
#  PER-MODEL SCATTER PLOTS
# ════════════════════════════════════════════════════════════
def plot_model_scatter(obs, pred, model_name, metrics_dict=None, output_dir="outputs"):
    """Scatter plot for a single model: Observed vs Predicted."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    color = _get_color(model_name)

    fig, ax = plt.subplots(figsize=(6, 6))
    min_val = min(np.min(obs), np.min(pred))
    max_val = max(np.max(obs), np.max(pred))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, alpha=0.6, label="Ideal (1:1)")
    ax.scatter(obs, pred, alpha=0.4, s=12, color=color, edgecolors="none")

    ax.set_xlabel("Observed Discharge (cumecs)")
    ax.set_ylabel("Predicted Discharge (cumecs)")
    title = f"Scatter — {model_name}"
    if metrics_dict:
        r2_val = metrics_dict.get("R2", None)
        rmse_val = metrics_dict.get("RMSE", None)
        if r2_val is not None and rmse_val is not None:
            title += f"\nR²={r2_val:.4f}  RMSE={rmse_val:.2f}"
    ax.set_title(title)
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_{model_name}.png", dpi=200)
    plt.close()


# ════════════════════════════════════════════════════════════
#  COMBINED SCATTER (ALL MODELS)
# ════════════════════════════════════════════════════════════
def plot_scatter_predictions(obs, pred_dict, output_dir="outputs"):
    """Combined scatter comparing all models."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    min_val = min(np.min(obs), min(np.min(p) for p in pred_dict.values()))
    max_val = max(np.max(obs), max(np.max(p) for p in pred_dict.values()))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Ideal (1:1)")

    for model_name, pred in pred_dict.items():
        color = _get_color(model_name)
        ax.scatter(obs, pred, alpha=0.4, s=12, label=model_name, color=color, edgecolors="none")

    ax.set_xlabel("Observed Discharge (cumecs)")
    ax.set_ylabel("Predicted Discharge (cumecs)")
    ax.set_title("All Models: Observed vs Predicted")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_comparison.png", dpi=200)
    plt.close()


# ════════════════════════════════════════════════════════════
#  BEST MODEL HYDROGRAPH (backward compat)
# ════════════════════════════════════════════════════════════
def plot_best_hydrograph(dates, obs, pred, model_name, output_dir="outputs"):
    plot_model_hydrograph(dates, obs, pred, model_name, output_dir=output_dir)


# ════════════════════════════════════════════════════════════
#  VIOLIN PLOT OF ERRORS
# ════════════════════════════════════════════════════════════
def plot_error_violin(obs, pred_dict, output_dir="outputs"):
    """Violin plot of prediction residuals for all models."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    data = []
    for model_name, pred in pred_dict.items():
        min_len = min(len(obs), len(pred))
        error = pred[:min_len] - obs[:min_len]
        df = pd.DataFrame({"Error": error, "Model": model_name})
        data.append(df)
    plot_df = pd.concat(data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(max(8, len(pred_dict) * 1.8), 6))
    sns.violinplot(x="Model", y="Error", data=plot_df, inner="quartile", ax=ax, palette="Set3")
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_ylabel("Prediction Error (Predicted − Observed)")
    ax.set_xlabel("Model")
    ax.set_title("Distribution of Prediction Errors (Residuals)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_violin_plot.png", dpi=200)
    plt.close()


# ════════════════════════════════════════════════════════════
#  ECKHARDT BASEFLOW SEPARATION (INVERTED RAINFALL Y-AXIS)
# ════════════════════════════════════════════════════════════
def plot_eckhardt_filter(dates, precip, discharge, baseflow, output_dir="outputs"):
    """
    Plots the Eckhardt baseflow separation with a clean, inverted rainfall hyetograph.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # ─── Primary Axis: Discharge and Baseflow ───
    ax1.plot(dates, discharge, color='#2E86C1', linewidth=1.2, label='Total Discharge')
    ax1.fill_between(dates, baseflow, 0, color='#27AE60', alpha=0.6, label='Baseflow (Eckhardt)')
    
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Flow (m³/s)', color='#2E86C1', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2E86C1')
    ax1.set_ylim(bottom=0)
    
    # ─── Secondary Axis: Rainfall (Inverted) ───
    ax2 = ax1.twinx()
    # Use a slightly darker grey so it stands out, but make it semi-transparent
    ax2.bar(dates, precip, color='#7F8C8D', alpha=0.5, width=1.0, label='Precipitation')
    ax2.set_ylabel('Rainfall (mm)', color='#7F8C8D', fontweight='bold')
    
    # Invert the y-axis and multiply max by 3 so rain only occupies the top third
    ax2.set_ylim(bottom=0, top=precip.max() * 3) 
    ax2.invert_yaxis()
    ax2.tick_params(axis='y', labelcolor='#7F8C8D')
    
    # ─── X-Axis Date Formatting (Fixes the black blob) ───
    ax1.xaxis.set_major_locator(mdates.YearLocator(2)) # Tick every 2 years
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Baseflow Separation (Eckhardt Filter) with Rainfall Hyetograph', pad=15, fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    
    save_path = out_dir / "eckhardt_separation.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

# ════════════════════════════════════════════════════════════
#  FLOOD DURATION CURVE
# ════════════════════════════════════════════════════════════
def plot_flow_duration_curve(discharge, output_dir="outputs"):
    """
    Plots a clean Flow Duration Curve on a log scale with Q50 and Q90 annotations.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate exceedance probability
    sorted_flow = np.sort(discharge)[::-1]
    ranks = np.arange(1, len(sorted_flow) + 1)
    prob = (ranks / (len(sorted_flow) + 1)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prob, sorted_flow, color='#3498db', linewidth=2)
    
    # Log scale for accurate hydrological representation
    ax.set_yscale('log')
    ax.set_xlim(0, 100)
    
    # ─── Annotations for Q50 and Q90 ───
    # Find the exact values for the 50% and 90% exceedance marks
    idx_50 = np.argmin(np.abs(prob - 50))
    idx_90 = np.argmin(np.abs(prob - 90))
    val_50 = sorted_flow[idx_50]
    val_90 = sorted_flow[idx_90]
    
    # Add horizontal dashed lines
    ax.axhline(y=val_50, color='gray', linestyle='--', alpha=0.6)
    ax.axhline(y=val_90, color='gray', linestyle='--', alpha=0.6)
    
    # Add text and arrows
    ax.annotate(f'Q50 = {val_50:.1f}', 
                xy=(50, val_50), xytext=(55, val_50 * 1.5),
                arrowprops=dict(arrowstyle="->", color='gray'))
                
    ax.annotate(f'Q90 = {val_90:.1f}', 
                xy=(90, val_90), xytext=(75, val_90 * 0.5), # Put text below line for Q90
                arrowprops=dict(arrowstyle="->", color='gray'))

    ax.set_xlabel('Exceedance Probability (%)', fontweight='bold')
    ax.set_ylabel('Discharge (m³/s, log scale)', fontweight='bold')
    plt.title('Flow Duration Curve', pad=15, fontweight='bold', fontsize=14)
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.tight_layout()
    
    save_path = out_dir / "flow_duration_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
# ════════════════════════════════════════════════════════════
#  FLOOD FREQUENCY CURVE
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
#  FLOOD FREQUENCY CURVE
# ════════════════════════════════════════════════════════════
def plot_flood_frequency(annual_max, flood_freq_dict, output_dir="outputs"):
    """
    Plots the empirical annual maxima against the theoretical Gumbel distribution.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Empirical plotting positions (Weibull)
    sorted_max = np.sort(annual_max)[::-1]
    n = len(sorted_max)
    ranks = np.arange(1, n + 1)
    return_periods = (n + 1) / ranks

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot empirical data
    ax.scatter(return_periods, sorted_max, color='#e74c3c', s=40, label='Observed Annual Maxima', zorder=5)

    # 2. Plot theoretical (Gumbel) from dictionary
    gumbel_t = []
    gumbel_q = []
    for k, v in flood_freq_dict.items():
        try:
            # Parse 'Q2yr', 'Q50yr', etc., to get the numeric return period
            t = int(k.replace('Q', '').replace('yr', ''))
            gumbel_t.append(t)
            gumbel_q.append(v)
        except ValueError:
            pass

    if gumbel_t and gumbel_q:
        # Sort values by return period to draw a clean line
        sort_idx = np.argsort(gumbel_t)
        gumbel_t = np.array(gumbel_t)[sort_idx]
        gumbel_q = np.array(gumbel_q)[sort_idx]
        
        ax.plot(gumbel_t, gumbel_q, color='#2c3e50', linestyle='--', linewidth=2, label='Gumbel Theoretical Curve')

        # Annotate the theoretical points
        for t, q in zip(gumbel_t, gumbel_q):
            ax.annotate(f"{q:.0f}", (t, q), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='#2c3e50')

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Return Period (Years, Log Scale)', fontweight='bold')
    ax.set_ylabel('Peak Discharge (m³/s)', fontweight='bold')
    plt.title('Flood Frequency Analysis (Gumbel Distribution)', pad=15, fontweight='bold', fontsize=14)

    # Add custom tick labels for standard return periods on the log axis
    xticks = [1.1, 2, 5, 10, 25, 50, 100]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()

    save_path = out_dir / "flood_frequency.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
# ════════════════════════════════════════════════════════════
#  CMIP6 PROJECTION PLOTS
# ════════════════════════════════════════════════════════════
def plot_cmip6_projections(summary_df, baseline_mean, output_dir="outputs"):
    """Bar chart of CMIP6 projected discharge change by scenario and period."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    if summary_df is None or summary_df.empty:
        return

    ssps = summary_df["SSP"].unique()
    n_ssps = len(ssps)
    fig, axes = plt.subplots(1, n_ssps, figsize=(7 * n_ssps, 5), sharey=True, squeeze=False)

    for i, ssp in enumerate(ssps):
        ax = axes[0, i]
        ssp_data = summary_df[summary_df["SSP"] == ssp]
        periods = sorted(ssp_data["Period"].unique())
        gcms = ssp_data["GCM"].unique()
        x = np.arange(len(periods))
        width = 0.8 / max(len(gcms), 1)

        for j, gcm in enumerate(gcms):
            gcm_data = ssp_data[ssp_data["GCM"] == gcm]
            vals = []
            for p in periods:
                match = gcm_data[gcm_data["Period"] == p]["Change_pct"]
                vals.append(match.values[0] if len(match) > 0 else 0)
            ax.bar(x + j * width, vals, width, label=gcm, alpha=0.8)

        ax.set_xticks(x + width * len(gcms) / 2)
        ax.set_xticklabels(periods, rotation=15)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"Discharge Change — {ssp.upper()}")
        ax.set_ylabel("Change from Baseline (%)")
        ax.legend(fontsize=7, ncol=2)

    plt.suptitle(f"CMIP6 Projected Discharge Change (baseline = {baseline_mean:.1f} cumecs)", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cmip6_projections.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_cmip6_ensemble_summary(ensemble_df, output_dir="outputs"):
    """Ensemble mean discharge with GCM spread per SSP x period."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    if ensemble_df is None or ensemble_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ssps = ensemble_df["SSP"].unique()
    periods = sorted(ensemble_df["Period"].unique())
    x = np.arange(len(periods))
    width = 0.35

    for i, ssp in enumerate(ssps):
        ssp_data = ensemble_df[ensemble_df["SSP"] == ssp].sort_values("Period")
        # Align to periods
        means = [ssp_data[ssp_data["Period"] == p]["Ensemble_Mean_Q"].values[0]
                 if p in ssp_data["Period"].values else 0 for p in periods]
        stds = [ssp_data[ssp_data["Period"] == p]["Ensemble_Std_Q"].values[0]
                if p in ssp_data["Period"].values else 0 for p in periods]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=ssp.upper(), alpha=0.8)

    ax.set_xticks(x + width * len(ssps) / 2)
    ax.set_xticklabels(periods)
    ax.set_ylabel("Mean Discharge (cumecs)")
    ax.set_title("Multi-Model Ensemble: Projected Mean Discharge")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cmip6_ensemble_summary.png", dpi=200)
    plt.close()
    


def plot_top_5_features(model, X_val, y_val, feature_names, model_name="Model", output_dir="outputs"):
    """
    Calculates and plots the Top 5 most important features.
    Uses native feature_importances_ for trees, and permutation importance for SVR/Deep Learning.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    importances = None
    
    # Strategy 1: Native Tree Importances (XGBoost, Random Forest)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    
    # Strategy 2: Permutation Importance (SVR, LSTMs via wrapper)
    else:
        print(f"  Calculating permutation importance for {model_name}...")
        result = permutation_importance(
            model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean

    # Sort and slice Top 5
    indices = np.argsort(importances)[-5:]
    top_5_importances = importances[indices]
    top_5_features = [feature_names[i] for i in indices]

    # Generate Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_5_importances, y=top_5_features, palette="viridis")
    
    plt.title(f"Top 5 Feature Importances ({model_name})", pad=15, fontweight='bold')
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Hydrological Features")
    plt.tight_layout()

    # Save
    save_path = Path(output_dir) / f"{model_name}_top5_features.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  ✓ Top 5 feature graph saved to {save_path}")
    


def plot_cmip6_timeseries_with_rainfall(all_projections, output_dir="outputs"):
    """
    Generates a dual-axis plot for each CMIP6 projection scenario, 
    showing discharge as a line graph and rainfall as inverted bars.
    """
    out_dir = Path(output_dir) / "cmip6_detailed_forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, data in all_projections.items():
        dates = data["dates"]
        discharge = data["discharge"]
        
        # Extract rainfall from the features dataframe
        if "precip" in data["features"].columns:
            precip = data["features"]["precip"]
        else:
            print(f"  ⚠ Skipping rainfall plot for {key}: 'precip' feature not found.")
            continue

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # ─── Primary Axis: Discharge ───
        color_discharge = '#c0392b' # Deep red for SSPs
        if 'ssp245' in key:
            color_discharge = '#8e44ad' # Purple for SSP245
            
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Projected Discharge (m³/s)', color=color_discharge, fontweight='bold')
        ax1.plot(dates, discharge, color=color_discharge, alpha=0.85, linewidth=1.2, label='Discharge')
        ax1.tick_params(axis='y', labelcolor=color_discharge)
        ax1.set_ylim(bottom=0) # Ensure discharge doesn't go below 0

        # ─── Secondary Axis: Rainfall (Inverted) ───
        ax2 = ax1.twinx()
        color_precip = '#3498db' # Blue for rainfall
        ax2.set_ylabel('Projected Rainfall (mm)', color=color_precip, fontweight='bold')
        # Use width=1.0 for continuous daily bars without gaps
        ax2.bar(dates, precip, color=color_precip, alpha=0.4, width=1.0, label='Rainfall')
        ax2.tick_params(axis='y', labelcolor=color_precip)
        
        # Invert the y-axis so rainfall "hangs" from the top
        ax2.set_ylim(bottom=0, top=max(precip)*3) # Multiply by 3 so rain only takes up top third of graph
        ax2.invert_yaxis() 

        # Formatting date axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(2)) # Tick every 2 years
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Titles and Grids
        parts = key.split('_')
        gcm, ssp = parts[0], parts[-3] if len(parts) > 3 else "Unknown"
        plt.title(f"Future Discharge vs Rainfall Projection: {gcm} ({ssp.upper()})", pad=15, fontsize=14, fontweight='bold')
        
        ax1.grid(True, alpha=0.2)
        fig.tight_layout()

        # Save Plot
        save_path = out_dir / f"{key}_forecast_plot.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"  ✓ Detailed forecast plots saved to {out_dir}/")


def plot_cmip6_yearly_timeseries(all_projections, output_dir="outputs"):
    """
    Plots the annual mean discharge and annual total rainfall for CMIP6 scenarios.
    """
    out_dir = Path(output_dir) / "cmip6_yearly_forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, data in all_projections.items():
        if "precip" not in data["features"].columns:
            continue

        # Build a temporary DataFrame for easy time-series resampling
        df = pd.DataFrame({
            "discharge": data["discharge"],
            "precip": data["features"]["precip"].values
        }, index=pd.to_datetime(data["dates"]))

        # Aggregate: Mean for river flow, Sum for total rainfall
        annual_df = df.resample('YE').agg({
            'discharge': 'mean',
            'precip': 'sum'
        })
        
        # Shift index to just the year for cleaner X-axis plotting
        annual_df.index = annual_df.index.year

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # ─── Primary Axis: Annual Mean Discharge ───
        color_discharge = '#8e44ad' if 'ssp245' in key else '#c0392b'
        ax1.set_xlabel('Year', fontweight='bold')
        ax1.set_ylabel('Annual Mean Discharge (m³/s)', color=color_discharge, fontweight='bold')
        
        # Using a line with markers for annual trends
        ax1.plot(annual_df.index, annual_df['discharge'], marker='o', color=color_discharge, linewidth=2, markersize=6, label='Discharge')
        ax1.tick_params(axis='y', labelcolor=color_discharge)
        ax1.set_ylim(bottom=0) 

        # ─── Secondary Axis: Total Annual Rainfall (Inverted Bars) ───
        ax2 = ax1.twinx()
        color_precip = '#3498db'
        ax2.set_ylabel('Total Annual Rainfall (mm)', color=color_precip, fontweight='bold')
        ax2.bar(annual_df.index, annual_df['precip'], color=color_precip, alpha=0.3, width=0.6, label='Rainfall')
        ax2.tick_params(axis='y', labelcolor=color_precip)
        
        # Invert the y-axis so rainfall hangs from the top
        ax2.set_ylim(bottom=0, top=annual_df['precip'].max() * 3) 
        ax2.invert_yaxis() 

        # Titles and Grid
        parts = key.split('_')
        gcm, ssp = parts[0], parts[-3] if len(parts) > 3 else "Unknown"
        plt.title(f"Yearly Trend: Discharge vs Total Rainfall - {gcm} ({ssp.upper()})", pad=15, fontweight='bold')
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()

        # Save Plot
        save_path = out_dir / f"{key}_yearly_trend.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"  ✓ Yearly aggregated forecast plots saved to {out_dir}/")


def plot_cmip6_annual_regime(all_projections, output_dir="outputs"):
    """
    Plots the average annual regime (climatology) hydrograph.
    Averages all ~30 years of daily predictions into a single 12-month typical year 
    to see how peak monsoon flows change under different scenarios.
    """
    out_dir = Path(output_dir)
    plt.figure(figsize=(12, 6))
    
    # We will use a seaborn color palette to handle multiple lines cleanly
    colors = sns.color_palette("husl", len(all_projections))
    
    for (key, data), color in zip(all_projections.items(), colors):
        df = pd.DataFrame({
            "discharge": data["discharge"]
        }, index=pd.to_datetime(data["dates"]))
        
        # Group by the month of the year to get the long-term monthly average
        df['month'] = df.index.month
        monthly_regime = df.groupby('month')['discharge'].mean()
        
        # Format label
        parts = key.split('_')
        label = f"{parts[0]} ({parts[-3]})" if len(parts) > 3 else key
        
        # Plot the regime curve
        plt.plot(monthly_regime.index, monthly_regime.values, marker='s', linewidth=2, color=color, label=label)

    plt.title("Long-Term Average Annual Regime Hydrograph (2020-2050)", pad=15, fontweight='bold', fontsize=14)
    plt.xlabel("Month", fontweight='bold')
    plt.ylabel("Average Monthly Discharge (m³/s)", fontweight='bold')
    
    # Force X-axis to show month names
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(ticks=range(1, 13), labels=months)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = out_dir / "cmip6_regime_hydrograph_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  ✓ CMIP6 regime hydrograph (climatology) saved to {out_dir}/")