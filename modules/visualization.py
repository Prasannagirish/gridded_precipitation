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
    Eckhardt baseflow separation with:
    - Bottom Y-axis: Total discharge + baseflow shading
    - Top Y-axis: Inverted rainfall hyetograph (bars hanging from top)
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # ── Bottom axis: Discharge & Baseflow ──
    ax1.plot(dates, discharge, color="#1f77b4", lw=1.5, label="Total Discharge")
    ax1.fill_between(dates, 0, baseflow, color="#2ca02c", alpha=0.4, label="Baseflow (Eckhardt)")
    ax1.plot(dates, baseflow, color="#175c17", lw=1.0)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Flow (cumecs)", color="#1f77b4", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(bottom=0)

    # ── Top axis: Inverted rainfall ──
    ax2 = ax1.twinx()
    ax2.bar(dates, precip, width=1.0, color="#7f7f7f", alpha=0.55, label="Precipitation")
    ax2.set_ylabel("Rainfall (mm)", color="#7f7f7f", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f")
    # INVERT the rainfall axis so bars hang from top
    max_precip = np.nanmax(precip) if np.nanmax(precip) > 0 else 1.0
    ax2.set_ylim(max_precip * 3, 0)
    ax2.invert_yaxis()

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, facecolor="white")

    plt.title("Baseflow Separation (Eckhardt Filter) with Rainfall Hyetograph")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eckhardt_separation.png", dpi=200)
    plt.close()


# ════════════════════════════════════════════════════════════
#  FLOW DURATION CURVE
# ════════════════════════════════════════════════════════════
def plot_flow_duration_curve(discharge, output_dir="outputs"):
    """Flow Duration Curve: exceedance probability vs discharge."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    sorted_q = np.sort(discharge)[::-1]
    n = len(sorted_q)
    exceedance = np.arange(1, n + 1) / (n + 1) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(exceedance, sorted_q, color="#1f77b4", lw=2)
    ax.fill_between(exceedance, sorted_q, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel("Discharge (cumecs, log scale)")
    ax.set_title("Flow Duration Curve")
    ax.set_xlim(0, 100)
    ax.grid(True, which="both", alpha=0.3)

    for pct, label in [(5, "Q5 (high)"), (50, "Q50 (median)"), (95, "Q95 (low)")]:
        idx = min(int(pct / 100 * n), n - 1)
        ax.annotate(f"{label}\n{sorted_q[idx]:.1f}", xy=(pct, sorted_q[idx]),
                    fontsize=9, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/flow_duration_curve.png", dpi=200)
    plt.close()


# ════════════════════════════════════════════════════════════
#  FLOOD FREQUENCY CURVE
# ════════════════════════════════════════════════════════════
def plot_flood_frequency(annual_max, flood_freq_dict, output_dir="outputs"):
    """Gumbel flood frequency curve with return periods."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    rp = [int(''.join(filter(str.isdigit, k))) for k in flood_freq_dict.keys()]
    qvals = list(flood_freq_dict.values())

    # Plotting positions for observed annual maxima
    n = len(annual_max)
    sorted_am = np.sort(annual_max)[::-1]
    plotting_rp = (n + 1) / np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rp, qvals, "o-", color="#D32F2F", lw=2, markersize=8, label="Gumbel Fit")
    ax.scatter(plotting_rp, sorted_am, color="#1f77b4", s=40, alpha=0.7,
              zorder=5, label="Observed Annual Max")
    ax.set_xscale("log")
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Discharge (cumecs)")
    ax.set_title("Gumbel Flood Frequency Analysis")
    ax.legend(frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/flood_frequency.png", dpi=200)
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