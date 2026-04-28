"""
FlowCast v2 — Visualization Module
Per-model hydrographs (annual aggregated), scatter plots, violin plots,
Eckhardt filter, flow duration curve, flood frequency, CMIP6 projection
plots, and MODEL PERFORMANCE RADAR CHART.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance
from math import pi

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

PALETTE = sns.color_palette("husl", 10)
MODEL_COLORS = {}


def _get_color(model_name, idx=0):
    if model_name not in MODEL_COLORS:
        MODEL_COLORS[model_name] = PALETTE[len(MODEL_COLORS) % len(PALETTE)]
    return MODEL_COLORS[model_name]


# ════════════════════════════════════════════════════════════
#  PER-MODEL HYDROGRAPHS  (ANNUAL AGGREGATION)
# ════════════════════════════════════════════════════════════
def plot_model_hydrograph(dates, obs, pred, model_name, metrics_dict=None, output_dir="outputs"):
    """
    Plot annual-mean hydrograph for a single model (obs vs pred).
    Daily inputs are resampled to yearly means before plotting so the
    chart is legible and publication-quality.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    color = _get_color(model_name)

    # ── Annual aggregation ──────────────────────────────────
    ann = pd.DataFrame({"obs": np.asarray(obs), "pred": np.asarray(pred)},
                       index=pd.to_datetime(dates))
    ann = ann.resample("YE").mean()
    ann_dates = ann.index.to_list()
    ann_obs   = ann["obs"].values
    ann_pred  = ann["pred"].values
    # ────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ann_dates, ann_obs,  color="black", lw=1.5, marker="o", ms=5,
            alpha=0.85, label="Observed (Annual Mean)")
    ax.plot(ann_dates, ann_pred, color=color,   lw=1.5, marker="s", ms=5,
            alpha=0.90, label=f"Predicted ({model_name}) — Annual Mean")
    ax.fill_between(ann_dates, ann_obs, ann_pred,
                    where=(ann_pred > ann_obs), interpolate=True,
                    color="red",  alpha=0.12, label="Over-prediction")
    ax.fill_between(ann_dates, ann_obs, ann_pred,
                    where=(ann_pred <= ann_obs), interpolate=True,
                    color="blue", alpha=0.12, label="Under-prediction")

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Annual Discharge (cumecs)")
    title = f"Annual Hydrograph — {model_name}"
    if metrics_dict:
        nse_val = metrics_dict.get("NSE", None)
        kge_val = metrics_dict.get("KGE", None)
        if nse_val is not None and kge_val is not None:
            title += f"  |  NSE={nse_val:.4f}  KGE={kge_val:.4f}"
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hydrograph_{model_name}.png", dpi=200)
    plt.close()


def plot_model_hydrograph_daily(dates, obs, pred, model_name, metrics_dict=None, output_dir="outputs"):
    """
    Raw daily hydrograph — kept for debugging / audit purposes.
    NOT called by default; invoke explicitly when needed.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    color = _get_color(model_name)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, obs,  color="black", lw=0.8, alpha=0.65, label="Observed")
    ax.plot(dates, pred, color=color,   lw=0.8, alpha=0.75,
            label=f"Predicted ({model_name})")
    ax.fill_between(dates, obs, pred, where=(pred > obs),
                    interpolate=True, color="red",  alpha=0.08)
    ax.fill_between(dates, obs, pred, where=(pred <= obs),
                    interpolate=True, color="blue", alpha=0.08)

    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (cumecs)")
    title = f"Daily Hydrograph — {model_name}"
    if metrics_dict:
        nse_val = metrics_dict.get("NSE", None)
        kge_val = metrics_dict.get("KGE", None)
        if nse_val is not None and kge_val is not None:
            title += f"  |  NSE={nse_val:.4f}  KGE={kge_val:.4f}"
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hydrograph_daily_{model_name}.png", dpi=200)
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
#  BEST MODEL HYDROGRAPH (backward compat — now annual)
# ════════════════════════════════════════════════════════════
def plot_best_hydrograph(dates, obs, pred, model_name, output_dir="outputs"):
    """Backward-compatible wrapper — delegates to annual hydrograph."""
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
#  MODEL PERFORMANCE RADAR CHART  (one polygon sub-radar per metric)
# ════════════════════════════════════════════════════════════
# Metrics where LOWER raw value = better performance.
# These are inverted before plotting so a larger polygon always = better model.
_INVERT_METRICS = {"RMSE", "PBIAS", "MAE", "MSE", "PBIAS_score"}


def _draw_single_radar(ax, model_scores: dict, metric_name: str,
                       fill_color: str):
    """
    Draw one polygon radar subplot for a single metric.

    - Spokes = models, radius = score on this metric.
    - For error metrics (RMSE, PBIAS) the axis is INVERTED so that a
      *larger* polygon still means *better* performance, matching the
      visual convention of the other panels.
    - Axis labels always show the original raw values so the reader
      can still read off actual RMSE / PBIAS numbers.
    - Auto-scales to the actual data range with 15 % headroom.

    Parameters
    ----------
    ax           : polar Axes (already created as polar=True)
    model_scores : { model_name: raw_value }  – any numeric range
    metric_name  : string shown in the subplot title
    fill_color   : hex / RGB colour for the filled polygon
    """
    models = list(model_scores.keys())
    N      = len(models)
    if N < 2:
        return

    raw_scores = [model_scores[m] for m in models]

    # ── Detect whether this is an "error" metric ────────────
    is_error = any(err in metric_name for err in ("RMSE", "PBIAS", "MAE", "MSE"))

    if is_error:
        # Invert: best model (lowest error) gets highest radius.
        # plot_scores = max + min - raw  →  min maps to max, max maps to min
        s_min = min(raw_scores)
        s_max = max(raw_scores)
        plot_scores = [s_max + s_min - v for v in raw_scores]
        # Keep the ORIGINAL labels so tick values show real RMSE/PBIAS
        display_scores = raw_scores          # used only for tick labelling
    else:
        plot_scores    = raw_scores
        display_scores = raw_scores

    # Evenly-spaced angles, first spoke at top (pi/2 offset), clockwise
    angles        = [n / float(N) * 2 * pi for n in range(N)]
    angles_closed = angles + angles[:1]
    plot_closed   = plot_scores + plot_scores[:1]

    # ── Auto-scale radial axis to the PLOT range ────────────
    data_min = min(plot_scores)
    data_max = max(plot_scores)
    span     = data_max - data_min if data_max != data_min else (data_max * 0.2 or 0.2)
    headroom = span * 0.15

    r_min = max(0.0, data_min - headroom)
    r_max = data_max + headroom

    # ── Tick labels: map plot_score ticks back to raw values ─
    tick_plot = np.linspace(r_min, r_max, 5)
    if is_error:
        # Invert tick values back to raw domain for display
        tick_raw = [s_max + s_min - t for t in tick_plot]
        # For error metrics show the axis increasing inward (highest raw = inner ring)
        tick_labels = [f"{v:.1f}" if (max(display_scores) - min(display_scores)) > 5
                       else f"{v:.2f}" for v in tick_raw]
    else:
        tick_labels = [f"{v:.2f}" if (r_max - r_min) < 5 else f"{v:.1f}"
                       for v in tick_plot]

    # ── Polar axis cosmetics ────────────────────────────────
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(models, fontsize=8.5, color="#333333",
                       fontfamily="DejaVu Sans")

    ax.set_ylim(r_min, r_max)
    ax.set_yticks(tick_plot)
    ax.set_yticklabels(tick_labels, fontsize=6.5, color="#999999")
    ax.yaxis.set_tick_params(labelsize=6.5)

    ax.grid(color="#cccccc", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.spines["polar"].set_visible(False)

    # ── Filled polygon (plotted on inverted scale) ──────────
    ax.fill(angles_closed, plot_closed, color=fill_color, alpha=0.25, zorder=2)
    ax.plot(angles_closed, plot_closed, color=fill_color, linewidth=2.0,
            linestyle="solid", zorder=3)
    ax.scatter(angles, plot_scores, s=30, color=fill_color,
               edgecolors="white", linewidths=0.8, zorder=4)

    # ── Subtitle note for inverted panels ───────────────────
    note = "  ↓ lower is better (axis inverted)" if is_error else ""
    ax.set_title(
        metric_name + note,
        size=10,
        fontweight="bold",
        color="#1a252f",
        pad=18,
    )


def plot_model_radar(metrics_per_model: dict, output_dir="outputs"):
    """
    Per-metric polygon radar panel.

    One subplot per performance criterion; each subplot auto-scales its
    radial axis to the actual data range for that metric, so raw RMSE
    (e.g. 15-80) renders just as well as NSE/KGE/R² (0-1 range).

    Parameters
    ----------
    metrics_per_model : dict
        Nested dict  { model_name: { metric: value, ... } }
        Pass RAW values — no normalisation needed.
        Higher = better is assumed for NSE / KGE / R².
        For RMSE and PBIAS pass the raw values as-is; the panel will
        auto-scale and the filled polygon will still show relative differences.

    Example
    -------
    metrics_per_model = {
        "RF-PSO":   {"NSE": 0.89, "KGE": 0.85, "R²": 0.91,
                     "PBIAS": 3.2, "RMSE": 18.4},
        "XGBoost":  {"NSE": 0.83, "KGE": 0.80, "R²": 0.85,
                     "PBIAS": 5.1, "RMSE": 22.7},
        "LSTM":     {"NSE": 0.76, "KGE": 0.72, "R²": 0.79,
                     "PBIAS": 7.3, "RMSE": 28.1},
    }
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if not metrics_per_model:
        print("  ⚠ plot_model_radar: metrics_per_model is empty — skipping.")
        return

    model_names = list(metrics_per_model.keys())
    categories  = list(next(iter(metrics_per_model.values())).keys())
    n_metrics   = len(categories)

    if n_metrics < 1:
        print("  ⚠ plot_model_radar: no metrics found — skipping.")
        return

    # Distinct colour per metric (one filled polygon per panel)
    metric_palette = sns.color_palette("tab10", n_metrics)

    # ── Grid layout: up to 3 columns ────────────────────────
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))

    fig = plt.figure(figsize=(5.8 * ncols, 5.6 * nrows + 0.9))
    fig.patch.set_facecolor("#ffffff")

    fig.suptitle(
        "Model Performance — Per-Metric Radar Panels",
        fontsize=15, fontweight="bold", color="#1a252f", y=0.99,
    )

    # ── One polar subplot per metric ────────────────────────
    for idx, (metric, color) in enumerate(zip(categories, metric_palette)):
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)
        ax.set_facecolor("#ffffff")

        # Raw values for this metric across all models
        model_scores = {m: float(metrics_per_model[m].get(metric, 0.0))
                        for m in model_names}

        _draw_single_radar(ax, model_scores, metric,
                           fill_color=matplotlib.colors.to_hex(color))

    # ── One legend entry per metric (coloured dot + name) ───
    legend_handles = [
        mpatches.Patch(
            facecolor=matplotlib.colors.to_hex(metric_palette[i]),
            alpha=0.6,
            label=cat,
        )
        for i, cat in enumerate(categories)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(n_metrics, 6),
        fontsize=9,
        title="Metrics",
        title_fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_path = Path(output_dir) / "model_performance_radar.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Per-metric radar panel saved to {save_path}")


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
    ax2.bar(dates, precip, color='#7F8C8D', alpha=0.5, width=1.0, label='Precipitation')
    ax2.set_ylabel('Rainfall (mm)', color='#7F8C8D', fontweight='bold')
    ax2.set_ylim(bottom=0, top=precip.max() * 3)
    ax2.invert_yaxis()
    ax2.tick_params(axis='y', labelcolor='#7F8C8D')

    # ─── X-Axis Date Formatting ───
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title('Baseflow Separation (Eckhardt Filter) with Rainfall Hyetograph',
              pad=15, fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = out_dir / "eckhardt_separation.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


# ════════════════════════════════════════════════════════════
#  FLOW DURATION CURVE
# ════════════════════════════════════════════════════════════
def plot_flow_duration_curve(discharge, output_dir="outputs"):
    """
    Plots a clean Flow Duration Curve on a log scale with Q50 and Q90 annotations.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_flow = np.sort(discharge)[::-1]
    ranks = np.arange(1, len(sorted_flow) + 1)
    prob = (ranks / (len(sorted_flow) + 1)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prob, sorted_flow, color='#3498db', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlim(0, 100)

    idx_50 = np.argmin(np.abs(prob - 50))
    idx_90 = np.argmin(np.abs(prob - 90))
    val_50 = sorted_flow[idx_50]
    val_90 = sorted_flow[idx_90]

    ax.axhline(y=val_50, color='gray', linestyle='--', alpha=0.6)
    ax.axhline(y=val_90, color='gray', linestyle='--', alpha=0.6)
    ax.annotate(f'Q50 = {val_50:.1f}',
                xy=(50, val_50), xytext=(55, val_50 * 1.5),
                arrowprops=dict(arrowstyle="->", color='gray'))
    ax.annotate(f'Q90 = {val_90:.1f}',
                xy=(90, val_90), xytext=(75, val_90 * 0.5),
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
def plot_flood_frequency(annual_max, flood_freq_dict, output_dir="outputs"):
    """
    Plots the empirical annual maxima against the theoretical Gumbel distribution.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_max = np.sort(annual_max)[::-1]
    n = len(sorted_max)
    ranks = np.arange(1, n + 1)
    return_periods = (n + 1) / ranks

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(return_periods, sorted_max, color='#e74c3c', s=40,
               label='Observed Annual Maxima', zorder=5)

    gumbel_t = []
    gumbel_q = []
    for k, v in flood_freq_dict.items():
        try:
            t = int(k.replace('Q', '').replace('yr', ''))
            gumbel_t.append(t)
            gumbel_q.append(v)
        except ValueError:
            pass

    if gumbel_t and gumbel_q:
        sort_idx = np.argsort(gumbel_t)
        gumbel_t = np.array(gumbel_t)[sort_idx]
        gumbel_q = np.array(gumbel_q)[sort_idx]
        ax.plot(gumbel_t, gumbel_q, color='#2c3e50', linestyle='--',
                linewidth=2, label='Gumbel Theoretical Curve')
        for t, q in zip(gumbel_t, gumbel_q):
            ax.annotate(f"{q:.0f}", (t, q), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9, color='#2c3e50')

    ax.set_xscale('log')
    ax.set_xlabel('Return Period (Years, Log Scale)', fontweight='bold')
    ax.set_ylabel('Peak Discharge (m³/s)', fontweight='bold')
    plt.title('Flood Frequency Analysis (Gumbel Distribution)',
              pad=15, fontweight='bold', fontsize=14)

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
    fig, axes = plt.subplots(1, n_ssps, figsize=(7 * n_ssps, 5),
                             sharey=True, squeeze=False)

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

    plt.suptitle(
        f"CMIP6 Projected Discharge Change (baseline = {baseline_mean:.1f} cumecs)",
        y=1.02)
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
        means = [ssp_data[ssp_data["Period"] == p]["Ensemble_Mean_Q"].values[0]
                 if p in ssp_data["Period"].values else 0 for p in periods]
        stds  = [ssp_data[ssp_data["Period"] == p]["Ensemble_Std_Q"].values[0]
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


# ════════════════════════════════════════════════════════════
#  TOP-5 FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
def plot_top_5_features(model, X_val, y_val, feature_names, model_name="Model",
                        output_dir="outputs"):
    """
    Calculates and plots the Top 5 most important features.
    Uses native feature_importances_ for trees, permutation importance otherwise.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"  Calculating permutation importance for {model_name}...")
        result = permutation_importance(
            model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean

    indices = np.argsort(importances)[-5:]
    top_5_importances = importances[indices]
    top_5_features    = [feature_names[i] for i in indices]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_5_importances, y=top_5_features, palette="viridis")
    plt.title(f"Top 5 Feature Importances ({model_name})", pad=15, fontweight='bold')
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Hydrological Features")
    plt.tight_layout()

    save_path = Path(output_dir) / f"{model_name}_top5_features.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ Top 5 feature graph saved to {save_path}")


# ════════════════════════════════════════════════════════════
#  CMIP6 DETAILED DAILY FORECAST (DISCHARGE + RAINFALL)
# ════════════════════════════════════════════════════════════
def plot_cmip6_timeseries_with_rainfall(all_projections, output_dir="outputs"):
    """
    Dual-axis plot per CMIP6 scenario: discharge line + inverted rainfall bars.
    """
    out_dir = Path(output_dir) / "cmip6_detailed_forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, data in all_projections.items():
        dates     = data["dates"]
        discharge = data["discharge"]

        if "precip" in data["features"].columns:
            precip = data["features"]["precip"]
        else:
            print(f"  ⚠ Skipping rainfall plot for {key}: 'precip' feature not found.")
            continue

        fig, ax1 = plt.subplots(figsize=(14, 6))

        color_discharge = '#8e44ad' if 'ssp245' in key else '#c0392b'
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Projected Discharge (m³/s)', color=color_discharge, fontweight='bold')
        ax1.plot(dates, discharge, color=color_discharge, alpha=0.85,
                 linewidth=1.2, label='Discharge')
        ax1.tick_params(axis='y', labelcolor=color_discharge)
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        color_precip = '#3498db'
        ax2.set_ylabel('Projected Rainfall (mm)', color=color_precip, fontweight='bold')
        ax2.bar(dates, precip, color=color_precip, alpha=0.4, width=1.0, label='Rainfall')
        ax2.tick_params(axis='y', labelcolor=color_precip)
        ax2.set_ylim(bottom=0, top=max(precip) * 3)
        ax2.invert_yaxis()

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        parts = key.split('_')
        gcm, ssp = parts[0], parts[-3] if len(parts) > 3 else "Unknown"
        plt.title(f"Future Discharge vs Rainfall Projection: {gcm} ({ssp.upper()})",
                  pad=15, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        fig.tight_layout()

        save_path = out_dir / f"{key}_forecast_plot.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"  ✓ Detailed forecast plots saved to {out_dir}/")


# ════════════════════════════════════════════════════════════
#  CMIP6 YEARLY AGGREGATED TIMESERIES
# ════════════════════════════════════════════════════════════
def plot_cmip6_yearly_timeseries(all_projections, output_dir="outputs"):
    """
    Annual mean discharge and annual total rainfall for CMIP6 scenarios.
    """
    out_dir = Path(output_dir) / "cmip6_yearly_forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, data in all_projections.items():
        if "precip" not in data["features"].columns:
            continue

        df = pd.DataFrame({
            "discharge": data["discharge"],
            "precip":    data["features"]["precip"].values
        }, index=pd.to_datetime(data["dates"]))

        annual_df = df.resample('YE').agg({'discharge': 'mean', 'precip': 'sum'})
        annual_df.index = annual_df.index.year

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color_discharge = '#8e44ad' if 'ssp245' in key else '#c0392b'
        ax1.set_xlabel('Year', fontweight='bold')
        ax1.set_ylabel('Annual Mean Discharge (m³/s)', color=color_discharge,
                       fontweight='bold')
        ax1.plot(annual_df.index, annual_df['discharge'], marker='o',
                 color=color_discharge, linewidth=2, markersize=6, label='Discharge')
        ax1.tick_params(axis='y', labelcolor=color_discharge)
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        color_precip = '#3498db'
        ax2.set_ylabel('Total Annual Rainfall (mm)', color=color_precip, fontweight='bold')
        ax2.bar(annual_df.index, annual_df['precip'], color=color_precip,
                alpha=0.3, width=0.6, label='Rainfall')
        ax2.tick_params(axis='y', labelcolor=color_precip)
        ax2.set_ylim(bottom=0, top=annual_df['precip'].max() * 3)
        ax2.invert_yaxis()

        parts = key.split('_')
        gcm, ssp = parts[0], parts[-3] if len(parts) > 3 else "Unknown"
        plt.title(f"Yearly Trend: Discharge vs Total Rainfall - {gcm} ({ssp.upper()})",
                  pad=15, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()

        save_path = out_dir / f"{key}_yearly_trend.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"  ✓ Yearly aggregated forecast plots saved to {out_dir}/")


# ════════════════════════════════════════════════════════════
#  CMIP6 ANNUAL REGIME HYDROGRAPH
# ════════════════════════════════════════════════════════════
def plot_cmip6_annual_regime(all_projections, output_dir="outputs"):
    """
    Long-term monthly climatology hydrograph (all scenarios overlaid).
    Averages all ~30 years of daily predictions into one typical year.
    """
    out_dir = Path(output_dir)
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(all_projections))

    for (key, data), color in zip(all_projections.items(), colors):
        df = pd.DataFrame({"discharge": data["discharge"]},
                          index=pd.to_datetime(data["dates"]))
        df['month'] = df.index.month
        monthly_regime = df.groupby('month')['discharge'].mean()

        parts = key.split('_')
        label = f"{parts[0]} ({parts[-3]})" if len(parts) > 3 else key
        plt.plot(monthly_regime.index, monthly_regime.values,
                 marker='s', linewidth=2, color=color, label=label)

    plt.title("Long-Term Average Annual Regime Hydrograph (2020–2050)",
              pad=15, fontweight='bold', fontsize=14)
    plt.xlabel("Month", fontweight='bold')
    plt.ylabel("Average Monthly Discharge (m³/s)", fontweight='bold')
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    plt.xticks(ticks=range(1, 13), labels=months)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = out_dir / "cmip6_regime_hydrograph_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ CMIP6 regime hydrograph (climatology) saved to {out_dir}/")