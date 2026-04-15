"""
FlowCast v2 — Hydrology Module
Provides importable hydrological functions used by main.py, including
advanced catchment analysis (Snyder UH, SCS-CN, Water Balance, etc.).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def eckhardt_baseflow(discharge: np.ndarray, a: float = 0.975, bfi_max: float = 0.80):
    """Eckhardt (2005) recursive digital baseflow filter."""
    Q = np.asarray(discharge, dtype=float)
    b = np.zeros_like(Q)
    b[0] = min(Q[0], Q[0] * bfi_max)
    for t in range(1, len(Q)):
        b[t] = ((1 - bfi_max) * a * b[t - 1] + (1 - a) * bfi_max * Q[t]) / (1 - a * bfi_max)
        b[t] = min(b[t], Q[t])
    quickflow = np.maximum(Q - b, 0.0)
    return b, quickflow

def gumbel_flood_frequency(annual_max: np.ndarray) -> dict:
    y_mean = np.mean(annual_max)
    y_std  = np.std(annual_max, ddof=1)
    alpha  = y_std * np.sqrt(6) / np.pi
    mu     = y_mean - 0.5772 * alpha

    result = {}
    for T in [2, 5, 10, 25, 50, 100]:
        yT = -np.log(-np.log(1 - 1 / T))
        result[f"Q{T}yr"] = round(float(mu + alpha * yT), 2)
    return result

def compute_hydrological_signatures(discharge: np.ndarray, precip: np.ndarray) -> dict:
    Q = np.asarray(discharge, dtype=float)
    P = np.asarray(precip, dtype=float)

    Q_sorted   = np.sort(Q)[::-1]
    exceedance = np.arange(1, len(Q_sorted) + 1) / len(Q_sorted) * 100

    Q10 = float(np.interp(10, exceedance, Q_sorted))
    Q50 = float(np.interp(50, exceedance, Q_sorted))
    Q90 = float(np.interp(90, exceedance, Q_sorted))
    Q95 = float(np.interp(95, exceedance, Q_sorted))

    runoff_ratio = float(np.sum(Q) / (np.sum(P) + 1e-10))
    base, _ = eckhardt_baseflow(Q)
    bfi = float(np.sum(base) / (np.sum(Q) + 1e-10))
    flashiness = float(np.sum(np.abs(np.diff(Q))) / (np.sum(Q) + 1e-10))

    return {
        "mean_discharge_m3s": round(float(np.mean(Q)), 3),
        "Q10_high_flow":      round(Q10, 3),
        "Q50_median_flow":    round(Q50, 3),
        "Q90_low_flow":       round(Q90, 3),
        "Q95_drought_flow":   round(Q95, 3),
        "runoff_ratio":       round(runoff_ratio, 4),
        "baseflow_index":     round(bfi, 4),
        "flashiness_index":   round(flashiness, 5),
    }

def flow_duration_curve(discharge: np.ndarray):
    Q = np.asarray(discharge, dtype=float)
    Q_sorted   = np.sort(Q)[::-1]
    exceedance = np.arange(1, len(Q_sorted) + 1) / len(Q_sorted) * 100
    return exceedance, Q_sorted


# ─────────────────────────────────────────────────────────────
#  ADVANCED CATCHMENT ANALYSIS (Integrates former standalone script)
# ─────────────────────────────────────────────────────────────

def run_advanced_catchment_analysis(df_in: pd.DataFrame, output_dir: str, area_km2: float):
    """
    Executes Unit Hydrograph, Water Balance, SCS-CN, and Recession analysis.
    """
    from scipy.special import gamma
    from scipy.integrate import trapezoid
    import warnings
    warnings.filterwarnings("ignore")

    print("\n" + "=" * 60)
    print("  ADVANCED HYDROLOGY ANALYSIS")
    print("=" * 60)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Prepare DataFrame
    df = df_in.copy()
    if 'date' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index': 'date'}, inplace=True)
            
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['P'] = df['precip']
    df['Q_total'] = df['discharge']
    
    # Recalculate Eckhardt to get dynamic recession constant
    Q_total = df['Q_total'].values
    dQ = np.diff(Q_total)
    recession_mask = dQ < 0
    ratios = Q_total[1:][recession_mask] / (Q_total[:-1][recession_mask] + 1e-6)
    ratios = ratios[(ratios > 0.5) & (ratios < 1.0)]
    recession_constant = np.median(ratios) if len(ratios) > 0 else 0.975

    b, qf = eckhardt_baseflow(Q_total, a=recession_constant, bfi_max=0.80)
    df['Q_base'] = b
    df['Q_direct'] = qf
    BFI = df['Q_base'].sum() / (df['Q_total'].sum() + 1e-6)
    
    CATCHMENT_AREA_M2 = area_km2 * 1e6

    # --- 1. RECESSION ANALYSIS ---
    min_length = 5
    segments, current = [], []
    for i in range(1, len(Q_total)):
        if Q_total[i] < Q_total[i-1]:
            current.append(i)
        else:
            if len(current) >= min_length:
                segments.append(current.copy())
            current = [i]

    k_values = []
    for seg in segments:
        q_seg = Q_total[seg]
        if q_seg.min() <= 0: continue
        try:
            slope, _ = np.polyfit(np.arange(len(q_seg)), np.log(q_seg), 1)
            k = np.exp(slope)
            if 0.5 < k < 1.0: k_values.append(k)
        except: pass

    k_median = np.median(k_values) if k_values else recession_constant

    fig, ax = plt.subplots(figsize=(8, 5))
    for seg in segments[:30]:
        if Q_total[seg].min() > 0:
            ax.semilogy(np.arange(len(seg)), Q_total[seg], color='#3B8BD4', alpha=0.25)
    t_fit = np.arange(0, 60)
    ax.semilogy(t_fit, np.median(Q_total) * k_median**t_fit, 'r--', lw=2, label=f'Theoretical (k={k_median:.3f})')
    ax.set_title("Master Recession Curve")
    ax.set_ylabel("Discharge (m³/s, log scale)")
    ax.set_xlim(0, 50)
    ax.legend()
    plt.savefig(out_dir / "recession_curve.png", dpi=150)
    plt.close()

    # --- 2. SYNTHETIC UNIT HYDROGRAPH (Snyder's) ---
    L = 75.0; Lc = 42.0; Ct = 2.0; Cp = 0.65
    tp_days = (Ct * (L * Lc)**0.3) / 24
    Qp_total_uh = ((0.275 * Cp) / tp_days) * area_km2
    Tb_days = 3 + 3 * (tp_days / 24)

    t_uh = np.linspace(0, Tb_days, 100)
    a_shape = 3.5 
    uh_values = Qp_total_uh * (t_uh/tp_days)**a_shape * np.exp(a_shape * (1 - t_uh/tp_days))
    current_vol = trapezoid(uh_values, t_uh) * 86400 / CATCHMENT_AREA_M2 * 1000
    uh_values = uh_values / current_vol # Normalize to 1mm

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_uh, uh_values, color='#E8593C', linewidth=3, label='Synthetic UH')
    ax.fill_between(t_uh, 0, uh_values, color='#E8593C', alpha=0.1)
    ax.axvline(tp_days, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Synthetic Unit Hydrograph (Snyder's Method)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Discharge (m³/s) per 1mm Excess Rain")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(out_dir / "snyder_unit_hydrograph.png", dpi=150)
    plt.close()

    # --- 3. SCS CURVE NUMBER ESTIMATION ---
    df['Q_direct_mm'] = df['Q_direct'] * 86400 / CATCHMENT_AREA_M2 * 1000
    event_P, event_Q = [], []
    for i in range(0, len(df) - 3, 3):
        p_sum = df['P'].iloc[i:i+3].sum()
        q_sum = df['Q_direct_mm'].iloc[i:i+3].sum()
        if p_sum > 10 and q_sum > 0:
            event_P.append(p_sum); event_Q.append(q_sum)

    event_P, event_Q = np.array(event_P), np.array(event_Q)
    cn_values = []
    for p, q in zip(event_P, event_Q):
        if q >= p or q <= 0: continue
        for S_try in np.arange(1, 500, 0.5):
            Ia = 0.2 * S_try
            if p <= Ia: continue
            if abs(((p - Ia)**2 / (p - Ia + S_try)) - q) < 0.5:
                cn = 25400 / (S_try + 254)
                if 30 < cn < 100: cn_values.append(cn)
                break

    CN_median = np.median(cn_values) if cn_values else None

    # --- 4. WATER BALANCE ---
    df['Q_total_mm'] = df['Q_total'] * 86400 / CATCHMENT_AREA_M2 * 1000
    annual = df.groupby('year').agg(
        P_total=('P', 'sum'),
        Q_total=('Q_total_mm', 'sum')
    ).reset_index()
    annual['C_runoff'] = (annual['Q_total'] / annual['P_total']).clip(0, 1)
    C_avg = annual['C_runoff'].mean()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    w = 0.35
    ax1.bar(annual['year'] - w/2, annual['P_total'], w, label='Rainfall', color='#3B8BD4')
    ax1.bar(annual['year'] + w/2, annual['Q_total'], w, label='Runoff', color='#E8593C')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Depth (mm)")
    ax1.set_title("Annual Water Balance")
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(annual['year'], annual['C_runoff'], 'ko-', label='Runoff Coefficient')
    ax2.set_ylabel("Runoff Coefficient (C)")
    ax2.set_ylim(0, 1)
    plt.savefig(out_dir / "annual_water_balance.png", dpi=150)
    plt.close()

    # --- 5. PARAMETER EXPORT ---
    params = {
        'Catchment area (km²)': area_km2,
        'Mean annual rainfall (mm)': annual['P_total'].mean(),
        'Mean annual runoff (mm)': annual['Q_total'].mean(),
        'Runoff coefficient (C)': C_avg,
        'Baseflow Index (BFI)': BFI,
        'Recession constant (k)': k_median,
        'Recession half-life (days)': -np.log(2)/np.log(k_median) if k_median else None,
        'SCS Curve Number (CN)': CN_median if CN_median else 'N/A',
        'UH Time to peak Tp (days)': tp_days,
        'UH Peak Qp (m³/s/mm)': Qp_total_uh,
    }
    pd.DataFrame([{'Parameter': k, 'Value': v} for k, v in params.items()]).to_csv(out_dir / "hydrology_parameters.csv", index=False)
    
    print(f"  ✓ Advanced hydrology analyses saved to {out_dir}/")