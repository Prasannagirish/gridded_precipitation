"""
FlowCast v2 — Main Pipeline
Orchestrates: Data → Features → Model Training → Evaluation → Climate Projections
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Enable KML support for geopandas (prevents shapefile read errors downstream)
import fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.config import data_cfg, model_cfg, climate_cfg, basin_cfg
from modules.loader import KabiniDataLoader
from modules.engineer import (
    HydroFeatureEngineer, select_features,
    create_sequences, create_sequences_with_static,
)
from modules.traditional_ml import RFModel, SVRModel, XGBoostModel
from modules.deep_learning import (
    LSTMNetwork, PhysicsInformedLSTM, PhysicsLoss, AdaptedGRU,
    DeepModelTrainer, build_dataloader, build_agru_dataloader,
)
from modules.ensemble import WeightedEnsemble, StackingEnsemble
from modules.metrics import evaluate_all, evaluate_flow_regimes
from modules.cmip6_projector import CMIP6RealProjector as CMIP6Projector    
from modules.hydro import (
    eckhardt_baseflow, gumbel_flood_frequency,
    compute_hydrological_signatures, flow_duration_curve,
    run_advanced_catchment_analysis 
)
from modules import visualization as vis


def run_pipeline(
    use_synthetic: bool = False,
    precip_path: str = "/Users/girish/cs101/projects/gridded_precip/gridded_precipitation/data/chirps_kabini_daily.csv",
    discharge_path: str = "/Users/girish/cs101/projects/gridded_precip/gridded_precipitation/data/discharge_daily_observed.csv",
    temp_path: str = None,
    feature_combo: str = "M1_full",
    run_traditional: bool = True,
    run_deep: bool = True,
    run_climate: bool = True,
    pso_particles: int = 15,
    pso_iterations: int = 30,
    dl_epochs: int = 80,
    seq_length: int = 30,
):
    """
    Execute the complete FlowCast v2 pipeline.
    """
    results = {"models": {}, "ensemble": {}, "climate": {}}
    start_time = time.time()

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 1: DATA LOADING")
    print("═" * 70)

    loader = KabiniDataLoader()
    if use_synthetic:
        raw_df = loader.generate_synthetic_data()
        print(f"  Generated synthetic data: {raw_df.shape[0]} days")
    else:
        # --- CUSTOM DATA LOADER FOR SPECIFIC CSV FORMATS ---
        print("  Parsing custom CSV formats...")
        
        # 1. Read Precipitation (using rainfall_mean_mm)
        df_p = pd.read_csv(precip_path)
        df_p['date'] = pd.to_datetime(df_p['date'])
        df_p = df_p.set_index('date')
        df_p = df_p[['rainfall_mean_mm']].rename(columns={'rainfall_mean_mm': 'precip'})
        
        # 2. Read Discharge (using q_upstream_mk)
        df_q = pd.read_csv(discharge_path)
        df_q['date'] = pd.to_datetime(df_q['date'])
        df_q = df_q.set_index('date')
        df_q = df_q[['q_upstream_mk']].rename(columns={'q_upstream_mk': 'discharge'})

        # 3. Merge observations together
        raw_df = df_p.join(df_q, how='inner').sort_index()

        # 4. INJECT REALISTIC TEMPERATURES FROM CMIP6 MIROC6
        print("  Injecting temperature proxies from MIROC6 GCM...")
        try:
            # Paths to the MIROC6 data you just downloaded
            miroc_hist = Path(climate_cfg.cmip6_download_dir) /"historical"/ "MIROC6_historical_daily.csv"
            miroc_fut  = Path(climate_cfg.cmip6_download_dir) /"ssp245"/"MIROC6_ssp245_daily.csv"

            # Read tmax and tmin (using index_col=0 as we figured out earlier!)
            df_t_hist = pd.read_csv(miroc_hist, index_col=0, parse_dates=True)[['tmax', 'tmin']]
            df_t_fut  = pd.read_csv(miroc_fut, index_col=0, parse_dates=True)[['tmax', 'tmin']]
            
            # Combine 1990-2014 (historical) and 2015-2020 (SSP245) to cover your full training period
            df_t_combined = pd.concat([df_t_hist, df_t_fut])
            df_t_combined = df_t_combined[~df_t_combined.index.duplicated(keep='first')]
            
            # Join the temperatures onto your raw observations
            raw_df = raw_df.join(df_t_combined, how='left')
            
            # Fill any missing days (like leap days if the GCM uses a 365-day calendar)
            raw_df['tmax'] = raw_df['tmax'].ffill().bfill()
            raw_df['tmin'] = raw_df['tmin'].ffill().bfill()
            
        except Exception as e:
            print(f"  ⚠ Could not load MIROC6 temperatures: {e}")
            print("  Falling back to seasonal sine-wave approximation.")
            # Fallback if files aren't found: A sine wave mimicking the Kabini basin climate
            doy = raw_df.index.dayofyear
            raw_df["tmax"] = 30.0 + 5.0 * np.sin(2 * np.pi * (doy - 100) / 365)
            raw_df["tmin"] = 20.0 + 3.0 * np.sin(2 * np.pi * (doy - 100) / 365)

        # Final cleanup for any lingering NaNs
        raw_df = raw_df.ffill().bfill() 
        print(f"  Loaded real data: {raw_df.shape[0]} days "
              f"({raw_df.index[0].date()} to {raw_df.index[-1].date()})")

    static_features = loader.get_static_features()
    static_array = np.array(list(static_features.values()), dtype=np.float32)
    print(f"  Static features: {len(static_features)} attributes")

    # Hydrological signatures
    signatures = compute_hydrological_signatures(
        raw_df["discharge"].values, raw_df["precip"].values
    )
    print(f"\n  Hydrological Signatures:")
    for k, v in signatures.items():
        print(f"    {k}: {v}")

    # Baseflow separation
    baseflow, quickflow = eckhardt_baseflow(raw_df["discharge"].values)
    raw_df["baseflow"] = baseflow
    raw_df["quickflow"] = quickflow

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Feature Engineering
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 2: FEATURE ENGINEERING")
    print("═" * 70)

    engineer = HydroFeatureEngineer(raw_df)
    df = engineer.build_all_features()
    
    feature_list = data_cfg.feature_combinations.get(feature_combo, data_cfg.feature_combinations["M1_full"])
    # Drop rows with NaNs resulting from lags/rolling windows before selecting features
    df = df.dropna()
    df_features = select_features(df, feature_list)
    target = df["discharge"]

    print(f"  Selected combo: {feature_combo} ({df_features.shape[1]} features)")

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Train/Val/Test Split
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 3: TEMPORAL SPLIT")
    print("═" * 70)

    train_end = data_cfg.train_end
    val_end = data_cfg.val_end

    train_X = df_features.loc[:train_end]
    val_X = df_features.loc[train_end:val_end].iloc[1:]
    test_X = df_features.loc[val_end:].iloc[1:]

    train_y = target.loc[:train_end]
    val_y = target.loc[train_end:val_end].iloc[1:]
    test_y = target.loc[val_end:].iloc[1:]

    print(f"  Train: {len(train_X)} samples ({train_X.index[0].date()} to {train_X.index[-1].date()})")
    print(f"  Val:   {len(val_X)} samples ({val_X.index[0].date()} to {val_X.index[-1].date()})")
    print(f"  Test:  {len(test_X)} samples ({test_X.index[0].date()} to {test_X.index[-1].date()})")

    X_train_np = train_X.values.astype(np.float32)
    X_val_np = val_X.values.astype(np.float32)
    X_test_np = test_X.values.astype(np.float32)
    y_train_np = train_y.values.astype(np.float32)
    y_val_np = val_y.values.astype(np.float32)
    y_test_np = test_y.values.astype(np.float32)

    all_val_preds = {}
    all_test_preds = {}

    # ═══════════════════════════════════════════════════════════
    # STEP 4A: Traditional ML Models (PSO-optimized)
    # ═══════════════════════════════════════════════════════════
    trained_model_objs = {} # Initialize tracking dictionary
    
    if run_traditional:
        print("\n" + "═" * 70)
        print("  STEP 4A: TRADITIONAL ML MODELS (PSO-OPTIMIZED)")
        print("═" * 70)

        # --- RF-PSO ---
        rf = RFModel()
        rf_result = rf.optimize_and_fit(
            X_train_np, y_train_np,
            n_particles=pso_particles, n_iterations=pso_iterations,
        )
        rf_val_pred = rf.predict(X_val_np)
        rf_test_pred = rf.predict(X_test_np)
        results["models"]["RF-PSO"] = {
            "val": evaluate_all(y_val_np, rf_val_pred),
            "test": evaluate_all(y_test_np, rf_test_pred),
            "best_params": rf_result["params"],
        }
        all_val_preds["RF-PSO"] = rf_val_pred
        all_test_preds["RF-PSO"] = rf_test_pred
        trained_model_objs["RF-PSO"] = rf # Save model
        print(f"\n  RF-PSO Test: {results['models']['RF-PSO']['test']}")

        # --- SVR-PSO ---
        svr = SVRModel()
        svr_result = svr.optimize_and_fit(
            X_train_np, y_train_np,
            n_particles=pso_particles, n_iterations=pso_iterations,
        )
        svr_val_pred = svr.predict(X_val_np)
        svr_test_pred = svr.predict(X_test_np)
        results["models"]["SVR-PSO"] = {
            "val": evaluate_all(y_val_np, svr_val_pred),
            "test": evaluate_all(y_test_np, svr_test_pred),
        }
        all_val_preds["SVR-PSO"] = svr_val_pred
        all_test_preds["SVR-PSO"] = svr_test_pred
        trained_model_objs["SVR-PSO"] = svr # Save model
        print(f"\n  SVR-PSO Test: {results['models']['SVR-PSO']['test']}")

        # --- XGBoost-PSO ---
        xgb = XGBoostModel()
        xgb_result = xgb.optimize_and_fit(
            X_train_np, y_train_np,
            n_particles=pso_particles, n_iterations=pso_iterations,
        )
        xgb_val_pred = xgb.predict(X_val_np)
        xgb_test_pred = xgb.predict(X_test_np)
        results["models"]["XGBoost-PSO"] = {
            "val": evaluate_all(y_val_np, xgb_val_pred),
            "test": evaluate_all(y_test_np, xgb_test_pred),
        }
        all_val_preds["XGBoost-PSO"] = xgb_val_pred
        all_test_preds["XGBoost-PSO"] = xgb_test_pred
        trained_model_objs["XGBoost-PSO"] = xgb # Save model
        print(f"\n  XGBoost-PSO Test: {results['models']['XGBoost-PSO']['test']}")

    # ═══════════════════════════════════════════════════════════
    # STEP 4B: Deep Learning Models
    # ═══════════════════════════════════════════════════════════
    if run_deep:
        print("\n" + "═" * 70)
        print("  STEP 4B: DEEP LEARNING MODELS")
        print("═" * 70)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_s = scaler_X.fit_transform(X_train_np)
        X_val_s = scaler_X.transform(X_val_np)
        X_test_s = scaler_X.transform(X_test_np)
        y_train_s = scaler_y.fit_transform(y_train_np.reshape(-1, 1)).ravel()
        y_val_s = scaler_y.transform(y_val_np.reshape(-1, 1)).ravel()
        y_test_s = scaler_y.transform(y_test_np.reshape(-1, 1)).ravel()

        n_features = X_train_s.shape[1]

        Xtr_seq, ytr_seq = create_sequences(X_train_s, y_train_s, seq_length)
        Xv_seq, yv_seq = create_sequences(X_val_s, y_val_s, seq_length)
        Xt_seq, yt_seq = create_sequences(X_test_s, y_test_s, seq_length)

        # --- Standard LSTM ---
        print("\n  Training Standard LSTM...")
        lstm_model = LSTMNetwork(
            input_size=n_features,
            hidden_size=model_cfg.lstm_hidden_size,
            num_layers=model_cfg.lstm_num_layers,
            dropout=model_cfg.lstm_dropout,
        )
        lstm_trainer = DeepModelTrainer(
            lstm_model,
            learning_rate=model_cfg.lstm_learning_rate,
            patience=model_cfg.lstm_patience,
            model_name="LSTM",
        )
        train_loader = build_dataloader(Xtr_seq, ytr_seq, model_cfg.lstm_batch_size, shuffle=True)
        val_loader = build_dataloader(Xv_seq, yv_seq, model_cfg.lstm_batch_size, shuffle=False)
        test_loader = build_dataloader(Xt_seq, yt_seq, model_cfg.lstm_batch_size, shuffle=False)

        lstm_trainer.train(train_loader, val_loader, epochs=dl_epochs)

        lstm_val_pred_s = lstm_trainer.predict(val_loader)
        lstm_test_pred_s = lstm_trainer.predict(test_loader)
        lstm_val_pred = scaler_y.inverse_transform(lstm_val_pred_s.reshape(-1, 1)).ravel()
        lstm_test_pred = scaler_y.inverse_transform(lstm_test_pred_s.reshape(-1, 1)).ravel()

        y_val_seq_real = scaler_y.inverse_transform(yv_seq.reshape(-1, 1)).ravel()
        y_test_seq_real = scaler_y.inverse_transform(yt_seq.reshape(-1, 1)).ravel()

        results["models"]["LSTM"] = {
            "val": evaluate_all(y_val_seq_real, lstm_val_pred),
            "test": evaluate_all(y_test_seq_real, lstm_test_pred),
        }
        all_val_preds["LSTM"] = lstm_val_pred
        all_test_preds["LSTM"] = lstm_test_pred
        print(f"\n  LSTM Test: {results['models']['LSTM']['test']}")

        # --- Physics-Informed LSTM ---
        print("\n  Training Physics-Informed LSTM...")
        pi_model = PhysicsInformedLSTM(
            input_size=n_features,
            hidden_size=model_cfg.lstm_hidden_size,
            num_layers=model_cfg.lstm_num_layers,
            dropout=model_cfg.lstm_dropout,
        )
        pi_trainer = DeepModelTrainer(
            pi_model,
            learning_rate=model_cfg.lstm_learning_rate,
            patience=model_cfg.lstm_patience,
            model_name="PI-LSTM",
        )
        physics_loss = PhysicsLoss(
            lambda_wb=model_cfg.pi_lambda_water_balance,
            lambda_mono=model_cfg.pi_lambda_monotonicity,
        )
        pi_trainer.train(
            train_loader, val_loader, epochs=dl_epochs,
            loss_fn=physics_loss, physics_mode=True, precip_idx=0,
        )

        pi_val_pred_s = pi_trainer.predict(val_loader)
        pi_test_pred_s = pi_trainer.predict(test_loader)
        pi_val_pred = scaler_y.inverse_transform(pi_val_pred_s.reshape(-1, 1)).ravel()
        pi_test_pred = scaler_y.inverse_transform(pi_test_pred_s.reshape(-1, 1)).ravel()

        results["models"]["PI-LSTM"] = {
            "val": evaluate_all(y_val_seq_real, pi_val_pred),
            "test": evaluate_all(y_test_seq_real, pi_test_pred),
        }
        all_val_preds["PI-LSTM"] = pi_val_pred
        all_test_preds["PI-LSTM"] = pi_test_pred
        print(f"\n  PI-LSTM Test: {results['models']['PI-LSTM']['test']}")

        # --- Adapted GRU ---
        print("\n  Training A-GRU...")
        static_scaler = StandardScaler()
        static_scaled = static_scaler.fit_transform(static_array.reshape(1, -1)).flatten()
        n_static = len(static_scaled)

        Xtr_d, Xtr_s, ytr_agru = create_sequences_with_static(
            X_train_s, static_scaled, y_train_s, seq_length
        )
        Xv_d, Xv_s, yv_agru = create_sequences_with_static(
            X_val_s, static_scaled, y_val_s, seq_length
        )
        Xt_d, Xt_s, yt_agru = create_sequences_with_static(
            X_test_s, static_scaled, y_test_s, seq_length
        )

        agru_model = AdaptedGRU(
            dynamic_size=n_features,
            static_size=n_static,
            hidden_size=model_cfg.agru_hidden_size,
            dropout=model_cfg.agru_dropout,
        )
        agru_trainer = DeepModelTrainer(
            agru_model,
            learning_rate=model_cfg.agru_learning_rate,
            patience=model_cfg.agru_patience,
            model_name="A-GRU",
        )

        agru_train_loader = build_agru_dataloader(Xtr_d, Xtr_s, ytr_agru, model_cfg.agru_batch_size, shuffle=True)
        agru_val_loader = build_agru_dataloader(Xv_d, Xv_s, yv_agru, model_cfg.agru_batch_size, shuffle=False)
        agru_test_loader = build_agru_dataloader(Xt_d, Xt_s, yt_agru, model_cfg.agru_batch_size, shuffle=False)

        agru_trainer.train(agru_train_loader, agru_val_loader, epochs=dl_epochs)

        agru_val_pred_s = agru_trainer.predict(agru_val_loader)
        agru_test_pred_s = agru_trainer.predict(agru_test_loader)
        agru_val_pred = scaler_y.inverse_transform(agru_val_pred_s.reshape(-1, 1)).ravel()
        agru_test_pred = scaler_y.inverse_transform(agru_test_pred_s.reshape(-1, 1)).ravel()

        y_val_agru_real = scaler_y.inverse_transform(yv_agru.reshape(-1, 1)).ravel()
        y_test_agru_real = scaler_y.inverse_transform(yt_agru.reshape(-1, 1)).ravel()

        results["models"]["A-GRU"] = {
            "val": evaluate_all(y_val_agru_real, agru_val_pred),
            "test": evaluate_all(y_test_agru_real, agru_test_pred),
        }
        all_val_preds["A-GRU"] = agru_val_pred
        all_test_preds["A-GRU"] = agru_test_pred
        print(f"\n  A-GRU Test: {results['models']['A-GRU']['test']}")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Ensemble
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 5: ENSEMBLE METHODS")
    print("═" * 70)

    if len(all_val_preds) > 1:
        min_val_len = min(len(v) for v in all_val_preds.values())
        min_test_len = min(len(v) for v in all_test_preds.values())

        val_preds_trimmed = {k: v[:min_val_len] for k, v in all_val_preds.items()}
        test_preds_trimmed = {k: v[:min_test_len] for k, v in all_test_preds.items()}

        y_val_ens = y_val_np[-min_val_len:]
        y_test_ens = y_test_np[-min_test_len:]

        weighted_ens = WeightedEnsemble()
        weighted_ens.fit(val_preds_trimmed, y_val_ens, method="optimize")
        ens_val_pred = weighted_ens.predict(val_preds_trimmed)
        ens_test_pred = weighted_ens.predict(test_preds_trimmed)

        results["ensemble"]["weighted"] = {
            "val": evaluate_all(y_val_ens, ens_val_pred),
            "test": evaluate_all(y_test_ens, ens_test_pred),
            "weights": dict(zip(weighted_ens.model_names, weighted_ens.weights.tolist())),
        }
        print(f"\n  Weighted Ensemble Test: {results['ensemble']['weighted']['test']}")

        stacking = StackingEnsemble()
        stacking.fit(val_preds_trimmed, y_val_ens)
        stack_test_pred = stacking.predict(test_preds_trimmed)
        results["ensemble"]["stacking"] = {
            "test": evaluate_all(y_test_ens, stack_test_pred),
        }
        print(f"  Stacking Ensemble Test: {results['ensemble']['stacking']['test']}")

    # ═══════════════════════════════════════════════════════════
    # STEP 6: Flood Frequency Analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 6: FLOOD FREQUENCY ANALYSIS")
    print("═" * 70)

    annual_max = raw_df["discharge"].resample("YE").max().dropna().values
    if len(annual_max) >= 5:
        flood_freq = gumbel_flood_frequency(annual_max)
        results["flood_frequency"] = flood_freq
        print(f"  Gumbel flood frequencies: {flood_freq}")

    # ═══════════════════════════════════════════════════════════
    # FIND BEST TRADITIONAL MODEL (For Climate & Feature Importance)
    # ═══════════════════════════════════════════════════════════
    best_model = None
    best_model_name = None
    if run_traditional:
        best_nse = -np.inf
        for name in ["RF-PSO", "SVR-PSO", "XGBoost-PSO"]:
            if name in results["models"]:
                test_nse = results["models"][name]["test"]["NSE"]
                if test_nse > best_nse:
                    best_nse = test_nse
                    best_model_name = name
                    if name == "RF-PSO": best_model = rf
                    elif name == "SVR-PSO": best_model = svr
                    else: best_model = xgb

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Climate Projections
    # ═══════════════════════════════════════════════════════════
    all_projections = {}
    if run_climate:
        print("\n" + "═" * 70)
        print("  STEP 7: CMIP6 CLIMATE PROJECTIONS")
        print("═" * 70)

        try:
            projector = CMIP6Projector(
                cmip6_dir=climate_cfg.cmip6_download_dir,
                gcm_models=climate_cfg.gcm_models,
                obs_df=raw_df,
                feature_columns=list(df_features.columns),
                scenarios=climate_cfg.scenarios,
                future_periods=climate_cfg.future_periods,
            )

            if best_model is not None:
                print(f"\n  Using {best_model_name} for climate projections")
                all_projections = projector.generate_all_scenarios(best_model)

                if all_projections:
                    summary = projector.summarize_projections(
                        all_projections, raw_df["discharge"].values
                    )
                    results["climate"]["projections"] = summary.to_dict("records")
                    print(f"\n  Climate Projection Summary:")
                    print(summary.to_string(index=False))
                else:
                    print("  ⚠ No projections generated.")
        except Exception as e:
            print(f"  ⚠ Climate projection skipped: {e}")

    # ═══════════════════════════════════════════════════════════
    # STEP 8: FLOW REGIME ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 8: FLOW REGIME ANALYSIS")
    print("═" * 70)

    for model_name in list(results["models"].keys())[:3]:
        if model_name in all_test_preds:
            pred = all_test_preds[model_name]
            obs = y_test_np[-len(pred):]
            regime_results = evaluate_flow_regimes(obs, pred)
            results["models"][model_name]["flow_regimes"] = {
                k: v for k, v in regime_results.items()
            }

    # ═══════════════════════════════════════════════════════════
    # STEP 9: VISUALIZATIONS  
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 9: GENERATING VISUALIZATIONS")
    print("═" * 70)
    
    # We use the trimmed dates to ensure DL models and Traditional models align perfectly
    if len(all_test_preds) > 0:
        min_test_len = min(len(v) for v in all_test_preds.values())
        test_dates = test_X.index[-min_test_len:]
        y_test_ens = y_test_np[-min_test_len:]
        test_preds_trimmed = {k: v[-min_test_len:] for k, v in all_test_preds.items()}
        
        # ─── HYDROLOGICAL OUTPUTS ──────────────────────────────────
        hydro_out_dir = str(ROOT / "hydrological_outputs")

        # 1. Baseflow Separation Plot (Using full dataset)
        vis.plot_eckhardt_filter(
            raw_df.index, raw_df["precip"], raw_df["discharge"], raw_df["baseflow"],
            output_dir=hydro_out_dir
        )
        print("  ✓ Baseflow separation plotted")

        # 2. Flow Duration Curve (Using full dataset)
        vis.plot_flow_duration_curve(
            raw_df["discharge"], 
            output_dir=hydro_out_dir
        )
        print("  ✓ Flow duration curve plotted")
        
        # 3. Flood Frequency Curve
        if "flood_frequency" in results:
            vis.plot_flood_frequency(
                annual_max, results["flood_frequency"],
                output_dir=hydro_out_dir
            )
            print("  ✓ Flood frequency curve plotted")
            
        # 4. Advanced Catchment Physics
        run_advanced_catchment_analysis(
            df_in=raw_df, 
            output_dir=hydro_out_dir, 
            area_km2=basin_cfg.area_km2
        )

        # ─── ML MODEL OUTPUTS ──────────────────────────────────────
        ml_out_dir = str(ROOT / "outputs")

        # 5. Model-specific Hydrographs & Scatters (Top 3 models + Ensemble)
        # Add Ensemble to the trimmed dictionary for plotting
        if "weighted" in results.get("ensemble", {}):
            test_preds_trimmed["Weighted-Ensemble"] = ens_test_pred
            results["models"]["Weighted-Ensemble"] = results["ensemble"]["weighted"]

        for model_name, pred in test_preds_trimmed.items():
            metrics = results["models"].get(model_name, {}).get("test", None)
            vis.plot_model_hydrograph(test_dates, y_test_ens, pred, model_name, metrics, output_dir=ml_out_dir)
            vis.plot_model_scatter(y_test_ens, pred, model_name, metrics, output_dir=ml_out_dir)
        print("  ✓ ML Hydrographs and scatter plots generated")

        # 6. Combined Model Comparisons
        vis.plot_scatter_predictions(y_test_ens, test_preds_trimmed, output_dir=ml_out_dir)
        vis.plot_error_violin(y_test_ens, test_preds_trimmed, output_dir=ml_out_dir)
        print("  ✓ Combined error and comparison plots generated")

        # 7. CMIP6 Projections
        # 7. CMIP6 Projections (Updated Visualizations)
        if run_climate and "projections" in results.get("climate", {}):
            summary_df = pd.DataFrame(results["climate"]["projections"])
            baseline_mean = raw_df["discharge"].mean()
            vis.plot_cmip6_projections(summary_df, baseline_mean, output_dir=ml_out_dir)
            print("  ✓ CMIP6 climate projection summary plots generated")
            
            # Trigger the new Detailed, Yearly Aggregation, and Regime plots
            if all_projections:
                try:
                    vis.plot_cmip6_timeseries_with_rainfall(all_projections, output_dir=ml_out_dir)
                    vis.plot_cmip6_yearly_timeseries(all_projections, output_dir=ml_out_dir)
                    vis.plot_cmip6_annual_regime(all_projections, output_dir=ml_out_dir)
                except Exception as e:
                    print(f"  ⚠ Could not generate advanced CMIP6 plots: {e}")
            
        # 8. Top 5 Feature Importance (ALL Traditional Models)
        print("\n  Generating Feature Importance for all trained models...")
        for model_name, model_obj in trained_model_objs.items():
            try:
                # Safely extract the raw sklearn estimator if wrapped in a custom class
                sklearn_model = getattr(model_obj, 'model', model_obj)
                vis.plot_top_5_features(
                    model=sklearn_model,
                    X_val=X_val_np,
                    y_val=y_val_np,
                    feature_names=list(df_features.columns),
                    model_name=model_name,
                    output_dir=ml_out_dir
                )
            except Exception as e:
                print(f"  ⚠ Could not generate feature importance for {model_name}: {e}")

    elapsed = time.time() - start_time
    print("\n" + "═" * 70)
    print("  PIPELINE COMPLETE")
    print("═" * 70)
    print(f"  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    output_path = ROOT / "outputs" / "pipeline_results.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\n  Results saved to {output_path}")

    return results
    

if __name__ == "__main__":
    results = run_pipeline(
        use_synthetic=False,
        feature_combo="M1_full",
        run_traditional=True,
        run_deep=True,
        run_climate=True,
        pso_particles=10,
        pso_iterations=20,
        dl_epochs=50,
        seq_length=30,
    )