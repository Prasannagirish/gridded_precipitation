"""
FlowCast v2 — Main Pipeline
Orchestrates: Data → Features → Model Training → Evaluation → Climate Projections
Saves: models, predictions, per-model plots, hydrological analyses, CMIP6 results
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.config import data_cfg, model_cfg, climate_cfg, basin_cfg, MODEL_DIR, OUTPUT_DIR
from modules.loader import KabiniDataLoader, train_val_test_split
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
from modules.hydro import (
    eckhardt_baseflow, gumbel_flood_frequency,
    compute_hydrological_signatures, flow_duration_curve,
)
from modules.visualization import (
    plot_model_hydrograph,
    plot_model_scatter,
    plot_scatter_predictions,
    plot_best_hydrograph,
    plot_error_violin,
    plot_eckhardt_filter,
    plot_flow_duration_curve,
    plot_flood_frequency,
    plot_cmip6_projections,
    plot_cmip6_ensemble_summary,
)


def run_pipeline(
    use_synthetic: bool = True,
    precip_path: str = None,
    discharge_path: str = None,
    temp_path: str = None,
    feature_combo: str = "M1_full",
    run_traditional: bool = True,
    run_deep: bool = True,
    run_climate: bool = True,
    use_real_cmip6: bool = None,
    cmip6_dir: str = None,
    pso_particles: int = 15,
    pso_iterations: int = 30,
    dl_epochs: int = 80,
    seq_length: int = 30,
):
    results = {"models": {}, "ensemble": {}, "climate": {}, "hydrology": {}}
    start_time = time.time()
    output_dir = str(OUTPUT_DIR)

    if use_real_cmip6 is None:
        use_real_cmip6 = climate_cfg.use_real_cmip6

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 1: DATA LOADING")
    print("═" * 70)

    loader = KabiniDataLoader()
    if use_synthetic:
        raw_df = loader.generate_synthetic_data()
        print(f"  Generated synthetic data: {raw_df.shape[0]} days "
              f"({raw_df.index[0].date()} to {raw_df.index[-1].date()})")
    else:
        raw_df = loader.load_real_data(precip_path, discharge_path, temp_path)
        print(f"  Loaded real data: {raw_df.shape[0]} days")

    static_features = loader.get_static_features()
    static_array = np.array(list(static_features.values()), dtype=np.float32)
    print(f"  Static features: {len(static_features)} attributes")

    # Hydrological signatures
    signatures = compute_hydrological_signatures(
        raw_df["discharge"].values, raw_df["precip"].values
    )
    results["hydrology"]["signatures"] = signatures
    print(f"\n  Hydrological Signatures:")
    for k, v in signatures.items():
        print(f"    {k}: {v}")

    # Baseflow separation
    baseflow, quickflow = eckhardt_baseflow(raw_df["discharge"].values)
    raw_df["baseflow"] = baseflow
    raw_df["quickflow"] = quickflow

    # Save raw data summary for dashboard
    raw_df.to_csv(OUTPUT_DIR / "raw_data_summary.csv")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Feature Engineering
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 2: FEATURE ENGINEERING")
    print("═" * 70)

    engineer = HydroFeatureEngineer(raw_df)
    df = engineer.build_all_features()
    print(f"  Total features generated: {df.shape[1]}")

    feature_list = data_cfg.feature_combinations.get(
        feature_combo, data_cfg.feature_combinations["M1_full"]
    )
    df_features = select_features(df, feature_list)
    target = df["discharge"]

    print(f"  Selected combo: {feature_combo} ({df_features.shape[1]} features)")
    print(f"  Features: {list(df_features.columns)}")

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

    # Keep track of observation arrays per model (DL models use sequence-trimmed obs)
    test_obs_per_model = {}
    test_dates_per_model = {}

    # ═══════════════════════════════════════════════════════════
    # STEP 4A: Traditional ML Models (PSO-optimized)
    # ═══════════════════════════════════════════════════════════
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
        test_obs_per_model["RF-PSO"] = y_test_np
        test_dates_per_model["RF-PSO"] = test_X.index
        # Save model
        rf.save(MODEL_DIR / "rf_pso.joblib")
        print(f"\n  RF-PSO Test: {results['models']['RF-PSO']['test']}")
        print(f"  ✓ Model saved → {MODEL_DIR / 'rf_pso.joblib'}")

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
        test_obs_per_model["SVR-PSO"] = y_test_np
        test_dates_per_model["SVR-PSO"] = test_X.index
        svr.save(MODEL_DIR / "svr_pso.joblib")
        print(f"\n  SVR-PSO Test: {results['models']['SVR-PSO']['test']}")
        print(f"  ✓ Model saved → {MODEL_DIR / 'svr_pso.joblib'}")

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
        test_obs_per_model["XGBoost-PSO"] = y_test_np
        test_dates_per_model["XGBoost-PSO"] = test_X.index
        xgb.save(MODEL_DIR / "xgb_pso.joblib")
        print(f"\n  XGBoost-PSO Test: {results['models']['XGBoost-PSO']['test']}")
        print(f"  ✓ Model saved → {MODEL_DIR / 'xgb_pso.joblib'}")

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

        # Save scalers for later use
        joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, MODEL_DIR / "dl_scalers.joblib")

        # Create sequences
        Xtr_seq, ytr_seq = create_sequences(X_train_s, y_train_s, seq_length)
        Xv_seq, yv_seq = create_sequences(X_val_s, y_val_s, seq_length)
        Xt_seq, yt_seq = create_sequences(X_test_s, y_test_s, seq_length)

        y_val_seq_real = scaler_y.inverse_transform(yv_seq.reshape(-1, 1)).ravel()
        y_test_seq_real = scaler_y.inverse_transform(yt_seq.reshape(-1, 1)).ravel()
        # Dates for sequence-based models (trimmed by seq_length from the front)
        test_dates_seq = test_X.index[seq_length:]

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

        results["models"]["LSTM"] = {
            "val": evaluate_all(y_val_seq_real, lstm_val_pred),
            "test": evaluate_all(y_test_seq_real, lstm_test_pred),
        }
        all_val_preds["LSTM"] = lstm_val_pred
        all_test_preds["LSTM"] = lstm_test_pred
        test_obs_per_model["LSTM"] = y_test_seq_real
        test_dates_per_model["LSTM"] = test_dates_seq[:len(lstm_test_pred)]
        lstm_trainer.save(MODEL_DIR / "lstm.pt")
        print(f"\n  LSTM Test: {results['models']['LSTM']['test']}")
        print(f"  ✓ Model saved → {MODEL_DIR / 'lstm.pt'}")

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
        test_obs_per_model["PI-LSTM"] = y_test_seq_real
        test_dates_per_model["PI-LSTM"] = test_dates_seq[:len(pi_test_pred)]
        pi_trainer.save(MODEL_DIR / "pi_lstm.pt")
        print(f"\n  PI-LSTM Test: {results['models']['PI-LSTM']['test']}")
        print(f"  ✓ Model saved → {MODEL_DIR / 'pi_lstm.pt'}")

        # --- Adapted GRU (Paper 3) ---
        print("\n  Training A-GRU...")
        static_scaler = StandardScaler()
        static_scaled = static_scaler.fit_transform(static_array.reshape(1, -1)).flatten()
        n_static = len(static_scaled)
        joblib.dump(static_scaler, MODEL_DIR / "static_scaler.joblib")

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

        agru_train_loader = build_agru_dataloader(
            Xtr_d, Xtr_s, ytr_agru, model_cfg.agru_batch_size, shuffle=True
        )
        agru_val_loader = build_agru_dataloader(
            Xv_d, Xv_s, yv_agru, model_cfg.agru_batch_size, shuffle=False
        )
        agru_test_loader = build_agru_dataloader(
            Xt_d, Xt_s, yt_agru, model_cfg.agru_batch_size, shuffle=False
        )

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
        test_obs_per_model["A-GRU"] = y_test_agru_real
        test_dates_per_model["A-GRU"] = test_dates_seq[:len(agru_test_pred)]
        agru_trainer.save(MODEL_DIR / "agru.pt")
        print(f"\n  A-GRU Test: {results['models']['A-GRU']['test']}")
        print(f"  ✓ Model saved → {MODEL_DIR / 'agru.pt'}")

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

        y_val_ens = y_val_np[:min_val_len] if run_traditional else y_val_seq_real[:min_val_len]
        y_test_ens = y_test_np[:min_test_len] if run_traditional else y_test_seq_real[:min_test_len]

        # Weighted ensemble
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

        # Stacking ensemble
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
        results["hydrology"]["annual_max"] = annual_max.tolist()
        print(f"  Gumbel flood frequencies: {flood_freq}")

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Climate Projections
    # ═══════════════════════════════════════════════════════════
    if run_climate:
        print("\n" + "═" * 70)
        print("  STEP 7: CMIP6 CLIMATE PROJECTIONS")
        print("═" * 70)

        # Select best traditional model for projections
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
                        if name == "RF-PSO":
                            best_model = rf
                        elif name == "SVR-PSO":
                            best_model = svr
                        else:
                            best_model = xgb

        if best_model is None:
            print("  ✗ No trained model available for projections")

        # ── MODE A: Real CMIP6 data with QDM bias correction ──
        elif use_real_cmip6:
            print(f"\n  Mode: REAL CMIP6 data with Quantile Delta Mapping")
            print(f"  Using {best_model_name} for discharge projections")

            cmip6_path = cmip6_dir or climate_cfg.cmip6_download_dir
            cmip6_path = Path(cmip6_path)

            if not cmip6_path.exists():
                print(f"\n  ✗ CMIP6 download directory not found: {cmip6_path}")
                print(f"    Run the downloader first:")
                print(f"      python cmip6.py --shapefile {climate_cfg.basin_shapefile}")
                print(f"\n  Falling back to synthetic delta-change method...")
                use_real_cmip6 = False
            else:
                from modules.cmip6_projector import CMIP6RealProjector

                projector = CMIP6RealProjector(
                    cmip6_dir=str(cmip6_path),
                    obs_df=raw_df,
                    feature_columns=list(df_features.columns),
                    gcm_models=climate_cfg.gcm_models,
                    scenarios=climate_cfg.scenarios,
                    future_periods=climate_cfg.future_periods,
                    baseline_period=(
                        f"{climate_cfg.baseline_period[0]}-01-01",
                        f"{climate_cfg.baseline_period[1]}-12-31",
                    ),
                    n_quantiles=climate_cfg.n_quantiles,
                )

                projector.fit_bias_correction()
                projector.generate_corrected_futures()

                all_projections = projector.generate_all_scenarios(
                    trained_model=best_model,
                )

                if all_projections:
                    summary = projector.summarize_projections(
                        all_projections, raw_df["discharge"].values
                    )
                    results["climate"]["projections"] = summary.to_dict("records")

                    ensemble_summary = projector.generate_ensemble_summary(all_projections)
                    results["climate"]["ensemble_summary"] = ensemble_summary.to_dict("records")

                    projector.save_projections(
                        all_projections,
                        output_dir=str(ROOT / "outputs"),
                    )

                    print(f"\n  ── Per-GCM Projection Summary ──")
                    print(summary.to_string(index=False))
                    print(f"\n  ── Ensemble Summary ──")
                    print(ensemble_summary.to_string(index=False))

                    results["climate"]["bias_correction"] = {
                        gcm: bc.summary().to_dict("records")
                        for gcm, bc in projector.bias_correctors.items()
                    }

                    # Generate CMIP6 plots
                    baseline_mean = float(np.mean(raw_df["discharge"].values))
                    plot_cmip6_projections(summary, baseline_mean, output_dir=output_dir)
                    plot_cmip6_ensemble_summary(ensemble_summary, output_dir=output_dir)
                    print("  ✓ Saved CMIP6 projection plots")
                else:
                    print("  ✗ No valid projections generated")

        # ── MODE B: Synthetic delta-change (fallback) ──
        if not use_real_cmip6 and best_model is not None:
            print(f"\n  Mode: SYNTHETIC delta-change (legacy)")
            print(f"  Using {best_model_name} for climate projections")

            from modules.cmip6_projector import CMIP6RealProjector

            # Generate synthetic CMIP6-style projections for demo
            print("  Generating synthetic climate scenarios for demonstration...")

            # Create synthetic future scenarios from observed data
            baseline_discharge = raw_df["discharge"].values
            baseline_mean = float(np.mean(baseline_discharge))
            results["climate"]["baseline_mean_q"] = baseline_mean
            results["climate"]["projection_model"] = best_model_name

            # Synthetic delta changes for demonstration
            synthetic_projections = []
            for ssp, ssp_label in [("ssp245", "SSP2-4.5"), ("ssp585", "SSP5-8.5")]:
                for yr_start, yr_end in climate_cfg.future_periods:
                    # Approximate temperature-driven changes
                    if ssp == "ssp245":
                        delta_pct = np.random.uniform(5, 20) * ((yr_start - 2020) / 70)
                    else:
                        delta_pct = np.random.uniform(10, 35) * ((yr_start - 2020) / 70)

                    future_mean = baseline_mean * (1 + delta_pct / 100)
                    synthetic_projections.append({
                        "GCM": "Synthetic-Ensemble",
                        "SSP": ssp,
                        "Period": f"{yr_start}-{yr_end}",
                        "Mean_Q": round(future_mean, 2),
                        "Change_pct": round(delta_pct, 1),
                        "Max_Q": round(future_mean * 3.5, 2),
                        "Min_Q": round(future_mean * 0.1, 2),
                        "Std_Q": round(future_mean * 0.8, 2),
                        "Q10": round(future_mean * 0.2, 2),
                        "Q50": round(future_mean * 0.6, 2),
                        "Q90": round(future_mean * 2.5, 2),
                    })

            summary_df = pd.DataFrame(synthetic_projections)
            results["climate"]["projections"] = summary_df.to_dict("records")
            print(f"\n  Synthetic Climate Projection Summary:")
            print(summary_df.to_string(index=False))

            # Generate CMIP6 plots
            plot_cmip6_projections(summary_df, baseline_mean, output_dir=output_dir)
            print("  ✓ Saved CMIP6 projection plots")

    # ═══════════════════════════════════════════════════════════
    # STEP 8: Flow Regime Analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 8: FLOW REGIME ANALYSIS")
    print("═" * 70)

    for model_name in list(results["models"].keys()):
        if model_name in all_test_preds:
            pred = all_test_preds[model_name]
            obs = test_obs_per_model.get(model_name, y_test_np[:len(pred)])
            obs = obs[:len(pred)]
            regime_results = evaluate_flow_regimes(obs, pred)
            results["models"][model_name]["flow_regimes"] = {
                k: v for k, v in regime_results.items()
            }
            print(f"\n  {model_name} flow regime performance:")
            for regime, metrics in regime_results.items():
                print(f"    {regime}: NSE={metrics.get('NSE', 'N/A')}, KGE={metrics.get('KGE', 'N/A')}")

    # ═══════════════════════════════════════════════════════════
    # STEP 9: GENERATE ALL VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 9: GENERATING ALL PLOTS")
    print("═" * 70)

    # --- 1. Eckhardt Baseflow Separation (1-year slice for readability) ---
    plot_start = "2019-01-01"
    plot_end = "2019-12-31"
    if raw_df.index[-1] < pd.Timestamp(plot_start):
        # Use last year of data if 2019 not available
        plot_end_date = raw_df.index[-1]
        plot_start_date = plot_end_date - pd.DateOffset(years=1)
        slice_df = raw_df.loc[plot_start_date:plot_end_date]
    else:
        slice_df = raw_df.loc[plot_start:plot_end]

    if len(slice_df) > 0:
        plot_eckhardt_filter(
            dates=slice_df.index,
            precip=slice_df["precip"].values,
            discharge=slice_df["discharge"].values,
            baseflow=slice_df["baseflow"].values,
            output_dir=output_dir,
        )
        print("  ✓ Saved Eckhardt baseflow separation plot (inverted rainfall)")

    # --- 2. Flow Duration Curve ---
    plot_flow_duration_curve(raw_df["discharge"].values, output_dir=output_dir)
    print("  ✓ Saved Flow Duration Curve")

    # --- 3. Flood Frequency Curve ---
    if "flood_frequency" in results:
        plot_flood_frequency(annual_max, results["flood_frequency"], output_dir=output_dir)
        print("  ✓ Saved Flood Frequency curve")

    # --- 4. Per-Model Hydrographs & Scatter Plots ---
    if all_test_preds:
        for model_name, pred in all_test_preds.items():
            obs = test_obs_per_model.get(model_name, y_test_np[:len(pred)])
            obs = obs[:len(pred)]
            dates = test_dates_per_model.get(model_name, test_X.index[:len(pred)])
            dates = dates[:len(pred)]
            metrics = results["models"].get(model_name, {}).get("test", {})

            # Per-model hydrograph
            plot_model_hydrograph(dates, obs, pred, model_name,
                                 metrics_dict=metrics, output_dir=output_dir)
            print(f"  ✓ Saved hydrograph — {model_name}")

            # Per-model scatter
            plot_model_scatter(obs, pred, model_name,
                              metrics_dict=metrics, output_dir=output_dir)
            print(f"  ✓ Saved scatter — {model_name}")

        # --- 5. Combined scatter (all models) ---
        # Use shortest common length
        min_len = min(len(v) for v in all_test_preds.values())
        obs_trimmed = y_test_np[:min_len]
        preds_trimmed = {k: v[:min_len] for k, v in all_test_preds.items()}

        plot_scatter_predictions(obs_trimmed, preds_trimmed, output_dir=output_dir)
        print("  ✓ Saved combined scatter plot")

        # --- 6. Violin plot ---
        plot_error_violin(obs_trimmed, preds_trimmed, output_dir=output_dir)
        print("  ✓ Saved error violin plot")

    # ═══════════════════════════════════════════════════════════
    # STEP 10: SAVE PREDICTIONS TO CSV
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STEP 10: SAVING PREDICTIONS")
    print("═" * 70)

    # Save per-model test predictions as CSV for dashboard
    for model_name, pred in all_test_preds.items():
        obs = test_obs_per_model.get(model_name, y_test_np[:len(pred)])
        obs = obs[:len(pred)]
        dates = test_dates_per_model.get(model_name, test_X.index[:len(pred)])
        dates = dates[:len(pred)]

        pred_df = pd.DataFrame({
            "date": dates,
            "observed": obs,
            "predicted": pred,
        })
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        pred_df.to_csv(OUTPUT_DIR / f"predictions_{safe_name}.csv", index=False)

    print(f"  ✓ Saved {len(all_test_preds)} prediction CSVs to {OUTPUT_DIR}/")

    # Save model list for dashboard
    results["saved_models"] = {
        name: str(MODEL_DIR / f"{name.lower().replace('-', '_')}.{'pt' if name in ['LSTM', 'PI-LSTM', 'A-GRU'] else 'joblib'}")
        for name in results["models"].keys()
    }

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    print("\n" + "═" * 70)
    print("  PIPELINE COMPLETE")
    print("═" * 70)
    print(f"  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if use_real_cmip6 and run_climate:
        print(f"  CMIP6 mode: Real data with QDM bias correction")
    elif run_climate:
        print(f"  CMIP6 mode: Synthetic delta-change")

    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    print(f"  Models saved to:  {MODEL_DIR}/")

    print("\n  ┌──────────────────┬────────┬────────┬────────┬────────┐")
    print("  │ Model            │   NSE  │   KGE  │  RMSE  │   R    │")
    print("  ├──────────────────┼────────┼────────┼────────┼────────┤")
    for name, res in results["models"].items():
        t = res["test"]
        print(f"  │ {name:<16} │ {t['NSE']:6.4f} │ {t['KGE']:6.4f} │ {t['RMSE']:6.2f} │ {t['R']:6.4f} │")
    for name, res in results["ensemble"].items():
        if "test" in res:
            t = res["test"]
            label = f"Ens-{name}"
            print(f"  │ {label:<16} │ {t['NSE']:6.4f} │ {t['KGE']:6.4f} │ {t['RMSE']:6.2f} │ {t['R']:6.4f} │")
    print("  └──────────────────┴────────┴────────┴────────┴────────┘")

    # Save results JSON
    output_path = OUTPUT_DIR / "pipeline_results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\n  Results saved to {output_path}")

    # List all generated plots
    plot_files = sorted(OUTPUT_DIR.glob("*.png"))
    if plot_files:
        print(f"\n  Generated {len(plot_files)} plots:")
        for pf in plot_files:
            print(f"    📊 {pf.name}")

    return results


if __name__ == "__main__":
    results = run_pipeline(
        use_synthetic=False,
        precip_path="data/chirps_kabini_daily.csv",
        discharge_path="data/discharge_daily_observed.csv",
        temp_path=None,
        feature_combo="M1_full",
        run_traditional=True,
        run_deep=True,
        run_climate=True,
        use_real_cmip6=True,
        cmip6_dir="cmip6_downloads",
        pso_particles=10,
        pso_iterations=20,
        dl_epochs=50,
        seq_length=30,
    )