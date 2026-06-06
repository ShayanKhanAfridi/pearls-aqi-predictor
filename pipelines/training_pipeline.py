# ============================================================
# Training Pipeline — Multi-Output Horizon
# ============================================================

import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

# ── DNS fix for Windows: ensure hopsworks domains resolve correctly ──
if sys.platform == "win32":
    import socket as _socket
    _HOPSWORKS_IP = "57.130.17.86"  # eu-west.cloud.hopsworks.ai
    _DNS_OVERRIDES = {
        "eu-west.cloud.hopsworks.ai": _HOPSWORKS_IP,
    }
    _orig_getaddrinfo = _socket.getaddrinfo

    def _patched_getaddrinfo(host, port, *args, **kwargs):
        if host in _DNS_OVERRIDES:
            port_num = port if port else 443
            return [(2, 1, 6, '', (_DNS_OVERRIDES[host], port_num))]
        return _orig_getaddrinfo(host, port, *args, **kwargs)

    _socket.getaddrinfo = _patched_getaddrinfo
# ── End DNS fix ──

import pandas as pd
import numpy as np
import hopsworks
import joblib
import warnings
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")

# ============================================================
# FEATURE COLUMNS
# ============================================================
FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2",
    "aqi_category",
    "temperature", "humidity", "wind_speed", "wind_x", "wind_y",
    "pressure", "clouds", "precipitation",
    "future_wind_24h", "future_precip_24h",
    "future_wind_48h", "future_precip_48h",
    "future_wind_72h", "future_precip_72h",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "is_weekend", "is_winter", "is_monsoon",
    "aqi_lag1", "aqi_lag3", "aqi_lag6",
    "aqi_lag12", "aqi_lag24", "aqi_lag48", "aqi_lag168",
    "aqi_change_1h", "aqi_change_3h", "aqi_accel_1h",
    "aqi_rolling_mean_6h", "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
    "aqi_rolling_mean_72h", "aqi_rolling_mean_168h", "aqi_rolling_std_72h",
    "aqi_rolling_max_24h", "aqi_rolling_min_24h",
    "wind_x_pm25", "humidity_x_pm25", "temp_x_aqi", "pollution_index",
    "aqi_residual_168h",
    "aqi_residual_72h",
    "aqi_regime_range",
]

TARGET_COLS = ["target_day1", "target_day2", "target_day3"]

DAY_WEIGHTS = {"day1": 0.20, "day2": 0.35, "day3": 0.45}

OVERFIT_THRESHOLD = 30.0  # % gap between train and val RMSE — models above this are disqualified


# ============================================================
# Fetch training data from Hopsworks
# ============================================================
def fetch_training_data():
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs  = project.get_feature_store()
    fg  = fs.get_feature_group(name="aqi_features", version=1)
    df  = fg.read()
    print(f"✅ Fetched {len(df):,} rows from feature store")
    return df, project


# ============================================================
# Build multi-output targets + splits
# ============================================================
def prepare_data(df: pd.DataFrame):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    time_diffs   = df["timestamp"].diff().dropna()
    median_hours = time_diffs.median().total_seconds() / 3600
    is_3hourly   = median_hours >= 2.5

    if is_3hourly:
        print(f"   Detected 3-hourly data (median gap = {median_hours:.1f}h)")
        D1_STEPS = list(range(1, 9))
        D2_STEPS = list(range(9, 17))
        D3_STEPS = list(range(17, 25))
    else:
        print(f"   Detected hourly data (median gap = {median_hours:.1f}h)")
        D1_STEPS = list(range(1, 25))
        D2_STEPS = list(range(25, 49))
        D3_STEPS = list(range(49, 73))

    print("⏳ Building multi-output targets...")

    def gaussian_weights(steps, sigma=8.0):
        steps_arr = np.array(steps, dtype=float)
        center    = steps_arr.mean()
        w         = np.exp(-0.5 * ((steps_arr - center) / sigma) ** 2)
        return w / w.sum()

    df["target_day1"] = sum(df["aqi"].shift(-s) for s in D1_STEPS) / len(D1_STEPS)

    w2 = gaussian_weights(D2_STEPS)
    df["target_day2"] = sum(
        float(w2[i]) * df["aqi"].shift(-s) for i, s in enumerate(D2_STEPS)
    )

    w3 = gaussian_weights(D3_STEPS)
    df["target_day3"] = sum(
        float(w3[i]) * df["aqi"].shift(-s) for i, s in enumerate(D3_STEPS)
    )

    df["aqi_residual_168h"] = df["aqi_lag1"] - df["aqi_rolling_mean_168h"]
    df["aqi_residual_72h"]  = df["aqi_lag1"] - df["aqi_rolling_mean_72h"]
    df["aqi_regime_range"]  = df["aqi_rolling_max_24h"] - df["aqi_rolling_min_24h"]

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"   ⚠️  Missing feature columns (will be skipped): {missing}")

    df = df.dropna(subset=available_features + TARGET_COLS).reset_index(drop=True)
    print(f"✅ Valid rows after target creation: {len(df):,}")

    X = df[available_features].reset_index(drop=True)
    Y = df[TARGET_COLS].reset_index(drop=True)

    n         = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_tr, Y_tr = X.iloc[:train_end],        Y.iloc[:train_end]
    X_va, Y_va = X.iloc[train_end:val_end], Y.iloc[train_end:val_end]
    X_te, Y_te = X.iloc[val_end:],          Y.iloc[val_end:]

    print(f"   Train: {len(X_tr):,} rows  |  Val: {len(X_va):,}  |  Test: {len(X_te):,}")
    print(f"   Features used: {len(available_features)}")

    return (X_tr, Y_tr, X_va, Y_va, X_te, Y_te), available_features, df


# ============================================================
# Model factory
# ============================================================
def build_candidate_models():
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(Ridge(alpha=10.0))),
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=30,
            max_features=0.4,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=600,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.65,
                colsample_bytree=0.6,
                min_child_weight=20,
                reg_alpha=0.2,
                reg_lambda=5.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            n_jobs=-1,
        ),

        "LightGBM": MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.03,
                num_leaves=20,
                subsample=0.65,
                colsample_bytree=0.6,
                min_child_samples=50,
                reg_alpha=0.2,
                reg_lambda=5.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            n_jobs=-1,
        ),

        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.65,
                min_samples_leaf=30,
                random_state=42,
            ),
            n_jobs=-1,
        ),
    }
    return models


# ============================================================
# Evaluate
# ============================================================
def evaluate_multi_output(model, X, Y, label=""):
    preds = np.clip(model.predict(X), 0, 500)
    Y_arr = Y.values if isinstance(Y, pd.DataFrame) else Y

    metrics = {}
    for i, day in enumerate(["day1", "day2", "day3"]):
        rmse = float(np.sqrt(mean_squared_error(Y_arr[:, i], preds[:, i])))
        mae  = float(mean_absolute_error(Y_arr[:, i], preds[:, i]))
        r2   = float(r2_score(Y_arr[:, i], preds[:, i]))
        metrics[day] = {"rmse": rmse, "mae": mae, "r2": r2, "preds": preds[:, i]}

    metrics["weighted_rmse"] = float(
        sum(DAY_WEIGHTS[d] * metrics[d]["rmse"] for d in ["day1", "day2", "day3"])
    )
    metrics["mean_rmse"] = float(
        np.mean([metrics[d]["rmse"] for d in ["day1", "day2", "day3"]])
    )

    if label:
        print(
            f"  {label:<18} │ "
            f"D1: RMSE={metrics['day1']['rmse']:6.2f} R²={metrics['day1']['r2']:5.3f} │ "
            f"D2: RMSE={metrics['day2']['rmse']:6.2f} R²={metrics['day2']['r2']:5.3f} │ "
            f"D3: RMSE={metrics['day3']['rmse']:6.2f} R²={metrics['day3']['r2']:5.3f} │ "
            f"Weighted={metrics['weighted_rmse']:6.2f}"
        )
    return metrics


# ============================================================
# Overfitting check
# ============================================================
def overfitting_check(train_metrics, val_metrics, name):
    print(f"\n  📊 Overfitting check — {name}:")
    print(f"  {'Day':<8} {'Train RMSE':>11} {'Val RMSE':>10} {'Gap %':>8}  Status")
    print(f"  {'─'*52}")
    any_overfit = False
    for day in ["day1", "day2", "day3"]:
        tr  = train_metrics[day]["rmse"]
        va  = val_metrics[day]["rmse"]
        gap = ((va - tr) / max(tr, 1e-9)) * 100
        status = "⚠️  OVERFIT" if gap > OVERFIT_THRESHOLD else "✅ OK"
        if gap > OVERFIT_THRESHOLD:
            any_overfit = True
        print(f"  {day:<8} {tr:>11.2f} {va:>10.2f} {gap:>7.1f}%  {status}")
    if not any_overfit:
        print(f"  → No significant overfitting detected.")
    return any_overfit


# ============================================================
# Train, validate, select best model
# ============================================================
def train_and_select_best(splits, feature_cols):
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = splits

    X_trval = pd.concat([X_tr, X_va]).reset_index(drop=True)
    Y_trval = pd.concat([Y_tr, Y_va]).reset_index(drop=True)

    print(f"\n{'='*110}")
    print("🏋️  TRAINING ALL CANDIDATE MODELS")
    print(f"    Model selection: WEIGHTED RMSE  (Day1×{DAY_WEIGHTS['day1']}  Day2×{DAY_WEIGHTS['day2']}  Day3×{DAY_WEIGHTS['day3']})")
    print(f"    Overfitting threshold: {OVERFIT_THRESHOLD}% gap on any day disqualifies a model")
    print(f"    5 candidates: Ridge, RandomForest, XGBoost, LightGBM, GradientBoosting")
    print(f"{'='*110}")
    print(f"\n{'Validation Results':^110}")
    print("─" * 110)

    val_results      = {}
    train_results    = {}
    disqualified     = {}
    candidates       = build_candidate_models()

    for name, model in candidates.items():
        print(f"\n  ▶ Training {name}...")
        model.fit(X_tr, Y_tr)

        tr_m = evaluate_multi_output(model, X_tr, Y_tr)
        va_m = evaluate_multi_output(model, X_va, Y_va, label=name)

        is_overfit = overfitting_check(tr_m, va_m, name)

        train_results[name] = tr_m

        if is_overfit:
            disqualified[name] = {"model": model, "val_metrics": va_m}
            print(f"  ❌ {name} disqualified — overfitting detected (gap > {OVERFIT_THRESHOLD}%)")
        else:
            val_results[name] = {"model": model, "val_metrics": va_m}

    if not val_results:
        print("\n  ⚠️  All models exceeded the overfitting threshold!")
        print("  ↩️  Falling back to least-overfit model from disqualified pool...")
        def max_gap(name):
            tr = train_results[name]
            va = disqualified[name]["val_metrics"]
            return max(
                ((va[d]["rmse"] - tr[d]["rmse"]) / max(tr[d]["rmse"], 1e-9)) * 100
                for d in ["day1", "day2", "day3"]
            )
        fallback_name = min(disqualified, key=max_gap)
        val_results[fallback_name] = disqualified[fallback_name]
        print(f"  ↩️  Using {fallback_name} as fallback.")

    best_name  = min(val_results, key=lambda k: val_results[k]["val_metrics"]["weighted_rmse"])
    best_model = candidates[best_name]

    print(f"\n  🏆 Best model on validation (weighted RMSE): {best_name}  "
          f"(weighted = {val_results[best_name]['val_metrics']['weighted_rmse']:.2f}  "
          f"mean = {val_results[best_name]['val_metrics']['mean_rmse']:.2f})")

    print(f"\n  ▶ Retraining {best_name} on train+val combined ({len(X_trval):,} rows)...")
    best_model.fit(X_trval, Y_trval)

    print(f"\n{'='*110}")
    print(f"📊  FULL METRICS — {best_name}  (retrained on train+val)")
    print(f"{'='*110}")

    tr_final = evaluate_multi_output(best_model, X_tr, Y_tr)
    va_final = evaluate_multi_output(best_model, X_va, Y_va)
    te_final = evaluate_multi_output(best_model, X_te, Y_te)

    print(f"\n  {'Split':<8} │ {'Day1 RMSE':>10} {'Day1 R²':>8} │ {'Day2 RMSE':>10} {'Day2 R²':>8} │ {'Day3 RMSE':>10} {'Day3 R²':>8} │ {'WtdRMSE':>9}")
    print("─" * 110)
    for split_name, m in [("Train", tr_final), ("Val", va_final), ("Test", te_final)]:
        print(
            f"  {split_name:<8} │ "
            f"{m['day1']['rmse']:>10.2f} {m['day1']['r2']:>8.3f} │ "
            f"{m['day2']['rmse']:>10.2f} {m['day2']['r2']:>8.3f} │ "
            f"{m['day3']['rmse']:>10.2f} {m['day3']['r2']:>8.3f} │ "
            f"{m['weighted_rmse']:>9.2f}"
        )
    print("─" * 110)

    print(f"\n  All models — Validation summary:")
    print(f"\n  {'Model':<18} {'Status':<14} │ {'D1 RMSE':>8} {'D1 R²':>7} │ {'D2 RMSE':>8} {'D2 R²':>7} │ {'D3 RMSE':>8} {'D3 R²':>7} │ {'WtdRMSE':>9} {'MeanRMSE':>10}")
    print("─" * 120)
    all_results = {**val_results, **{k: v for k, v in disqualified.items() if k not in val_results}}
    for name, info in all_results.items():
        m      = info["val_metrics"]
        status = "✅ Qualified" if name in val_results else "❌ Overfit"
        marker = "  ◀ BEST" if name == best_name else ""
        print(
            f"  {name:<18} {status:<14} │ "
            f"{m['day1']['rmse']:>8.2f} {m['day1']['r2']:>7.3f} │ "
            f"{m['day2']['rmse']:>8.2f} {m['day2']['r2']:>7.3f} │ "
            f"{m['day3']['rmse']:>8.2f} {m['day3']['r2']:>7.3f} │ "
            f"{m['weighted_rmse']:>9.2f} {m['mean_rmse']:>10.2f}{marker}"
        )
    print("─" * 120)

    return best_model, best_name, te_final, val_results, (tr_final, va_final, te_final)


# ============================================================
# Sample predictions
# ============================================================
def show_sample_predictions(model, X_te, Y_te, n_samples=20):
    preds  = np.clip(model.predict(X_te), 0, 500)
    Y_arr  = Y_te.values if isinstance(Y_te, pd.DataFrame) else Y_te

    rng = np.random.default_rng(seed=42)
    idx = np.sort(rng.choice(len(X_te), size=min(n_samples, len(X_te)), replace=False))

    print(f"\n{'='*110}")
    print(f"🔍  SAMPLE PREDICTIONS (n={len(idx)}, random from test set)")
    print(f"{'='*110}")
    print(
        f"  {'#':>4}  │ "
        f"{'Act D1':>7} {'Pred D1':>8} {'Err D1':>7} │ "
        f"{'Act D2':>7} {'Pred D2':>8} {'Err D2':>7} │ "
        f"{'Act D3':>7} {'Pred D3':>8} {'Err D3':>7}"
    )
    print("─" * 110)
    for i in idx:
        a1, a2, a3 = Y_arr[i]
        p1, p2, p3 = preds[i]
        print(
            f"  {i:>4}  │ "
            f"{a1:>7.1f} {p1:>8.1f} {abs(a1-p1):>7.1f} │ "
            f"{a2:>7.1f} {p2:>8.1f} {abs(a2-p2):>7.1f} │ "
            f"{a3:>7.1f} {p3:>8.1f} {abs(a3-p3):>7.1f}"
        )

    sample_a = Y_arr[idx]
    sample_p = preds[idx]
    mae_d1   = np.mean(np.abs(sample_a[:, 0] - sample_p[:, 0]))
    mae_d2   = np.mean(np.abs(sample_a[:, 1] - sample_p[:, 1]))
    mae_d3   = np.mean(np.abs(sample_a[:, 2] - sample_p[:, 2]))
    print("─" * 110)
    print(
        f"  {'MAE':>4}  │ "
        f"{'':>7} {'':>8} {mae_d1:>7.1f} │ "
        f"{'':>7} {'':>8} {mae_d2:>7.1f} │ "
        f"{'':>7} {'':>8} {mae_d3:>7.1f}"
    )
    print("─" * 110)


# ============================================================
# Save model to Hopsworks Model Registry
# ============================================================
def save_to_model_registry(project, best_model, best_name, test_metrics,
                            val_results, feature_cols):
    mr = project.get_model_registry()

    artifact = {
        "model":          best_model,
        "model_name":     best_name,
        "feature_cols":   feature_cols,
        "target_cols":    TARGET_COLS,
        "day_weights":    DAY_WEIGHTS,
        "val_comparison": {
            name: {
                "day1_rmse":      info["val_metrics"]["day1"]["rmse"],
                "day1_r2":        info["val_metrics"]["day1"]["r2"],
                "day2_rmse":      info["val_metrics"]["day2"]["rmse"],
                "day2_r2":        info["val_metrics"]["day2"]["r2"],
                "day3_rmse":      info["val_metrics"]["day3"]["rmse"],
                "day3_r2":        info["val_metrics"]["day3"]["r2"],
                "weighted_rmse":  info["val_metrics"]["weighted_rmse"],
                "mean_rmse":      info["val_metrics"]["mean_rmse"],
            }
            for name, info in val_results.items()
        },
    }

    os.makedirs("model", exist_ok=True)
    model_path = "model/aqi_multioutput_model.pkl"
    joblib.dump(artifact, model_path)

    registry_metrics = {
        "test_day1_rmse":     round(test_metrics["day1"]["rmse"], 4),
        "test_day1_mae":      round(test_metrics["day1"]["mae"],  4),
        "test_day1_r2":       round(test_metrics["day1"]["r2"],   4),
        "test_day2_rmse":     round(test_metrics["day2"]["rmse"], 4),
        "test_day2_mae":      round(test_metrics["day2"]["mae"],  4),
        "test_day2_r2":       round(test_metrics["day2"]["r2"],   4),
        "test_day3_rmse":     round(test_metrics["day3"]["rmse"], 4),
        "test_day3_mae":      round(test_metrics["day3"]["mae"],  4),
        "test_day3_r2":       round(test_metrics["day3"]["r2"],   4),
        "test_mean_rmse":     round(test_metrics["mean_rmse"],    4),
        "test_weighted_rmse": round(test_metrics["weighted_rmse"], 4),
    }

    hw_model = mr.python.create_model(
        name="aqi_predictor_multioutput",
        metrics=registry_metrics,
        description=(
            f"Multi-output horizon AQI forecasting for Karachi. "
            f"Best algorithm: {best_name}. "
            f"Predicts Day1 (0-24h), Day2 (24-48h), Day3 (48-72h) simultaneously. "
            f"Features: {len(feature_cols)}. "
            f"Trained on 70%% of data, selected on weighted RMSE (D1×0.20 D2×0.35 D3×0.45) "
            f"from 15%% val, evaluated on 15%% test."
        ),
    )
    hw_model.save(model_path)

    model_uri = getattr(hw_model, "model_uri", None) or getattr(hw_model, "version_path", "N/A")

    print(f"\n✅ Model saved to Hopsworks Model Registry!")
    print(f"   Model name:  aqi_predictor_multioutput")
    print(f"   Algorithm:   {best_name}")
    print(f"   Model URI:   {model_uri}")

    print(f"\n   📋 Registry metrics (TEST SET — unseen data):")
    print(f"   {'Metric':<25}  Value")
    print(f"   {'─'*38}")
    for k, v in registry_metrics.items():
        print(f"   {k:<25}  {v}")

    return hw_model


# ============================================================
# Inference helper
# ============================================================
def predict_3day(artifact_path: str, feature_row: pd.DataFrame) -> dict:
    artifact  = joblib.load(artifact_path)
    model     = artifact["model"]
    feat_cols = artifact["feature_cols"]

    X     = feature_row[feat_cols].values
    preds = np.clip(model.predict(X), 0, 500)[0]

    return {
        "day1_aqi": float(preds[0]),
        "day2_aqi": float(preds[1]),
        "day3_aqi": float(preds[2]),
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 110)
    print("🚀  AQI MULTI-OUTPUT HORIZON TRAINING PIPELINE")
    print("=" * 110)

    print("\n[1/5] Fetching data from Hopsworks Feature Store...")
    df, project = fetch_training_data()

    print("\n[2/5] Preparing multi-output targets and train/val/test splits...")
    splits, feature_cols, df_full = prepare_data(df)

    print(f"\n   AQI stats from full dataset:")
    print(f"   Mean={df_full['aqi'].mean():.1f}  Std={df_full['aqi'].std():.1f}  "
          f"Min={df_full['aqi'].min():.1f}  Max={df_full['aqi'].max():.1f}")

    print("\n[3/5] Training and selecting best model...")
    best_model, best_name, test_metrics, val_results, all_split_metrics = \
        train_and_select_best(splits, feature_cols)

    print("\n[4/5] Showing sample predictions...")
    _, _, X_te, Y_te = (splits[0], splits[1], splits[4], splits[5])
    show_sample_predictions(best_model, X_te, Y_te, n_samples=25)

    print("\n[5/5] Saving best model to Hopsworks Model Registry...")
    hw_model = save_to_model_registry(
        project, best_model, best_name, test_metrics, val_results, feature_cols
    )

    print("\n" + "=" * 110)
    print("🎉  TRAINING PIPELINE COMPLETE")
    print("=" * 110)
    print(f"\n  ✅ Best model         : {best_name}")
    print(f"  ✅ Test Day1 RMSE     : {test_metrics['day1']['rmse']:.2f}   "
          f"MAE={test_metrics['day1']['mae']:.2f}   R²={test_metrics['day1']['r2']:.3f}")
    print(f"  ✅ Test Day2 RMSE     : {test_metrics['day2']['rmse']:.2f}   "
          f"MAE={test_metrics['day2']['mae']:.2f}   R²={test_metrics['day2']['r2']:.3f}")
    print(f"  ✅ Test Day3 RMSE     : {test_metrics['day3']['rmse']:.2f}   "
          f"MAE={test_metrics['day3']['mae']:.2f}   R²={test_metrics['day3']['r2']:.3f}")
    print(f"  ✅ Weighted RMSE      : {test_metrics['weighted_rmse']:.2f}")
    print(f"  ✅ Mean RMSE          : {test_metrics['mean_rmse']:.2f}")
    print(f"\n  Model artifact: model/aqi_multioutput_model.pkl")
    model_uri = getattr(hw_model, "model_uri", None) or getattr(hw_model, "version_path", "N/A")
    print(f"  Hopsworks URI : {model_uri}")
    print("\n  Use predict_3day() to load artifact and run inference.")
    print("=" * 110)