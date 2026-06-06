# ============================================================
# NOTEBOOK 5 — Inference Pipeline  [v7]
#
# Loads the best model from Hopsworks Model Registry,
# fetches the latest live features from the Feature Store,
# generates AQI predictions for Day1 / Day2 / Day3,
# and saves a predictions JSON + CSV for the dashboard.
#
# Matches:
#   Feature Pipeline v6  (feature columns + residual features)
#   Training Pipeline v7 (model artifact format)
#
# Run:  python notebook5_inference.py
# Schedule via GitHub Actions: every hour (after feature pipeline)
# ============================================================

# ---- CELL 1: Imports ----
import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import hopsworks
import joblib
import json
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")

CITY              = "karachi"
MODEL_NAME        = "aqi_predictor_multioutput"
LOCAL_ARTIFACT    = "model/aqi_multioutput_model.pkl"   # fallback local path
PREDICTIONS_JSON  = "predictions_latest.json"
PREDICTIONS_CSV   = "predictions_history.csv"
PIPELINE_VERSION  = "v7"
# ================================================================

# AQI category helper
def aqi_to_category_label(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"

def aqi_color(aqi: float) -> str:
    if aqi <= 50:   return "#00e400"
    if aqi <= 100:  return "#ffff00"
    if aqi <= 150:  return "#ff7e00"
    if aqi <= 200:  return "#ff0000"
    if aqi <= 300:  return "#8f3f97"
    return "#7e0023"

def health_advice(aqi: float) -> str:
    if aqi <= 50:
        return "Air quality is satisfactory. Enjoy outdoor activities."
    if aqi <= 100:
        return "Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion."
    if aqi <= 150:
        return "Sensitive groups (children, elderly, those with respiratory conditions) should reduce prolonged outdoor exertion."
    if aqi <= 200:
        return "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion."
    if aqi <= 300:
        return "Health alert: everyone may experience serious health effects. Avoid outdoor activities."
    return "Health warning of emergency conditions. Everyone should avoid all outdoor activities."


# ================================================================
# CELL 2: Connect to Hopsworks
# ================================================================
def connect_hopsworks():
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    print(f"   ✅ Connected to project: {project.name}")
    return project


# ================================================================
# CELL 3: Load model from registry (with local fallback)
# ================================================================
def load_model(project) -> dict:
    print(f"\n📦 Loading best model '{MODEL_NAME}' from registry (lowest test_weighted_rmse)...")
    try:
        mr        = project.get_model_registry()
        hw_model  = mr.get_best_model(MODEL_NAME, metric="test_weighted_rmse", direction="min")
        model_dir = hw_model.download()
        pkl_files = list(Path(model_dir).glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError("No .pkl found in downloaded model dir")
        artifact = joblib.load(pkl_files[0])
        print(f"   ✅ Model downloaded from registry: {pkl_files[0].name}")
    except Exception as e:
        print(f"   ⚠️  Registry load failed ({e}) — trying local fallback...")
        if not Path(LOCAL_ARTIFACT).exists():
            raise FileNotFoundError(
                f"Local artifact '{LOCAL_ARTIFACT}' not found either. "
                "Run Training Pipeline (nb3) first."
            )
        artifact = joblib.load(LOCAL_ARTIFACT)
        print(f"   ✅ Loaded local artifact: {LOCAL_ARTIFACT}")

    model      = artifact["model"]
    feat_cols  = artifact["feature_cols"]
    algo_name  = artifact.get("model_name", "Unknown")
    print(f"   Algorithm    : {algo_name}")
    print(f"   Features     : {len(feat_cols)}")
    return artifact


# ================================================================
# CELL 4: Fetch latest feature row from Feature Store
# ================================================================
def fetch_latest_features(project) -> pd.DataFrame:
    print("\n🗄️  Fetching latest features from Feature Store...")
    fs  = project.get_feature_store()
    fg  = fs.get_feature_group(name="aqi_features", version=1)

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=3)   # last 3 hours

    df = None
    try:
        df = (
            fg.select_all()
              .filter(fg.city == CITY)
              .filter(fg.timestamp >= cutoff)
              .read(read_options={"use_hive": True})
        )
        print(f"   Got {len(df)} rows from filtered query")
    except Exception as e:
        print(f"   ⚠️  Filtered query failed ({e}) — full read...")

    if df is None or len(df) == 0:
        try:
            df_full = fg.read(read_options={"use_hive": True})
            df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], utc=True)
            df = df_full[df_full["city"] == CITY].sort_values("timestamp").tail(5)
            print(f"   Got {len(df)} rows from full read fallback")
        except Exception as e:
            raise RuntimeError(f"Cannot fetch features from Hopsworks: {e}")

    if len(df) == 0:
        raise RuntimeError("No feature rows found for Karachi. Run Feature Pipeline first.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    latest = df.iloc[[-1]].reset_index(drop=True)
    print(f"   Latest row timestamp : {latest['timestamp'].iloc[0]}")
    print(f"   Current AQI          : {latest['aqi'].iloc[0]:.0f}")
    return latest


# ================================================================
# CELL 5: Build feature row — ensure all columns present
# ================================================================
def prepare_feature_row(latest: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    feat_cols = artifact["feature_cols"]
    missing   = [c for c in feat_cols if c not in latest.columns]

    if missing:
        print(f"   ⚠️  {len(missing)} feature cols missing in live row — filling with 0:")
        print(f"       {missing}")
        for c in missing:
            latest[c] = 0.0

    # Ensure float
    for c in feat_cols:
        latest[c] = pd.to_numeric(latest[c], errors="coerce").fillna(0.0)

    return latest[feat_cols]


# ================================================================
# CELL 6: Run inference
# ================================================================
def run_inference(X: pd.DataFrame, artifact: dict) -> dict:
    model = artifact["model"]
    preds = np.clip(model.predict(X), 0, 500)[0]
    return {
        "day1_aqi": float(round(preds[0], 1)),
        "day2_aqi": float(round(preds[1], 1)),
        "day3_aqi": float(round(preds[2], 1)),
    }


# ================================================================
# CELL 7: Build prediction payload
# ================================================================
def build_payload(latest_row: pd.DataFrame, preds: dict) -> dict:
    now  = datetime.now(timezone.utc)
    ts   = latest_row["timestamp"].iloc[0]
    curr = float(latest_row["aqi"].iloc[0])

    day1_date = (now + timedelta(days=1)).strftime("%A, %b %d")
    day2_date = (now + timedelta(days=2)).strftime("%A, %b %d")
    day3_date = (now + timedelta(days=3)).strftime("%A, %b %d")

    def enrich(aqi_val, label):
        return {
            "aqi":          aqi_val,
            "category":     aqi_to_category_label(aqi_val),
            "color":        aqi_color(aqi_val),
            "health_advice":health_advice(aqi_val),
            "date_label":   label,
        }

    payload = {
        "generated_at":   now.isoformat(),
        "data_timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        "city":           CITY.title(),
        "pipeline_version": PIPELINE_VERSION,
        "current": {
            "aqi":           curr,
            "category":      aqi_to_category_label(curr),
            "color":         aqi_color(curr),
            "health_advice": health_advice(curr),
            "pm25":          float(latest_row.get("pm25", [np.nan]).iloc[0]) if "pm25" in latest_row.columns else None,
            "pm10":          float(latest_row.get("pm10", [np.nan]).iloc[0]) if "pm10" in latest_row.columns else None,
            "temperature":   float(latest_row.get("temperature", [np.nan]).iloc[0]) if "temperature" in latest_row.columns else None,
            "humidity":      float(latest_row.get("humidity", [np.nan]).iloc[0]) if "humidity" in latest_row.columns else None,
            "wind_speed":    float(latest_row.get("wind_speed", [np.nan]).iloc[0]) if "wind_speed" in latest_row.columns else None,
        },
        "forecast": {
            "day1": enrich(preds["day1_aqi"], day1_date),
            "day2": enrich(preds["day2_aqi"], day2_date),
            "day3": enrich(preds["day3_aqi"], day3_date),
        },
        "alert": None,
    }

    # Alert logic
    max_forecast = max(preds["day1_aqi"], preds["day2_aqi"], preds["day3_aqi"])
    if max_forecast > 300:
        payload["alert"] = {"level": "HAZARDOUS",     "message": "🚨 Hazardous AQI forecast — avoid all outdoor activity!"}
    elif max_forecast > 200:
        payload["alert"] = {"level": "VERY_UNHEALTHY","message": "⚠️ Very unhealthy AQI forecast — stay indoors."}
    elif max_forecast > 150:
        payload["alert"] = {"level": "UNHEALTHY",     "message": "⚠️ Unhealthy AQI forecast — sensitive groups take care."}
    elif curr > 150:
        payload["alert"] = {"level": "CURRENT_HIGH",  "message": "⚠️ Current AQI is unhealthy — limit outdoor exposure."}

    return payload


# ================================================================
# CELL 8: Save outputs
# ================================================================
def save_outputs(payload: dict, latest_row: pd.DataFrame):
    # Save latest predictions JSON (dashboard reads this)
    with open(PREDICTIONS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n   💾 Predictions JSON  → {PREDICTIONS_JSON}")

    # Append to history CSV
    row = {
        "generated_at":   payload["generated_at"],
        "current_aqi":    payload["current"]["aqi"],
        "day1_aqi":       payload["forecast"]["day1"]["aqi"],
        "day1_category":  payload["forecast"]["day1"]["category"],
        "day2_aqi":       payload["forecast"]["day2"]["aqi"],
        "day2_category":  payload["forecast"]["day2"]["category"],
        "day3_aqi":       payload["forecast"]["day3"]["aqi"],
        "day3_category":  payload["forecast"]["day3"]["category"],
    }

    hist_path = Path(PREDICTIONS_CSV)
    hist_df   = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    hist_df   = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)
    hist_df.to_csv(hist_path, index=False)
    print(f"   💾 History CSV       → {PREDICTIONS_CSV}  ({len(hist_df)} rows)")


# ================================================================
# CELL 9: Print results
# ================================================================
def print_results(payload: dict):
    print("\n" + "="*60)
    print(f"🌫️  AQI FORECAST — {payload['city'].upper()}")
    print("="*60)

    c = payload["current"]
    print(f"\n  📍 Current  AQI : {c['aqi']:>6.0f}  [{c['category']}]")
    print(f"     PM2.5       : {c.get('pm25') or 'N/A'}")
    print(f"     Temp        : {c.get('temperature') or 'N/A'} °C")
    print(f"     Humidity    : {c.get('humidity') or 'N/A'} %")
    print(f"     Wind        : {c.get('wind_speed') or 'N/A'} m/s")

    print(f"\n  📅 3-Day Forecast:")
    print(f"  {'Day':<8} {'Date':<22} {'AQI':>6}  Category")
    print("  " + "-"*54)
    for day in ["day1","day2","day3"]:
        fc = payload["forecast"][day]
        print(f"  {day:<8} {fc['date_label']:<22} {fc['aqi']:>6.0f}  {fc['category']}")

    if payload.get("alert"):
        print(f"\n  {payload['alert']['message']}")

    print(f"\n  ⏰ Generated at: {payload['generated_at']}")
    print("="*60)


# ================================================================
# CELL 10: Push predictions to Feature Store (optional logging)
# ================================================================
def push_predictions_to_store(project, payload: dict):
    """
    Optionally logs forecast results to a separate 'aqi_predictions'
    feature group for dashboard and monitoring purposes.
    """
    print("\n📤 Pushing predictions to 'aqi_predictions' feature group...")
    try:
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name="aqi_predictions",
            version=1,
            primary_key=["city", "generated_at"],
            event_time="generated_at",
            description=f"AQI 3-day forecasts for Karachi ({PIPELINE_VERSION})",
        )

        row = {
            "city":          payload["city"].lower(),
            "generated_at":  pd.Timestamp(payload["generated_at"]),
            "current_aqi":   float(payload["current"]["aqi"]),
            "day1_aqi":      float(payload["forecast"]["day1"]["aqi"]),
            "day1_category": payload["forecast"]["day1"]["category"],
            "day2_aqi":      float(payload["forecast"]["day2"]["aqi"]),
            "day2_category": payload["forecast"]["day2"]["category"],
            "day3_aqi":      float(payload["forecast"]["day3"]["aqi"]),
            "day3_category": payload["forecast"]["day3"]["category"],
            "has_alert":     float(1 if payload["alert"] else 0),
        }

        df = pd.DataFrame([row])
        fg.insert(df, write_options={"wait_for_job": False})
        print("   ✅ Predictions logged to Hopsworks.")
    except Exception as e:
        print(f"   ⚠️  Could not push predictions to store: {e}")


# ================================================================
# CELL 11: Main
# ================================================================
def run_inference_pipeline(push_to_store: bool = True):
    print("="*60)
    print(f"🚀  INFERENCE PIPELINE  [{PIPELINE_VERSION}]")
    print(f"    City: {CITY.title()}")
    print("="*60)

    print("\n[1/5] Connecting to Hopsworks...")
    project = connect_hopsworks()

    print("\n[2/5] Loading model from registry...")
    artifact = load_model(project)

    print("\n[3/5] Fetching latest live features...")
    latest_row = fetch_latest_features(project)

    print("\n[4/5] Running inference...")
    X      = prepare_feature_row(latest_row.copy(), artifact)
    preds  = run_inference(X, artifact)
    payload = build_payload(latest_row, preds)

    print("\n[5/5] Saving outputs...")
    save_outputs(payload, latest_row)

    if push_to_store:
        push_predictions_to_store(project, payload)

    print_results(payload)
    return payload


if __name__ == "__main__":
    payload = run_inference_pipeline(push_to_store=True)
