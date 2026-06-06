# ============================================================
# Backfill Pipeline — Historical Data
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import hopsworks
import openmeteo_requests
import requests_cache
from retry_requests import retry
import time
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")

CITY       = "karachi"
LAT        = 24.9056
LON        = 67.0822

START_DATE = "2023-06-05"
END_DATE   = "2026-06-04"

CHUNK_DAYS = 90

RAW_CSV_PATH   = "data/raw/aqi_backfill_raw.csv"
ENGINEERED_CSV = "data/engineered/aqi_backfill_engineered.csv"

PIPELINE_VERSION = "v6"
# ================================================================


# ---- Shared OM client ----
_cache_session = requests_cache.CachedSession('.cache_backfill', expire_after=86400)
_retry_session = retry(_cache_session, retries=3, backoff_factor=0.1)
OM_CLIENT      = openmeteo_requests.Client(session=_retry_session)


# ---- Fetch AQI chunk ----
def fetch_aqi_chunk(lat, lon, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                       "sulphur_dioxide", "ozone", "us_aqi"],
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "Asia/Karachi",
    }
    response = OM_CLIENT.weather_api(
        "https://air-quality-api.open-meteo.com/v1/air-quality", params=params
    )[0]
    hrly = response.Hourly()
    timestamps = pd.date_range(
        start=pd.to_datetime(hrly.Time(),    unit="s", utc=True),
        end=pd.to_datetime(hrly.TimeEnd(),   unit="s", utc=True),
        freq=pd.Timedelta(seconds=hrly.Interval()),
        inclusive="left",
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "pm10": hrly.Variables(0).ValuesAsNumpy().astype(float),
        "pm25": hrly.Variables(1).ValuesAsNumpy().astype(float),
        "co":   hrly.Variables(2).ValuesAsNumpy().astype(float),
        "no2":  hrly.Variables(3).ValuesAsNumpy().astype(float),
        "so2":  hrly.Variables(4).ValuesAsNumpy().astype(float),
        "o3":   hrly.Variables(5).ValuesAsNumpy().astype(float),
        "aqi":  hrly.Variables(6).ValuesAsNumpy().astype(float),
    })


# ---- Fetch weather chunk ----
def fetch_weather_chunk(lat, lon, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                       "wind_speed_10m", "wind_direction_10m", "cloud_cover", "precipitation"],
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "Asia/Karachi",
    }
    response = OM_CLIENT.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )[0]
    hrly = response.Hourly()
    timestamps = pd.date_range(
        start=pd.to_datetime(hrly.Time(),    unit="s", utc=True),
        end=pd.to_datetime(hrly.TimeEnd(),   unit="s", utc=True),
        freq=pd.Timedelta(seconds=hrly.Interval()),
        inclusive="left",
    )
    return pd.DataFrame({
        "timestamp":     timestamps,
        "temperature":   hrly.Variables(0).ValuesAsNumpy().astype(float),
        "humidity":      hrly.Variables(1).ValuesAsNumpy().astype(float),
        "pressure":      hrly.Variables(2).ValuesAsNumpy().astype(float),
        "wind_speed":    hrly.Variables(3).ValuesAsNumpy().astype(float) / 3.6,  # km/h → m/s
        "wind_deg":      hrly.Variables(4).ValuesAsNumpy().astype(float),
        "clouds":        hrly.Variables(5).ValuesAsNumpy().astype(float),
        "precipitation": hrly.Variables(6).ValuesAsNumpy().astype(float),
    })


# ---- Fetch all chunks ----
def fetch_all_historical_data() -> pd.DataFrame:
    all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq=f"{CHUNK_DAYS}D")
    chunks = []
    for i in range(len(all_dates)):
        cs = all_dates[i]
        ce = (all_dates[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(all_dates) else pd.Timestamp(END_DATE)
        chunks.append((cs.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))

    print(f"📅 Period   : {START_DATE}  →  {END_DATE}  (HOURLY)")
    print(f"   Chunks   : {len(chunks)}  ×  {CHUNK_DAYS} days")
    print(f"   Expected : ~{len(chunks) * CHUNK_DAYS * 24:,} rows\n")

    all_merged = []
    t_start    = time.time()

    for idx, (cs, ce) in enumerate(chunks, 1):
        t0 = time.time()
        print(f"  Chunk {idx:>2}/{len(chunks)}  {cs} → {ce}", end="  ", flush=True)
        try:
            aqi_df     = fetch_aqi_chunk(LAT, LON, cs, ce)
            weather_df = fetch_weather_chunk(LAT, LON, cs, ce)
            merged     = pd.merge(aqi_df, weather_df, on="timestamp", how="inner")
            all_merged.append(merged)
            print(f"→ {len(merged):,} rows  ({time.time()-t0:.1f}s) ✅")
        except Exception as ex:
            print(f"→ ERROR: {ex} ⚠️  skipped")

    if not all_merged:
        raise RuntimeError("No data fetched — check your API connectivity")

    raw_df = (
        pd.concat(all_merged, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    print(f"\n✅ Fetch done in {time.time()-t_start:.1f}s  |  {len(raw_df):,} rows total")
    return raw_df


# ---- AQI category helper ----
def aqi_to_category(aqi_series: pd.Series) -> pd.Series:
    """Ordinal encoding: Good=0 … Hazardous=5"""
    bins   = [-np.inf, 50, 100, 150, 200, 300, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    return pd.cut(aqi_series, bins=bins, labels=labels).astype(float)


# ---- Feature engineering — mirrors feature_pipeline v6 and training_pipeline v6 ----
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    fill_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                 "temperature", "humidity", "pressure",
                 "wind_speed", "wind_deg", "clouds", "precipitation"]
    df[fill_cols] = df[fill_cols].ffill().bfill()

    # ── AQI category ────────────────────────────────────────────────
    df["aqi_category"] = aqi_to_category(df["aqi"])

    # ── Time features ────────────────────────────────────────────────
    df["hour"]        = df["timestamp"].dt.hour.astype(float)
    df["month"]       = df["timestamp"].dt.month.astype(float)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(float)
    df["day_of_year"] = df["timestamp"].dt.dayofyear.astype(float)

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] /  7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] /  7)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["is_winter"]  = ((df["month"] >= 11) | (df["month"] <= 2)).astype(float)
    df["is_monsoon"] = ((df["month"] >= 7)  & (df["month"] <= 9)).astype(float)

    # ── Wind components ──────────────────────────────────────────────
    df["wind_x"] = df["wind_speed"] * np.cos(np.radians(df["wind_deg"]))
    df["wind_y"] = df["wind_speed"] * np.sin(np.radians(df["wind_deg"]))

    # ── Lag features ─────────────────────────────────────────────────
    df["aqi_lag1"]   = df["aqi"].shift(1)
    df["aqi_lag3"]   = df["aqi"].shift(3)
    df["aqi_lag6"]   = df["aqi"].shift(6)
    df["aqi_lag12"]  = df["aqi"].shift(12)
    df["aqi_lag24"]  = df["aqi"].shift(24)
    df["aqi_lag48"]  = df["aqi"].shift(48)
    df["aqi_lag168"] = df["aqi"].shift(168)

    # ── 1st & 2nd order differences ──────────────────────────────────
    df["aqi_change_1h"] = df["aqi"].diff(1)
    df["aqi_change_3h"] = df["aqi"].diff(3)
    df["aqi_accel_1h"]  = df["aqi_change_1h"].diff(1)

    # ── Short rolling ────────────────────────────────────────────────
    df["aqi_rolling_mean_6h"]  = df["aqi"].rolling(window=6,   min_periods=1).mean()
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24,  min_periods=1).mean()
    df["aqi_rolling_std_24h"]  = df["aqi"].rolling(window=24,  min_periods=1).std().fillna(0.0)

    # ── Long rolling ─────────────────────────────────────────────────
    df["aqi_rolling_mean_72h"]  = df["aqi"].rolling(window=72,  min_periods=1).mean()
    df["aqi_rolling_mean_168h"] = df["aqi"].rolling(window=168, min_periods=1).mean()
    df["aqi_rolling_std_72h"]   = df["aqi"].rolling(window=72,  min_periods=1).std().fillna(0.0)
    df["aqi_rolling_max_24h"]   = df["aqi"].rolling(window=24,  min_periods=1).max()
    df["aqi_rolling_min_24h"]   = df["aqi"].rolling(window=24,  min_periods=1).min()

    # ── Future weather (simulated forecast values) ───────────────────
    df["future_wind_24h"]   = df["wind_speed"].shift(-24).ffill()
    df["future_precip_24h"] = df["precipitation"].shift(-24).fillna(0.0)
    df["future_wind_48h"]   = df["wind_speed"].shift(-48).ffill()
    df["future_precip_48h"] = df["precipitation"].shift(-48).fillna(0.0)
    df["future_wind_72h"]   = df["wind_speed"].shift(-72).ffill()
    df["future_precip_72h"] = df["precipitation"].shift(-72).fillna(0.0)

    # ── Interaction features ──────────────────────────────────────────
    df["wind_x_pm25"]     = df["wind_speed"] * df["pm25"]
    df["humidity_x_pm25"] = df["humidity"]    * df["pm25"]
    df["temp_x_aqi"]      = df["temperature"] * df["aqi"]
    df["pollution_index"] = (df["pm25"] * 0.5 + df["pm10"] * 0.3 + df["no2"] * 0.2)

    # ── v6: Residual / regime features ───────────────────────────────
    df["aqi_residual_168h"] = df["aqi_lag1"] - df["aqi_rolling_mean_168h"]
    df["aqi_residual_72h"]  = df["aqi_lag1"] - df["aqi_rolling_mean_72h"]
    df["aqi_regime_range"]  = df["aqi_rolling_max_24h"] - df["aqi_rolling_min_24h"]

    # ── Target ────────────────────────────────────────────────────────
    df["next_hour_aqi"] = df["aqi"].shift(-1)

    # ── City ──────────────────────────────────────────────────────────
    df["city"] = CITY

    return df


# ---- Column order (matches training pipeline v6 exactly) ----
COLUMN_ORDER = [
    "timestamp", "city",
    # Raw pollutants
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    # AQI category
    "aqi_category",
    # Weather
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_deg", "wind_x", "wind_y",
    "clouds", "precipitation",
    # Future weather forecasts
    "future_wind_24h", "future_precip_24h",
    "future_wind_48h", "future_precip_48h",
    "future_wind_72h", "future_precip_72h",
    # Time
    "hour", "month", "day_of_week", "day_of_year",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "is_weekend", "is_winter", "is_monsoon",
    # Lag features
    "aqi_lag1", "aqi_lag3", "aqi_lag6", "aqi_lag12",
    "aqi_lag24", "aqi_lag48", "aqi_lag168",
    # Differences
    "aqi_change_1h", "aqi_change_3h", "aqi_accel_1h",
    # Short rolling
    "aqi_rolling_mean_6h", "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
    # Long rolling
    "aqi_rolling_mean_72h", "aqi_rolling_mean_168h", "aqi_rolling_std_72h",
    "aqi_rolling_max_24h", "aqi_rolling_min_24h",
    # Interactions
    "wind_x_pm25", "humidity_x_pm25", "temp_x_aqi", "pollution_index",
    # v6: Residual / regime features
    "aqi_residual_168h",
    "aqi_residual_72h",
    "aqi_regime_range",
    # Target
    "next_hour_aqi",
]

RAW_COLS = [
    "timestamp", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds", "precipitation",
]


# ---- Push to Hopsworks ----
def push_to_hopsworks(df: pd.DataFrame, project):
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,   # <-- disables Kafka/online store; no twofish needed
        description=(
            f"Hourly historical AQI features for Karachi — "
            f"Open-Meteo archive backfill ({PIPELINE_VERSION}). "
            f"Includes residual/regime features: aqi_residual_168h, "
            f"aqi_residual_72h, aqi_regime_range."
        ),
    )

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    float_cols = [c for c in df.columns if c not in ("city", "timestamp")]
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    print(f"\n📤 Pushing {len(df):,} rows in a single insert...")
    fg.insert(df, write_options={"wait_for_job": True})
    print("✅ All data pushed to Hopsworks!")


# ---- Main ----
def run_backfill_pipeline():
    # Ensure output directories exist
    os.makedirs("data/raw",        exist_ok=True)
    os.makedirs("data/engineered", exist_ok=True)
    os.makedirs("model",           exist_ok=True)

    print("=" * 65)
    print(f"📦  BACKFILL PIPELINE — Hourly, 3-Year Historical Data  [{PIPELINE_VERSION}]")
    print(f"    {START_DATE}  →  {END_DATE}")
    print(f"    New in v6: aqi_residual_168h, aqi_residual_72h, aqi_regime_range")
    print("=" * 65)

    print("\n[1/5] Fetching hourly data from Open-Meteo...")
    raw_df = fetch_all_historical_data()

    print(f"\n[2/5] Saving raw CSV → '{RAW_CSV_PATH}'")
    raw_df[RAW_COLS].to_csv(RAW_CSV_PATH, index=False)
    print(f"   ✅ {len(raw_df):,} rows  |  {len(RAW_COLS)} columns")

    print("\n[3/5] Engineering features...")
    eng_df = engineer_features(raw_df.copy())
    eng_df = eng_df[[c for c in COLUMN_ORDER if c in eng_df.columns]].reset_index(drop=True)

    # Drop rows where core lag features are NaN (first 168 rows expected)
    before = len(eng_df)
    eng_df = eng_df.dropna(
        subset=["aqi_lag1", "aqi_lag24", "aqi_lag168"]
    ).reset_index(drop=True)
    print(f"   Dropped {before - len(eng_df)} rows (NaN lags — expected for first 168 rows)")
    print(f"   Final rows: {len(eng_df):,}")

    # Verify v6 columns are present
    v6_cols = ["aqi_residual_168h", "aqi_residual_72h", "aqi_regime_range"]
    missing_v6 = [c for c in v6_cols if c not in eng_df.columns]
    if missing_v6:
        raise RuntimeError(f"❌ v6 feature columns missing after engineering: {missing_v6}")
    print(f"   ✅ v6 residual/regime features verified: {v6_cols}")

    print(f"\n[4/5] Saving engineered CSV → '{ENGINEERED_CSV}'")
    eng_df.to_csv(ENGINEERED_CSV, index=False)
    print(f"   ✅ {len(eng_df):,} rows  |  {len(eng_df.columns)} columns")

    print("\n[5/5] Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    push_to_hopsworks(eng_df, project)

    print("\n" + "=" * 65)
    print(f"✅  BACKFILL COMPLETE  [{PIPELINE_VERSION}]")
    print("=" * 65)
    print(f"   Rows            : {len(eng_df):,}  (~{len(eng_df)/24/365:.1f} years hourly)")
    print(f"   Date range      : {eng_df['timestamp'].min()}  →  {eng_df['timestamp'].max()}")
    print(f"   Columns         : {len(eng_df.columns)}")
    print(f"   AQI mean ± std  : {eng_df['aqi'].mean():.1f} ± {eng_df['aqi'].std():.1f}")
    print(f"   AQI range       : {eng_df['aqi'].min():.0f}  –  {eng_df['aqi'].max():.0f}")
    print(f"\n   📁 Raw CSV      : {RAW_CSV_PATH}")
    print(f"   📁 Engineered   : {ENGINEERED_CSV}")
    return eng_df


if __name__ == "__main__":
    eng_df = run_backfill_pipeline()
    print("\n📋 Last 5 rows:")
    print(eng_df.tail(5)[[
        "timestamp", "aqi", "aqi_lag1", "aqi_lag24",
        "aqi_rolling_mean_72h",
        "aqi_residual_168h", "aqi_residual_72h", "aqi_regime_range",
        "future_wind_24h", "next_hour_aqi",
    ]].to_string())