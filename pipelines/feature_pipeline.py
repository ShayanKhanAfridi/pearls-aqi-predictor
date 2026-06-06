# ============================================================
# Feature Pipeline — Live Hourly Updates
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

import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")
HOPSWORKS_HOST    = os.environ.get("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

CITY = "karachi"
LAT  = 24.9056
LON  = 67.0822
PIPELINE_VERSION = "v6"
# ================================================================


# ---- Shared OM client ----
_cache_session = requests_cache.CachedSession('.cache_feature', expire_after=3600)
_retry_session = retry(_cache_session, retries=3, backoff_factor=0.3)
OM_CLIENT      = openmeteo_requests.Client(session=_retry_session)


# ---- Fetch live AQI ----
def fetch_live_aqi(lat, lon):
    params = {
        "latitude":  lat,
        "longitude": lon,
        "current":   ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                      "sulphur_dioxide", "ozone", "us_aqi"],
        "timezone":  "Asia/Karachi",
    }
    response = OM_CLIENT.weather_api(
        "https://air-quality-api.open-meteo.com/v1/air-quality", params=params
    )[0]

    cur = response.Current()
    aqi_data = {
        "pm10": float(cur.Variables(0).Value()),
        "pm25": float(cur.Variables(1).Value()),
        "co":   float(cur.Variables(2).Value()),
        "no2":  float(cur.Variables(3).Value()),
        "so2":  float(cur.Variables(4).Value()),
        "o3":   float(cur.Variables(5).Value()),
        "aqi":  float(cur.Variables(6).Value()),
    }
    print(f"   AQI={aqi_data['aqi']:.0f}  PM2.5={aqi_data['pm25']:.1f}  PM10={aqi_data['pm10']:.1f}")
    return aqi_data


# ---- Fetch current weather + 72-hour forecast ----
def fetch_live_weather_and_forecast(lat, lon):
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "current":       ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                          "wind_speed_10m", "wind_direction_10m", "cloud_cover", "precipitation"],
        "hourly":        ["wind_speed_10m", "precipitation"],
        "forecast_days": 4,
        "timezone":      "Asia/Karachi",
    }
    response = OM_CLIENT.weather_api(
        "https://api.open-meteo.com/v1/forecast", params=params
    )[0]

    cur = response.Current()
    weather_data = {
        "temperature":   float(cur.Variables(0).Value()),
        "humidity":      float(cur.Variables(1).Value()),
        "pressure":      float(cur.Variables(2).Value()),
        "wind_speed":    float(cur.Variables(3).Value()) / 3.6,
        "wind_deg":      float(cur.Variables(4).Value()),
        "clouds":        float(cur.Variables(5).Value()),
        "precipitation": float(cur.Variables(6).Value()),
    }

    hrly = response.Hourly()
    timestamps = pd.date_range(
        start=pd.to_datetime(hrly.Time(),    unit="s", utc=True),
        end=pd.to_datetime(hrly.TimeEnd(),   unit="s", utc=True),
        freq=pd.Timedelta(seconds=hrly.Interval()),
        inclusive="left",
    )
    fc_df = pd.DataFrame({
        "timestamp":     timestamps,
        "wind_speed":    hrly.Variables(0).ValuesAsNumpy().astype(float) / 3.6,
        "precipitation": hrly.Variables(1).ValuesAsNumpy().astype(float),
    })

    now    = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    future = {}
    for h in [24, 48, 72]:
        target_ts = now + pd.Timedelta(hours=h)
        row       = fc_df[fc_df["timestamp"] >= target_ts]
        future[f"future_wind_{h}h"]   = float(row.iloc[0]["wind_speed"])    if len(row) > 0 else weather_data["wind_speed"]
        future[f"future_precip_{h}h"] = float(row.iloc[0]["precipitation"]) if len(row) > 0 else 0.0

    print(f"   Temp={weather_data['temperature']:.1f}°C  "
          f"Humidity={weather_data['humidity']:.0f}%  "
          f"Wind={weather_data['wind_speed']:.1f} m/s")
    return weather_data, future


# ---- Fetch lag features from Hopsworks ----
def fetch_real_lags(project):
    fs     = project.get_feature_store()
    fg     = fs.get_feature_group(name="aqi_features", version=1)
    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=175)

    df = None

    try:
        print("   Trying filtered Hive query...")
        df = (
            fg.select(["timestamp", "city", "aqi"])
              .filter(fg.timestamp >= cutoff)
              .read(read_options={"use_hive": True})
        )
        print(f"   Got {len(df)} rows")
    except Exception as e:
        print(f"   ⚠️  Filtered query failed ({e}) — falling back to full read...")

    if df is None or len(df) == 0:
        try:
            full_df = fg.read(read_options={"use_hive": True})
            full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
            df = full_df[
                (full_df["city"] == CITY) &
                (full_df["timestamp"] >= pd.Timestamp(cutoff))
            ][["timestamp", "city", "aqi"]]
            print(f"   Fallback got {len(df)} rows")
        except Exception as e:
            print(f"   ❌ Both reads failed: {e} — using current AQI as fallback")
            return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["city"] == CITY) & (df["timestamp"] < pd.Timestamp(now))]
    df = df.sort_values("timestamp").tail(175)

    if len(df) < 2:
        print("   ⚠️  Not enough history — using current AQI as fallback")
        return None

    s = df["aqi"].values

    def sg(n):
        return float(s[-n]) if len(s) >= n else float(s[-1])

    lags = {
        "aqi_lag1":              sg(1),
        "aqi_lag3":              sg(3),
        "aqi_lag6":              sg(6),
        "aqi_lag12":             sg(12),
        "aqi_lag24":             sg(24),
        "aqi_lag48":             sg(48),
        "aqi_lag168":            sg(168),
        "aqi_change_1h":         float(s[-1] - s[-2]) if len(s) >= 2 else 0.0,
        "aqi_change_3h":         float(s[-1] - s[-4]) if len(s) >= 4 else 0.0,
        "aqi_accel_1h":          float((s[-1] - s[-2]) - (s[-2] - s[-3])) if len(s) >= 3 else 0.0,
        "aqi_rolling_mean_6h":   float(np.mean(s[-6:]))   if len(s) >= 6   else float(s[-1]),
        "aqi_rolling_mean_24h":  float(np.mean(s[-24:]))  if len(s) >= 24  else float(s[-1]),
        "aqi_rolling_std_24h":   float(np.std(s[-24:]))   if len(s) >= 24  else 0.0,
        "aqi_rolling_mean_72h":  float(np.mean(s[-72:]))  if len(s) >= 72  else float(s[-1]),
        "aqi_rolling_mean_168h": float(np.mean(s[-168:])) if len(s) >= 168 else float(s[-1]),
        "aqi_rolling_std_72h":   float(np.std(s[-72:]))   if len(s) >= 72  else 0.0,
        "aqi_rolling_max_24h":   float(np.max(s[-24:]))   if len(s) >= 24  else float(s[-1]),
        "aqi_rolling_min_24h":   float(np.min(s[-24:]))   if len(s) >= 24  else float(s[-1]),
    }
    print(f"   lag1={lags['aqi_lag1']:.1f}  lag24={lags['aqi_lag24']:.1f}  lag168={lags['aqi_lag168']:.1f}")
    return lags


# ---- AQI category helper ----
def aqi_to_category(aqi: float) -> float:
    if aqi <= 50:   return 0.0
    if aqi <= 100:  return 1.0
    if aqi <= 150:  return 2.0
    if aqi <= 200:  return 3.0
    if aqi <= 300:  return 4.0
    return 5.0


# ---- Compute full feature row ----
def compute_features(aqi_data, weather_data, future_weather,
                     lag_features=None, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    hour         = timestamp.hour
    month        = timestamp.month
    day_of_week  = timestamp.weekday()
    day_of_year  = timestamp.timetuple().tm_yday
    aqi          = aqi_data["aqi"]
    wind_speed   = weather_data["wind_speed"]
    wind_deg     = weather_data["wind_deg"]
    pm25         = aqi_data["pm25"]

    def L(key, fallback):
        return lag_features[key] if (lag_features and key in lag_features) else fallback

    return {
        "timestamp":               timestamp.isoformat(),
        "city":                    CITY,
        "aqi":                     float(aqi),
        "pm25":                    float(pm25),
        "pm10":                    float(aqi_data["pm10"]),
        "o3":                      float(aqi_data["o3"]),
        "no2":                     float(aqi_data["no2"]),
        "so2":                     float(aqi_data["so2"]),
        "co":                      float(aqi_data["co"]),
        "aqi_category":            aqi_to_category(aqi),
        "temperature":             float(weather_data["temperature"]),
        "humidity":                float(weather_data["humidity"]),
        "pressure":                float(weather_data["pressure"]),
        "wind_speed":              float(wind_speed),
        "wind_deg":                float(wind_deg),
        "wind_x":                  float(wind_speed * np.cos(np.radians(wind_deg))),
        "wind_y":                  float(wind_speed * np.sin(np.radians(wind_deg))),
        "clouds":                  float(weather_data["clouds"]),
        "precipitation":           float(weather_data["precipitation"]),
        "future_wind_24h":         float(future_weather.get("future_wind_24h",  wind_speed)),
        "future_precip_24h":       float(future_weather.get("future_precip_24h", 0.0)),
        "future_wind_48h":         float(future_weather.get("future_wind_48h",  wind_speed)),
        "future_precip_48h":       float(future_weather.get("future_precip_48h", 0.0)),
        "future_wind_72h":         float(future_weather.get("future_wind_72h",  wind_speed)),
        "future_precip_72h":       float(future_weather.get("future_precip_72h", 0.0)),
        "hour":                    float(hour),
        "month":                   float(month),
        "day_of_week":             float(day_of_week),
        "day_of_year":             float(day_of_year),
        "hour_sin":                float(np.sin(2 * np.pi * hour        / 24)),
        "hour_cos":                float(np.cos(2 * np.pi * hour        / 24)),
        "month_sin":               float(np.sin(2 * np.pi * month       / 12)),
        "month_cos":               float(np.cos(2 * np.pi * month       / 12)),
        "dow_sin":                 float(np.sin(2 * np.pi * day_of_week /  7)),
        "dow_cos":                 float(np.cos(2 * np.pi * day_of_week /  7)),
        "doy_sin":                 float(np.sin(2 * np.pi * day_of_year / 365)),
        "doy_cos":                 float(np.cos(2 * np.pi * day_of_year / 365)),
        "is_weekend":              float(day_of_week >= 5),
        "is_winter":               float((month >= 11) or  (month <= 2)),
        "is_monsoon":              float((month >= 7)  and (month <= 9)),
        "aqi_lag1":                L("aqi_lag1",              aqi),
        "aqi_lag3":                L("aqi_lag3",              aqi),
        "aqi_lag6":                L("aqi_lag6",              aqi),
        "aqi_lag12":               L("aqi_lag12",             aqi),
        "aqi_lag24":               L("aqi_lag24",             aqi),
        "aqi_lag48":               L("aqi_lag48",             aqi),
        "aqi_lag168":              L("aqi_lag168",            aqi),
        "aqi_change_1h":           L("aqi_change_1h",         0.0),
        "aqi_change_3h":           L("aqi_change_3h",         0.0),
        "aqi_accel_1h":            L("aqi_accel_1h",          0.0),
        "aqi_rolling_mean_6h":     L("aqi_rolling_mean_6h",   aqi),
        "aqi_rolling_mean_24h":    L("aqi_rolling_mean_24h",  aqi),
        "aqi_rolling_std_24h":     L("aqi_rolling_std_24h",   0.0),
        "aqi_rolling_mean_72h":    L("aqi_rolling_mean_72h",  aqi),
        "aqi_rolling_mean_168h":   L("aqi_rolling_mean_168h", aqi),
        "aqi_rolling_std_72h":     L("aqi_rolling_std_72h",   0.0),
        "aqi_rolling_max_24h":     L("aqi_rolling_max_24h",   aqi),
        "aqi_rolling_min_24h":     L("aqi_rolling_min_24h",   aqi),
        "wind_x_pm25":             float(wind_speed * pm25),
        "humidity_x_pm25":         float(weather_data["humidity"] * pm25),
        "temp_x_aqi":              float(weather_data["temperature"] * aqi),
        "pollution_index":         float(pm25 * 0.5 + aqi_data["pm10"] * 0.3 + aqi_data["no2"] * 0.2),
        # ── v6: residual / regime features ──────────────────────────────
        "aqi_residual_168h":       float(L("aqi_lag1", aqi) - L("aqi_rolling_mean_168h", aqi)),
        "aqi_residual_72h":        float(L("aqi_lag1", aqi) - L("aqi_rolling_mean_72h",  aqi)),
        "aqi_regime_range":        float(L("aqi_rolling_max_24h", aqi) - L("aqi_rolling_min_24h", aqi)),
        "next_hour_aqi":           float("nan"),
    }


# ---- Push row to Hopsworks ----
def push_to_feature_store(row_dict, project):
    import platform
    if platform.system() == "Windows":
        print("\n⚠️  HDFS/Delta Lake push skipped on Windows (known limitation).")
        print("   Direct writes to Hopsworks HDFS are not supported on Windows external clients.")
        print(f"   Local row was computed successfully for timestamp: {row_dict['timestamp']}")
        return

    fs = project.get_feature_store()
    df = pd.DataFrame([row_dict])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for col in df.columns:
        if col not in ("city", "timestamp"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
        description=f"Hourly AQI features for Karachi — Open-Meteo ({PIPELINE_VERSION})",
    )
    try:
        fg.insert(df, write_options={"wait_for_job": False})
        print(f"✅ Pushed 1 row → Hopsworks at {row_dict['timestamp']}")
    except (OSError, ImportError, Exception) as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ["hdfs", "rpc", "delta", "listener"]):
            print(f"\n⚠️  HDFS/Delta Lake push skipped on Windows (known limitation).")
            print(f"   (Error: {str(e)[:120]})")
        else:
            raise


# ---- Main ----
def run_feature_pipeline():
    print("=" * 60)
    print(f"🌫️  FEATURE PIPELINE — LIVE RUN  [{PIPELINE_VERSION}]")
    print("=" * 60)

    print("\n[1/5] Fetching live AQI...")
    aqi_data = fetch_live_aqi(LAT, LON)

    print("\n[2/5] Fetching weather + 72h forecast...")
    weather_data, future_weather = fetch_live_weather_and_forecast(LAT, LON)

    print("\n[3/5] Logging into Hopsworks...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
    )


    print("\n[4/5] Fetching lag features...")
    lag_features = fetch_real_lags(project)

    print("\n[5/5] Computing features & pushing to store...")
    row = compute_features(aqi_data, weather_data, future_weather, lag_features)
    push_to_feature_store(row, project)

    print("\n📋 Feature row summary:")
    print(pd.Series(row).to_string())
    return row


if __name__ == "__main__":
    row = run_feature_pipeline()
