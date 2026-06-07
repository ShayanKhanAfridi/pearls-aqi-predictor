# ============================================================
# Streamlit Dashboard — Pearls AQI Predictor (Dark Theme)
# 5 pages: Home | Forecast | Historical | Health Advisory | Pollutant Breakdown
# ============================================================

import os
import sys
from dotenv import load_dotenv
load_dotenv()

try:
    import streamlit as st
    HOPSWORKS_API_KEY = st.secrets.get("HOPSWORKS_API_KEY", os.environ.get("HOPSWORKS_API_KEY", ""))
except Exception:
    HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")
    import streamlit as st

import pandas as pd
import numpy as np
import json
import joblib
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta

import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ── Windows-only DNS fix (not needed on Streamlit Cloud / Linux) ──
if sys.platform == "win32":
    import socket as _socket
    _HOPSWORKS_IP = "57.130.17.86"
    _DNS_OVERRIDES = {"eu-west.cloud.hopsworks.ai": _HOPSWORKS_IP}
    _orig_getaddrinfo = _socket.getaddrinfo
    def _patched_getaddrinfo(host, port, *args, **kwargs):
        if host in _DNS_OVERRIDES:
            return [(2, 1, 6, '', (_DNS_OVERRIDES[host], port if port else 443))]
        return _orig_getaddrinfo(host, port, *args, **kwargs)
    _socket.getaddrinfo = _patched_getaddrinfo

# ================================================================
# CONFIG
# ================================================================
CITY              = "karachi"
MODEL_NAME        = "aqi_predictor_multioutput"
BACKFILL_CSV      = "data/engineered/aqi_backfill_engineered.csv"
SHAP_PLOT_DIR     = Path("shap_plots")
EDA_PLOT_DIR      = Path("eda_plots")
PIPELINE_VERSION  = "v7"
HOPSWORKS_HOST    = os.environ.get("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

# ================================================================
# PAGE CONFIG — must be first Streamlit call
# ================================================================
st.set_page_config(
    page_title="Pearls AQI Predictor · Karachi",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# HELPERS  (logic unchanged)
# ================================================================
def aqi_color(aqi: float) -> str:
    if aqi <= 50:   return "#00e400"
    if aqi <= 100:  return "#ffff00"
    if aqi <= 150:  return "#ff7e00"
    if aqi <= 200:  return "#ff0000"
    if aqi <= 300:  return "#8f3f97"
    return "#7e0023"

def aqi_text_color(aqi: float) -> str:
    return "#111" if aqi <= 100 else "#ffffff"

def aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"

def health_advice(aqi: float) -> str:
    if aqi <= 50:   return "Air quality is satisfactory. Enjoy outdoor activities."
    if aqi <= 100:  return "Acceptable quality. Unusually sensitive people should limit prolonged outdoor exertion."
    if aqi <= 150:  return "Sensitive groups should reduce prolonged outdoor exertion."
    if aqi <= 200:  return "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion."
    if aqi <= 300:  return "Health alert: everyone may experience serious health effects. Avoid outdoor activities."
    return "Health warning of emergency conditions. Everyone should avoid all outdoor activities."

AQI_SCALE = [
    {"range": "0–50",    "category": "Good",                           "color": "#00e400"},
    {"range": "51–100",  "category": "Moderate",                       "color": "#ffff00"},
    {"range": "101–150", "category": "Unhealthy for Sensitive Groups",  "color": "#ff7e00"},
    {"range": "151–200", "category": "Unhealthy",                      "color": "#ff0000"},
    {"range": "201–300", "category": "Very Unhealthy",                  "color": "#8f3f97"},
    {"range": "301+",    "category": "Hazardous",                      "color": "#7e0023"},
]

# ── Chart theme helper ────────────────────────────────────────────
def dark_layout(**kwargs):
    """Return a dict of dark-theme plotly layout overrides."""
    base = dict(
        paper_bgcolor="#1e2235",
        plot_bgcolor="#1e2235",
        font=dict(color="#e8eaed", family="Space Grotesk"),
        xaxis=dict(gridcolor="#2d3748", tickfont=dict(color="#a0aec0"), linecolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748", tickfont=dict(color="#a0aec0"), linecolor="#2d3748"),
        legend=dict(bgcolor="#252a3d", bordercolor="#2d3748", borderwidth=1, font=dict(color="#e8eaed")),
        margin=dict(t=50, b=40, l=40, r=30),
    )
    base.update(kwargs)
    return base

def section_title(text: str) -> str:
    return f"""<div style="font-size:0.72em;font-weight:700;letter-spacing:0.12em;
        text-transform:uppercase;color:#a0aec0;padding:6px 0 8px 0;
        border-bottom:1px solid #2d3748;margin:20px 0 14px 0">{text}</div>"""

# ================================================================
# CUSTOM CSS — DARK THEME
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;900&display=swap');

/* ── Global ── */
html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0f1117 !important;
    color: #e8eaed !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1a1d27 !important;
    border-right: 1px solid #2d3748;
}
[data-testid="stSidebar"] * { color: #e8eaed !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e0 !important; font-size: 0.93em; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] { background: transparent !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Main area headers ── */
h1, h2, h3 { color: #f0f4f8 !important; }
h1 { font-size: 1.8em !important; font-weight: 700 !important; }

/* ── Streamlit native widget overrides ── */
.stSlider [data-baseweb="slider"] { background: #2d3748; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    width: 100% !important; padding: 0.55rem 1rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stMetric { background: #1e2235; border-radius: 12px; padding: 14px !important; }
.stMetric label { color: #a0aec0 !important; font-size: 0.78em !important; text-transform: uppercase; letter-spacing: 0.08em; }
.stMetric [data-testid="metric-container"] > div:nth-child(2) { color: #f0f4f8 !important; font-size: 1.8em !important; font-weight: 700 !important; }
div[data-testid="stHorizontalBlock"] { gap: 12px; }

/* ── Alerts / info / error ── */
.stAlert { background: rgba(30,34,53,0.9) !important; border-radius: 12px !important; color: #e8eaed !important; }

/* ── Cards ── */
.aqi-card {
    background: linear-gradient(135deg, #1e2235, #252a3d);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 22px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    margin: 6px 0;
}
.aqi-big {
    font-size: 4em;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -2px;
}
.aqi-badge {
    display: inline-block;
    font-size: 0.82em;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-top: 8px;
    letter-spacing: 0.02em;
}
.forecast-card {
    background: linear-gradient(135deg, #1e2235, #252a3d);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 22px 16px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    height: 100%;
    transition: transform 0.2s ease;
}
.forecast-card:hover { transform: translateY(-4px); }
.forecast-aqi {
    font-size: 3.2em;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1;
}
.weather-chip {
    display: inline-block;
    background: rgba(45,55,72,0.8);
    border: 1px solid #3d4a5e;
    border-radius: 20px;
    padding: 5px 14px;
    margin: 3px 3px;
    font-size: 0.84em;
    font-weight: 500;
    color: #cbd5e0;
}
.weather-chip b { color: #e8eaed; }
.alert-banner {
    border-radius: 12px;
    padding: 14px 20px;
    margin: 10px 0;
    font-weight: 700;
    font-size: 1em;
    border-left: 5px solid;
    backdrop-filter: blur(4px);
}
.kpi-card {
    background: linear-gradient(135deg, #1e2235, #252a3d);
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.kpi-value {
    font-size: 2.4em;
    font-weight: 900;
    color: #f0f4f8;
    line-height: 1;
}
.kpi-label {
    font-size: 0.7em;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #a0aec0;
    margin-top: 6px;
}
.risk-card {
    background: linear-gradient(135deg, #1e2235, #252a3d);
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 16px 18px;
    margin: 5px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.pollutant-card {
    background: linear-gradient(135deg, #1e2235, #252a3d);
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.who-bar-track {
    background: #2d3748;
    border-radius: 6px;
    height: 8px;
    margin-top: 10px;
    overflow: hidden;
}
.info-banner {
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.35);
    border-radius: 12px;
    padding: 12px 18px;
    color: #93c5fd;
    font-size: 0.9em;
    margin: 10px 0;
}
.sidebar-logo {
    text-align: center;
    padding: 10px 0 6px 0;
}
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 5px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #1a1d27; }
::-webkit-scrollbar-thumb { background: #3d4a5e; border-radius: 4px; }

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 14px; overflow: hidden; }

/* ── Section title via markdown ── */
</style>
""", unsafe_allow_html=True)


# ================================================================
# INFERENCE ENGINE  (UNCHANGED)
# ================================================================

@st.cache_resource(show_spinner=False)
def get_hopsworks_project():
    """Connect to Hopsworks — cached for entire session."""
    import hopsworks
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
    )
    return project


@st.cache_resource(show_spinner=False)
def load_model_artifact():
    """Load model from Hopsworks registry (or local fallback). Cached for session."""
    try:
        project  = get_hopsworks_project()
        mr       = project.get_model_registry()
        hw_model = mr.get_best_model(
            MODEL_NAME,
            metric="test_weighted_rmse",
            direction="min"
        )
        model_dir = hw_model.download()
        pkl_files = list(Path(model_dir).glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError("No .pkl in downloaded model dir")
        artifact = joblib.load(pkl_files[0])
        return artifact, "Hopsworks Registry"
    except Exception as e:
        local = Path("model/aqi_multioutput_model.pkl")
        if local.exists():
            artifact = joblib.load(local)
            return artifact, "Local Fallback"
        raise RuntimeError(f"Model not found. Train first. Error: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def run_inference_and_get_payload():
    """
    Full inference pipeline — runs on dashboard load, re-runs every hour.
    Flow:
      1. Fetch latest feature row from Hopsworks Feature Store
      2. Run model prediction
      3. Return rich payload dict
    Predictions are also pushed back to 'aqi_predictions' feature group.
    """
    project  = get_hopsworks_project()
    artifact, source = load_model_artifact()

    fs  = project.get_feature_store()
    fg  = fs.get_feature_group(name="aqi_features", version=1)
    now = datetime.now(timezone.utc)

    df = None
    try:
        cutoff = now - timedelta(hours=6)
        df = (
            fg.select_all()
              .filter(fg.city == CITY)
              .filter(fg.timestamp >= cutoff)
              .read(read_options={"use_hive": True})
        )
    except Exception:
        pass

    if df is None or len(df) == 0:
        try:
            full = fg.read(read_options={"use_hive": True})
            full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
            df = full[full["city"] == CITY].sort_values("timestamp").tail(10)
        except Exception as e:
            raise RuntimeError(f"Cannot fetch features: {e}")

    if len(df) == 0:
        raise RuntimeError("No feature rows for Karachi. Run Feature Pipeline (nb1) first.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    latest = df.sort_values("timestamp").iloc[[-1]].reset_index(drop=True)

    feat_cols = artifact["feature_cols"]
    for c in feat_cols:
        if c not in latest.columns:
            latest[c] = 0.0
        latest[c] = pd.to_numeric(latest[c], errors="coerce").fillna(0.0)
    X = latest[feat_cols]

    model = artifact["model"]
    preds = np.clip(model.predict(X), 0, 500)[0]
    d1, d2, d3 = float(round(preds[0], 1)), float(round(preds[1], 1)), float(round(preds[2], 1))

    curr_aqi = float(latest["aqi"].iloc[0])
    ts       = latest["timestamp"].iloc[0]

    def enrich(aqi_val, day_offset):
        date_label = (now + timedelta(days=day_offset)).strftime("%A, %b %d")
        return {
            "aqi": aqi_val,
            "category": aqi_category(aqi_val),
            "color": aqi_color(aqi_val),
            "health_advice": health_advice(aqi_val),
            "date_label": date_label,
        }

    payload = {
        "generated_at":   now.isoformat(),
        "data_timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        "city":           CITY.title(),
        "pipeline_version": PIPELINE_VERSION,
        "model_source":   source,
        "best_model_name": artifact.get("model_name", ""),
        "current": {
            "aqi":         curr_aqi,
            "category":    aqi_category(curr_aqi),
            "color":       aqi_color(curr_aqi),
            "health_advice": health_advice(curr_aqi),
            "pm25":        _safe(latest, "pm25"),
            "pm10":        _safe(latest, "pm10"),
            "temperature": _safe(latest, "temperature"),
            "humidity":    _safe(latest, "humidity"),
            "wind_speed":  _safe(latest, "wind_speed"),
            "pressure":    _safe(latest, "pressure"),
            "o3":          _safe(latest, "o3"),
            "no2":         _safe(latest, "no2"),
            "so2":         _safe(latest, "so2") if "so2" in latest.columns else None,
            "co":          _safe(latest, "co") if "co" in latest.columns else None,
        },
        "forecast": {
            "day1": enrich(d1, 1),
            "day2": enrich(d2, 2),
            "day3": enrich(d3, 3),
        },
        "alert": None,
    }

    max_fc = max(d1, d2, d3)
    if max_fc > 300:
        payload["alert"] = {"level": "HAZARDOUS",      "message": "🚨 Hazardous AQI forecast — avoid all outdoor activity!"}
    elif max_fc > 200:
        payload["alert"] = {"level": "VERY_UNHEALTHY", "message": "⚠️ Very unhealthy AQI forecast — stay indoors."}
    elif max_fc > 150:
        payload["alert"] = {"level": "UNHEALTHY",      "message": "⚠️ Unhealthy AQI forecast — sensitive groups take care."}
    elif curr_aqi > 150:
        payload["alert"] = {"level": "CURRENT_HIGH",   "message": "⚠️ Current AQI is unhealthy — limit outdoor exposure."}

    try:
        pred_fg = fs.get_or_create_feature_group(
            name="aqi_predictions",
            version=1,
            primary_key=["city", "generated_at"],
            event_time="generated_at",
            online_enabled=True,
            stream=True,
            description=f"AQI 3-day forecasts for Karachi ({PIPELINE_VERSION})",
        )
        row_df = pd.DataFrame([{
            "city":          CITY,
            "generated_at":  pd.Timestamp(now),
            "current_aqi":   curr_aqi,
            "day1_aqi":      d1,
            "day1_category": aqi_category(d1),
            "day2_aqi":      d2,
            "day2_category": aqi_category(d2),
            "day3_aqi":      d3,
            "day3_category": aqi_category(d3),
            "has_alert":     float(1 if payload["alert"] else 0),
        }])
        pred_fg.insert(row_df, write_options={"wait_for_job": False})
    except Exception:
        pass

    return payload


def _safe(df, col):
    """Safely extract a float value from a DataFrame column."""
    if col in df.columns:
        val = df[col].iloc[0]
        return float(val) if pd.notna(val) else None
    return None


@st.cache_data(ttl=600, show_spinner=False)
def load_prediction_history():
    """Load historical predictions from Hopsworks aqi_predictions feature group."""
    try:
        project = get_hopsworks_project()
        fs      = project.get_feature_store()
        fg      = fs.get_feature_group(name="aqi_predictions", version=1)
        df      = fg.read(read_options={"use_hive": True})
        df["generated_at"] = pd.to_datetime(df["generated_at"], utc=True)
        return df.sort_values("generated_at").tail(50)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_backfill(n_days: int = 90) -> pd.DataFrame:
    path = Path(BACKFILL_CSV)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if n_days:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=n_days)
        df = df[df["timestamp"] >= cutoff]
    return df


# ================================================================
# RUN INFERENCE ON LOAD  (UNCHANGED)
# ================================================================
if "payload" not in st.session_state:
    with st.spinner("🔄 Connecting to Hopsworks and running inference..."):
        try:
            st.session_state["payload"] = run_inference_and_get_payload()
            st.session_state["inference_error"] = None
        except Exception as e:
            st.session_state["payload"] = None
            st.session_state["inference_error"] = str(e)
else:
    try:
        st.session_state["payload"] = run_inference_and_get_payload()
        st.session_state["inference_error"] = None
    except Exception as e:
        st.session_state["inference_error"] = str(e)


# ================================================================
# SIDEBAR — REDESIGNED
# ================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2em">🌫️</div>
        <div style="font-size:1.1em;font-weight:700;color:#f0f4f8;margin-top:4px">Pearls AQI</div>
        <div style="font-size:0.78em;color:#a0aec0;margin-top:2px">Karachi Forecaster</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2d3748;margin:12px 0'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📈 Forecast", "📊 Historical", "⚠️ Health Advisory", "🌬️ Pollutant Breakdown"],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#2d3748;margin:12px 0'>", unsafe_allow_html=True)

    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if "payload" in st.session_state:
            del st.session_state["payload"]
        st.rerun()

    # Live status indicator
    _pred_sidebar = st.session_state.get("payload")
    if _pred_sidebar:
        try:
            _dt = datetime.fromisoformat(_pred_sidebar["generated_at"])
            _age = datetime.now(timezone.utc) - _dt.replace(tzinfo=timezone.utc) if _dt.tzinfo is None else datetime.now(timezone.utc) - _dt
            _mins = int(_age.total_seconds() / 60)
            _age_str = f"{_mins}m ago" if _mins < 60 else f"{_mins//60}h ago"
            _model_src = _pred_sidebar.get('model_source', 'Registry')
            _model_name = _pred_sidebar.get('best_model_name', '')
            _model_line = (
                f"<br><div style=\"margin-top:6px;background:rgba(96,165,250,0.12);"
                f"border:1px solid rgba(96,165,250,0.3);border-radius:7px;"
                f"padding:5px 8px;display:inline-block\">"
                f"<span style=\"color:#94a3b8;font-size:0.65em;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.07em\">🏆 Best Model</span><br>"
                f"<span style=\"color:#60a5fa;font-size:0.82em;font-weight:700\">{_model_name}</span>"
                f"</div>"
            ) if _model_name else ""
            st.markdown(f"""
            <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);
                        border-radius:10px;padding:10px 14px;margin-top:10px;text-align:center">
                <span class="live-dot"></span>
                <span style="color:#22c55e;font-weight:700;font-size:0.85em">LIVE</span><br>
                <span style="color:#a0aec0;font-size:0.75em">Updated {_age_str}</span><br>
                <span style="color:#718096;font-size:0.7em">{_model_src}</span>{_model_line}
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass
    else:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
                    border-radius:10px;padding:8px 12px;margin-top:10px;text-align:center;
                    color:#f87171;font-size:0.82em;font-weight:600">
            ⚠️ Inference failed
        </div>
        """, unsafe_allow_html=True)

    # Footer pinned at bottom
    st.markdown("""
    <div style="position:fixed;bottom:16px;left:0;width:220px;text-align:center;
                color:#4a5568;font-size:0.68em;padding:0 12px">
        Pearls AQI v7 · Open-Meteo · Hopsworks
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# INFERENCE ERROR STATE  (UNCHANGED)
# ================================================================
err = st.session_state.get("inference_error")
if err and st.session_state.get("payload") is None:
    st.markdown(f"""
    <div class="alert-banner" style="background:rgba(239,68,68,0.1);border-color:#ef4444;color:#fca5a5">
        ❌ <b>Inference failed:</b> {err}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="info-banner">
        Make sure the Feature Pipeline has run at least once, Training Pipeline has saved a model,
        and HOPSWORKS_API_KEY is correct.
    </div>""", unsafe_allow_html=True)
    st.stop()

pred = st.session_state.get("payload", {}) or {}


# ================================================================
# PAGE: 🏠 HOME
# ================================================================
if page == "🏠 Home":
    st.markdown("<h1>🌫️ Karachi Air Quality Index</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0aec0;margin-top:-10px'>Real-time monitoring & 3-day AQI forecast powered by ML</p>", unsafe_allow_html=True)

    if not pred:
        st.error("No prediction data available."); st.stop()

    curr = pred.get("current", {})
    fc   = pred.get("forecast", {})

    # Alert banner
    if pred.get("alert"):
        alert = pred["alert"]
        lvl   = alert["level"]
        if "HAZARD" in lvl:
            bg, bc = "rgba(126,0,35,0.25)", "#7e0023"
        elif "VERY" in lvl:
            bg, bc = "rgba(143,63,151,0.25)", "#8f3f97"
        else:
            bg, bc = "rgba(255,126,0,0.2)", "#ff7e00"
        st.markdown(
            f'<div class="alert-banner" style="background:{bg};border-color:{bc};color:#f0f4f8">'
            f'{alert["message"]}</div>',
            unsafe_allow_html=True
        )

    c_aqi   = curr.get("aqi", 0)
    c_color = aqi_color(c_aqi)
    c_text  = aqi_text_color(c_aqi)

    col_main, col_weather = st.columns([1.6, 1])
    with col_main:
        st.markdown(f"""
        <div class="aqi-card" style="border-left:7px solid {c_color}">
            <div style="font-size:0.75em;color:#a0aec0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">
                Current AQI — Karachi
            </div>
            <div class="aqi-big" style="color:{c_color}">{int(c_aqi)}</div>
            <span class="aqi-badge" style="background:{c_color}22;color:{c_color};border:1px solid {c_color}55">
                {curr.get('category','')}
            </span>
            <div style="margin-top:14px;color:#a0aec0;font-size:0.88em;line-height:1.5">
                {curr.get('health_advice','')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_weather:
        st.markdown(section_title("Current Conditions"), unsafe_allow_html=True)
        def _fmt(v, unit=""):
            if v is None: return "N/A"
            return f"{v:.1f}{(' ' + unit) if unit else ''}"
        chips = [
            ("🌡️", "Temperature", _fmt(curr.get("temperature"), "°C")),
            ("💧", "Humidity",    _fmt(curr.get("humidity"), "%")),
            ("💨", "Wind",        _fmt(curr.get("wind_speed"), "m/s")),
            ("🔬", "PM2.5",       _fmt(curr.get("pm25"), "µg/m³")),
            ("🔬", "PM10",        _fmt(curr.get("pm10"), "µg/m³")),
            ("📊", "Pressure",    _fmt(curr.get("pressure"), "hPa")),
        ]
        chips_html = "".join(
            f'<span class="weather-chip">{icon} {lbl}: <b>{val}</b></span>'
            for icon, lbl, val in chips
        )
        st.markdown(f"<div style='line-height:2.2'>{chips_html}</div>", unsafe_allow_html=True)

    # 3-Day Forecast
    st.markdown(section_title("📅 3-Day Forecast"), unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns(3)
    for col, day_key, day_label in [(fcol1,"day1","Day 1"),(fcol2,"day2","Day 2"),(fcol3,"day3","Day 3")]:
        day_data = fc.get(day_key, {})
        d_aqi    = day_data.get("aqi", 0)
        d_color  = aqi_color(d_aqi)
        with col:
            st.markdown(f"""
            <div class="forecast-card" style="border-top:5px solid {d_color}">
                <div style="font-size:0.95em;font-weight:700;color:#cbd5e0;margin-bottom:2px">{day_label}</div>
                <div style="font-size:0.75em;color:#718096;margin-bottom:12px">{day_data.get('date_label','')}</div>
                <div class="forecast-aqi" style="color:{d_color}">{int(d_aqi)}</div>
                <div style="margin:10px 0">
                    <span class="aqi-badge" style="background:{d_color}22;color:{d_color};border:1px solid {d_color}55;font-size:0.75em">
                        {day_data.get('category','')}
                    </span>
                </div>
                <div style="font-size:0.75em;color:#718096;line-height:1.45;margin-top:8px">
                    {day_data.get('health_advice','')[:80]}…
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Gauge
    st.markdown(section_title("🎯 AQI Gauge"), unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=c_aqi,
        number={"font": {"size": 56, "color": c_color}},
        title={"text": "Current AQI — Karachi", "font": {"size": 15, "color": "#a0aec0"}},
        gauge={
            "axis":  {"range": [0, 500], "tickwidth": 1, "tickcolor": "#a0aec0"},
            "bar":   {"color": c_color, "thickness": 0.28},
            "bgcolor": "#1e2235",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   50],  "color": "rgba(0,228,0,0.15)"},
                {"range": [50,  100], "color": "rgba(255,255,0,0.15)"},
                {"range": [100, 150], "color": "rgba(255,126,0,0.15)"},
                {"range": [150, 200], "color": "rgba(255,0,0,0.15)"},
                {"range": [200, 300], "color": "rgba(143,63,151,0.15)"},
                {"range": [300, 500], "color": "rgba(126,0,35,0.15)"},
            ],
            "threshold": {"line": {"color": "#e8eaed", "width": 3}, "thickness": 0.75, "value": c_aqi},
        },
    ))
    fig_gauge.update_layout(
        height=300,
        paper_bgcolor="#1e2235",
        font=dict(color="#e8eaed", family="Space Grotesk"),
        margin=dict(t=30, b=10, l=30, r=30),
    )
    st.plotly_chart(fig_gauge, width='stretch')


# ================================================================
# PAGE: 📈 FORECAST
# ================================================================
elif page == "📈 Forecast":
    st.markdown("<h1>📈 AQI Forecast — Next 3 Days</h1>", unsafe_allow_html=True)

    if not pred:
        st.error("No prediction data."); st.stop()

    fc   = pred.get("forecast", {})
    curr = pred.get("current",  {})
    now  = datetime.now(timezone.utc)

    aqis   = [curr.get("aqi",0), fc.get("day1",{}).get("aqi",0),
              fc.get("day2",{}).get("aqi",0), fc.get("day3",{}).get("aqi",0)]
    labels = ["Now",
              fc.get("day1",{}).get("date_label","Day1"),
              fc.get("day2",{}).get("date_label","Day2"),
              fc.get("day3",{}).get("date_label","Day3")]
    colors_fc = [aqi_color(a) for a in aqis]

    fig = go.Figure()
    zones = [
        (0,   50,  "rgba(0,228,0,0.07)"),
        (50,  100, "rgba(255,255,0,0.07)"),
        (100, 150, "rgba(255,126,0,0.07)"),
        (150, 200, "rgba(255,0,0,0.07)"),
        (200, 300, "rgba(143,63,151,0.07)"),
        (300, 500, "rgba(126,0,35,0.07)"),
    ]
    for y0, y1, c in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=c, line_width=0)

    fig.add_trace(go.Scatter(
        x=labels, y=aqis, mode="lines+markers+text",
        line=dict(color="#60a5fa", width=3),
        marker=dict(size=16, color=colors_fc, line=dict(color="#1e2235", width=2)),
        text=[f"  {int(a)}" for a in aqis],
        textposition="top center",
        textfont=dict(size=13, color="#e8eaed"),
        name="AQI",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color="#ff7e00", line_width=1.5,
                  annotation_text="Moderate threshold", annotation_position="right",
                  annotation_font_color="#ff7e00")
    fig.add_hline(y=150, line_dash="dot", line_color="#ff0000", line_width=1.5,
                  annotation_text="Unhealthy threshold", annotation_position="right",
                  annotation_font_color="#ff0000")
    fig.update_layout(
        **dark_layout(
            title=dict(text="Karachi AQI — Current + 3-Day Forecast", font=dict(size=17, color="#f0f4f8")),
            xaxis_title="Date", yaxis_title="AQI",
            yaxis_range=[0, max(aqis)*1.4 if aqis else 300],
            height=420, showlegend=False,
        )
    )
    st.plotly_chart(fig, width='stretch')

    # Forecast detail cards
    st.markdown(section_title("Forecast Detail"), unsafe_allow_html=True)
    for day_key, label in [("day1","Day 1"), ("day2","Day 2"), ("day3","Day 3")]:
        d = fc.get(day_key, {})
        a = d.get("aqi", 0)
        c = aqi_color(a)
        st.markdown(f"""
        <div class="aqi-card" style="border-left:5px solid {c};padding:16px 22px;margin:8px 0">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
                <div>
                    <span style="font-size:1em;font-weight:700;color:#e8eaed">{label}</span>
                    <span style="color:#718096;font-size:0.85em;margin-left:10px">{d.get('date_label','')}</span>
                </div>
                <div>
                    <span style="font-size:1.6em;font-weight:900;color:{c}">{int(a)}</span>
                    <span class="aqi-badge" style="background:{c}22;color:{c};border:1px solid {c}55;
                          font-size:0.78em;margin-left:8px">{d.get('category','')}</span>
                </div>
            </div>
            <div style="color:#a0aec0;font-size:0.85em;margin-top:10px;padding-top:10px;
                        border-top:1px solid #2d3748">{d.get('health_advice','')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Forecast history
    st.markdown(section_title("📋 Forecast History"), unsafe_allow_html=True)
    hist = load_prediction_history()
    if len(hist) > 1:
        fig2 = go.Figure()
        color_map  = {"day1_aqi": "#60a5fa", "day2_aqi": "#fb923c", "day3_aqi": "#4ade80", "current_aqi": "#94a3b8"}
        names_map  = {"day1_aqi": "Day 1",   "day2_aqi": "Day 2",   "day3_aqi": "Day 3",   "current_aqi": "Current"}
        for col in ["current_aqi","day1_aqi","day2_aqi","day3_aqi"]:
            if col in hist.columns:
                fig2.add_trace(go.Scatter(
                    x=hist["generated_at"], y=hist[col],
                    mode="lines+markers", name=names_map[col],
                    line=dict(color=color_map[col], width=2),
                    marker=dict(size=5),
                ))
        fig2.update_layout(**dark_layout(
            title="Historical Forecast Runs (last 50)",
            xaxis_title="Run Time", yaxis_title="AQI",
            height=340,
        ))
        st.plotly_chart(fig2, width='stretch')
    else:
        st.markdown('<div class="info-banner">Forecast history will appear after multiple inference runs.</div>',
                    unsafe_allow_html=True)


# ================================================================
# PAGE: 📊 HISTORICAL
# ================================================================
elif page == "📊 Historical":
    st.markdown("<h1>📊 Historical AQI Analysis — Karachi</h1>", unsafe_allow_html=True)

    days_back = st.slider("Show last N days", 7, 365, 90)
    df = load_backfill(n_days=days_back)

    if df.empty:
        st.error(f"Historical data not found at '{BACKFILL_CSV}'. Run Backfill Pipeline first.")
        st.stop()

    st.markdown(
        f'<div class="info-banner">📊 Showing {len(df):,} hourly records from '
        f'{df["timestamp"].min().date()} to {df["timestamp"].max().date()}</div>',
        unsafe_allow_html=True
    )

    # KPI cards
    st.markdown(section_title("Key Metrics"), unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        (k1, f"{df['aqi'].mean():.1f}", "Average AQI"),
        (k2, f"{df['aqi'].max():.0f}",  "Peak AQI"),
        (k3, f"{100*(df['aqi'] <= 50).mean():.1f}%", "Good Days"),
        (k4, f"{100*(df['aqi'] > 150).mean():.1f}%", "Unhealthy Days"),
    ]
    for col, value, label in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Time series
    st.markdown(section_title("AQI Over Time"), unsafe_allow_html=True)
    step = max(1, len(df)//3000)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"][::step], y=df["aqi"][::step],
        mode="lines", name="AQI",
        line=dict(color="#60a5fa", width=1.2), opacity=0.8,
    ))
    if "aqi_rolling_mean_24h" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"][::step], y=df["aqi_rolling_mean_24h"][::step],
            mode="lines", name="24h Rolling Mean",
            line=dict(color="#fb923c", width=2.5),
        ))
    fig.update_layout(**dark_layout(xaxis_title="Date", yaxis_title="AQI", height=380))
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(section_title("Hourly Pattern"), unsafe_allow_html=True)
        if "hour" in df.columns:
            hourly = df.groupby("hour")["aqi"].mean().reset_index()
            fig_h  = px.bar(hourly, x="hour", y="aqi", color="aqi",
                            color_continuous_scale="YlOrRd",
                            labels={"hour":"Hour of Day","aqi":"Avg AQI"},
                            title="Average AQI by Hour")
            fig_h.update_layout(**dark_layout(height=320, showlegend=False, title="Average AQI by Hour"))
            fig_h.update_coloraxes(colorbar_tickfont_color="#a0aec0")
            st.plotly_chart(fig_h, width='stretch')

    with col_b:
        st.markdown(section_title("Monthly Pattern"), unsafe_allow_html=True)
        if "month" in df.columns:
            monthly  = df.groupby("month")["aqi"].mean().reset_index()
            month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly["month_name"] = monthly["month"].map(month_map)
            fig_m = px.bar(monthly, x="month_name", y="aqi", color="aqi",
                           color_continuous_scale="YlOrRd",
                           labels={"month_name":"Month","aqi":"Avg AQI"},
                           title="Average AQI by Month")
            fig_m.update_layout(**dark_layout(height=320, showlegend=False, title="Average AQI by Month"))
            fig_m.update_coloraxes(colorbar_tickfont_color="#a0aec0")
            st.plotly_chart(fig_m, width='stretch')

    # Pie
    st.markdown(section_title("AQI Category Breakdown"), unsafe_allow_html=True)
    cat_bins   = [0, 50, 100, 150, 200, 300, float("inf")]
    cat_labels = ["Good","Moderate","Unhealthy/Sensitive","Unhealthy","Very Unhealthy","Hazardous"]
    df["category"] = pd.cut(df["aqi"], bins=cat_bins, labels=cat_labels)
    cat_counts     = df["category"].value_counts().sort_index()
    fig_pie = go.Figure(go.Pie(
        labels=cat_counts.index, values=cat_counts.values,
        marker_colors=["#00e400","#ffff00","#ff7e00","#ff0000","#8f3f97","#7e0023"],
        hole=0.42, textinfo="label+percent",
        textfont=dict(color="#e8eaed"),
    ))
    fig_pie.update_layout(
        height=360, paper_bgcolor="#1e2235",
        font=dict(color="#e8eaed", family="Space Grotesk"),
        legend=dict(bgcolor="#252a3d", bordercolor="#2d3748", borderwidth=1, font=dict(color="#e8eaed")),
        margin=dict(t=30, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pie, width='stretch')

    # Correlation
    st.markdown(section_title("Weather–AQI Correlation"), unsafe_allow_html=True)
    weather_cols = [c for c in ["temperature","humidity","wind_speed","pressure","precipitation"] if c in df.columns]
    corrs   = {c: df[c].corr(df["aqi"]) for c in weather_cols}
    corr_df = pd.DataFrame({"Feature": list(corrs.keys()), "Correlation": list(corrs.values())})
    corr_df = corr_df.sort_values("Correlation", key=abs, ascending=True)
    fig_c   = px.bar(corr_df, x="Correlation", y="Feature", orientation="h",
                     color="Correlation", color_continuous_scale="RdBu_r",
                     range_color=[-1,1], title="Pearson Correlation with AQI")
    fig_c.update_layout(**dark_layout(height=320, title="Pearson Correlation with AQI"))
    fig_c.update_coloraxes(colorbar_tickfont_color="#a0aec0")
    st.plotly_chart(fig_c, width='stretch')


# ================================================================
# PAGE: ⚠️ HEALTH ADVISORY
# ================================================================
elif page == "⚠️ Health Advisory":
    st.markdown("<h1>⚠️ Health Advisory</h1>", unsafe_allow_html=True)

    if not pred:
        st.error("No prediction data."); st.stop()

    curr    = pred.get("current", {})
    fc      = pred.get("forecast", {})
    c_aqi   = curr.get("aqi", 0)
    c_color = aqi_color(c_aqi)

    # Current AQI mini-banner
    st.markdown(f"""
    <div class="aqi-card" style="border-left:6px solid {c_color};padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:18px">
        <div class="aqi-big" style="color:{c_color};font-size:2.8em">{int(c_aqi)}</div>
        <div>
            <span class="aqi-badge" style="background:{c_color}22;color:{c_color};border:1px solid {c_color}55">{aqi_category(c_aqi)}</span>
            <div style="color:#a0aec0;font-size:0.85em;margin-top:8px">{health_advice(c_aqi)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Who is at risk today? ────────────────────────────────────────
    st.markdown(section_title("👥 Who is at Risk Today?"), unsafe_allow_html=True)

    RISK_GROUPS = [
        ("🧒", "Children",          c_aqi > 50,  "Limit outdoor play above AQI 50.", "Avoid outdoor activities above AQI 100."),
        ("👴", "Elderly (65+)",     c_aqi > 50,  "Reduce strenuous outdoor activity.", "Stay indoors; use air purifier."),
        ("❤️", "Heart Conditions",  c_aqi > 50,  "Limit exertion; monitor symptoms.", "Rest indoors; consult doctor if symptomatic."),
        ("🫁", "Respiratory Issues",c_aqi > 50,  "Carry inhaler; limit outdoor time.", "Stay indoors; avoid all outdoor exposure."),
        ("🏃", "Athletes",          c_aqi > 100, "Shorten intense training sessions.", "Move training indoors today."),
        ("🤰", "Pregnant Women",    c_aqi > 50,  "Minimize outdoor exposure.", "Stay indoors; open windows only when AQI improves."),
    ]

    g_cols = st.columns(2)
    for i, (icon, group, at_risk, safe_advice, risk_advice) in enumerate(RISK_GROUPS):
        badge_color = "#ef4444" if at_risk else "#22c55e"
        badge_bg    = "rgba(239,68,68,0.15)" if at_risk else "rgba(34,197,94,0.15)"
        badge_text  = "At Risk" if at_risk else "Safe"
        advice_text = risk_advice if at_risk else safe_advice
        border_color = "#ef4444" if at_risk else "#22c55e"
        with g_cols[i % 2]:
            st.markdown(f"""
            <div class="risk-card" style="border-left:4px solid {border_color}">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
                    <span style="font-weight:700;color:#e8eaed;font-size:0.97em">{icon} {group}</span>
                    <span style="background:{badge_bg};color:{badge_color};border:1px solid {badge_color}55;
                          border-radius:14px;padding:2px 12px;font-size:0.75em;font-weight:700">{badge_text}</span>
                </div>
                <div style="color:#a0aec0;font-size:0.82em">{advice_text}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Activity Guide ───────────────────────────────────────────────
    st.markdown(section_title("🏃 Activity Guide"), unsafe_allow_html=True)

    def _activity_rec(aqi, good_limit, caution_limit, avoid_limit):
        if aqi <= good_limit:
            return "#22c55e", "rgba(34,197,94,0.12)", "✅ Go ahead"
        elif aqi <= caution_limit:
            return "#facc15", "rgba(250,204,21,0.12)", "⚠️ With caution"
        elif aqi <= avoid_limit:
            return "#fb923c", "rgba(251,146,60,0.12)", "🟠 Limit duration"
        else:
            return "#ef4444", "rgba(239,68,68,0.12)", "🚫 Avoid today"

    ACTIVITIES = [
        ("🏃", "Outdoor Running",  50,  100, 150),
        ("🚴", "Cycling",          50,  100, 150),
        ("🧘", "Yoga / Park",      100, 150, 200),
        ("🚌", "Commuting",        150, 200, 300),
        ("🌱", "Gardening",        50,  100, 150),
        ("🏫", "School PE",        50,  100, 150),
    ]

    a_cols = st.columns(3)
    for i, (icon, act, good_lim, caution_lim, avoid_lim) in enumerate(ACTIVITIES):
        a_color, a_bg, a_label = _activity_rec(c_aqi, good_lim, caution_lim, avoid_lim)
        with a_cols[i % 3]:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {a_color};padding:16px 14px">
                <div style="font-size:1.6em">{icon}</div>
                <div style="font-size:0.9em;font-weight:700;color:#e8eaed;margin-top:6px">{act}</div>
                <div style="margin-top:8px;background:{a_bg};border-radius:10px;padding:5px 10px;
                            font-size:0.8em;font-weight:700;color:{a_color}">{a_label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── 3-Day Advisory Timeline ──────────────────────────────────────
    st.markdown(section_title("📅 3-Day Advisory Timeline"), unsafe_allow_html=True)

    for day_key, day_label in [("day1","Day 1"),("day2","Day 2"),("day3","Day 3")]:
        d     = fc.get(day_key, {})
        d_aqi = d.get("aqi", 0)
        d_col = aqi_color(d_aqi)
        st.markdown(f"""
        <div class="aqi-card" style="border-left:5px solid {d_col};padding:14px 20px;margin:8px 0;
             display:flex;align-items:center;gap:18px;flex-wrap:wrap">
            <div style="min-width:80px">
                <div style="font-weight:700;color:#e8eaed">{day_label}</div>
                <div style="font-size:0.75em;color:#718096">{d.get('date_label','')}</div>
            </div>
            <div style="font-size:2.2em;font-weight:900;color:{d_col}">{int(d_aqi)}</div>
            <div>
                <span class="aqi-badge" style="background:{d_col}22;color:{d_col};border:1px solid {d_col}55;font-size:0.78em">
                    {d.get('category','')}
                </span>
                <div style="font-size:0.82em;color:#a0aec0;margin-top:8px">{d.get('health_advice','')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Protective Measures ──────────────────────────────────────────
    st.markdown(section_title("🛡️ Protective Measures"), unsafe_allow_html=True)

    MEASURES = [
        (50,  "😷", "Wear a Mask", "Use a surgical or N95 mask when outdoors.", "#60a5fa"),
        (100, "🪟", "Close Windows", "Keep windows and doors closed to prevent indoor pollution.", "#facc15"),
        (150, "🏠", "Stay Indoors", "Limit outdoor activities; work from home if possible.", "#fb923c"),
        (200, "🏥", "Seek Medical Advice", "If you experience symptoms, contact a healthcare provider immediately.", "#ef4444"),
    ]

    active_measures = [(icon, title, desc, color) for threshold, icon, title, desc, color in MEASURES if c_aqi > threshold]

    if not active_measures:
        st.markdown("""
        <div class="aqi-card" style="border-left:5px solid #22c55e;padding:16px 20px;text-align:center">
            <div style="font-size:1.5em">✅</div>
            <div style="font-weight:700;color:#22c55e;font-size:1em;margin-top:6px">All Clear</div>
            <div style="color:#a0aec0;font-size:0.85em;margin-top:6px">Air quality is good. No special precautions needed today.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        m_cols = st.columns(min(len(active_measures), 4))
        for i, (icon, title, desc, color) in enumerate(active_measures):
            with m_cols[i]:
                st.markdown(f"""
                <div class="kpi-card" style="border-top:3px solid {color}">
                    <div style="font-size:1.8em">{icon}</div>
                    <div style="font-weight:700;color:#e8eaed;font-size:0.9em;margin-top:8px">{title}</div>
                    <div style="font-size:0.78em;color:#a0aec0;margin-top:8px;line-height:1.5">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ================================================================
# PAGE: 🌬️ POLLUTANT BREAKDOWN
# ================================================================
elif page == "🌬️ Pollutant Breakdown":
    st.markdown("<h1>🌬️ Pollutant Breakdown</h1>", unsafe_allow_html=True)

    days_p = st.slider("Show last N days", 7, 180, 30, key="pollutant_days")
    df_p   = load_backfill(n_days=days_p)

    curr = pred.get("current", {}) if pred else {}

    if df_p.empty:
        st.markdown('<div class="info-banner">Historical data not found. Run Backfill Pipeline first.</div>',
                    unsafe_allow_html=True)

    # WHO 24h limits
    WHO = {"pm25": 15, "pm10": 45, "no2": 25, "o3": 100, "so2": 40, "co": 4000}
    POLLUTANT_META = {
        "pm25": ("PM2.5",  "µg/m³", "Fine particles — penetrate deep into lungs"),
        "pm10": ("PM10",   "µg/m³", "Coarse particles — irritate airways"),
        "no2":  ("NO₂",   "µg/m³", "Nitrogen dioxide — traffic & combustion"),
        "o3":   ("O₃",    "µg/m³", "Ground-level ozone — sunlight + pollution"),
        "so2":  ("SO₂",   "µg/m³", "Sulfur dioxide — fossil fuels"),
        "co":   ("CO",    "µg/m³", "Carbon monoxide — incomplete combustion"),
    }

    # ── WHO KPI cards ────────────────────────────────────────────────
    st.markdown(section_title("📊 Current Levels vs WHO 24h Guidelines"), unsafe_allow_html=True)

    p_cols = st.columns(3)
    for i, (key, (name, unit, _)) in enumerate(POLLUTANT_META.items()):
        current_val = curr.get(key)
        who_limit   = WHO[key]
        with p_cols[i % 3]:
            if current_val is not None:
                pct = min((current_val / who_limit) * 100, 200)
                if pct < 75:
                    bar_color, status_color, status = "#22c55e", "#22c55e", "OK"
                elif pct < 100:
                    bar_color, status_color, status = "#facc15", "#facc15", "Near Limit"
                else:
                    bar_color, status_color, status = "#ef4444", "#ef4444", "Exceeds WHO"
                fill_w = min(pct, 100)
                st.markdown(f"""
                <div class="pollutant-card">
                    <div style="font-size:0.75em;color:#a0aec0;text-transform:uppercase;letter-spacing:0.08em">{name}</div>
                    <div style="font-size:2em;font-weight:900;color:#f0f4f8;margin:6px 0">
                        {current_val:.1f}<span style="font-size:0.42em;color:#718096;font-weight:400"> {unit}</span>
                    </div>
                    <div style="font-size:0.72em;color:{status_color};font-weight:700">{status}</div>
                    <div style="font-size:0.68em;color:#4a5568;margin-top:2px">WHO limit: {who_limit} {unit}</div>
                    <div class="who-bar-track">
                        <div style="width:{fill_w:.0f}%;height:100%;background:{bar_color};border-radius:6px;transition:width 0.5s"></div>
                    </div>
                    <div style="font-size:0.68em;color:#718096;margin-top:4px">{pct:.0f}% of WHO limit</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pollutant-card">
                    <div style="font-size:0.75em;color:#a0aec0;text-transform:uppercase">{name}</div>
                    <div style="font-size:1.4em;color:#4a5568;margin:10px 0">N/A</div>
                    <div style="font-size:0.68em;color:#4a5568">WHO limit: {who_limit} {unit}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Pollutant Trends ─────────────────────────────────────────────
    if not df_p.empty:
        st.markdown(section_title("📈 Pollutant Trends"), unsafe_allow_html=True)

        trend_cols = st.columns(2)
        _plot_idx  = 0
        for key, (name, unit, desc) in POLLUTANT_META.items():
            if key not in df_p.columns:
                continue
            step   = max(1, len(df_p)//1500)
            who_lim = WHO[key]
            fig_t  = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=df_p["timestamp"][::step], y=df_p[key][::step],
                mode="lines", name=name,
                line=dict(color="#60a5fa", width=1.5),
                fill="tozeroy", fillcolor="rgba(96,165,250,0.08)",
            ))
            fig_t.add_hline(y=who_lim, line_dash="dot", line_color="#facc15", line_width=1.5,
                            annotation_text=f"WHO {who_lim}", annotation_position="right",
                            annotation_font_color="#facc15")
            fig_t.update_layout(**dark_layout(
                title=f"{name} ({unit})",
                height=240,
                margin=dict(t=40, b=30, l=40, r=30),
                showlegend=False,
            ))
            with trend_cols[_plot_idx % 2]:
                st.plotly_chart(fig_t, width='stretch')
            _plot_idx += 1

        # ── Correlation with AQI ─────────────────────────────────────
        st.markdown(section_title("🔗 Correlation with AQI"), unsafe_allow_html=True)
        poll_cols  = [k for k in POLLUTANT_META if k in df_p.columns]
        poll_corrs = {k: df_p[k].corr(df_p["aqi"]) for k in poll_cols}
        pc_df      = pd.DataFrame({"Pollutant": [POLLUTANT_META[k][0] for k in poll_cols],
                                   "Correlation": [poll_corrs[k] for k in poll_cols]})
        pc_df      = pc_df.sort_values("Correlation", key=abs, ascending=True)
        bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in pc_df["Correlation"]]
        fig_pc     = go.Figure(go.Bar(
            x=pc_df["Correlation"], y=pc_df["Pollutant"], orientation="h",
            marker_color=bar_colors, text=[f"{v:.2f}" for v in pc_df["Correlation"]],
            textposition="outside", textfont=dict(color="#e8eaed"),
        ))
        fig_pc.update_layout(**dark_layout(
            title="Pearson Correlation of Pollutants with AQI",
            xaxis=dict(range=[-1,1], gridcolor="#2d3748", tickfont=dict(color="#a0aec0"), linecolor="#2d3748"),
            height=300,
        ))
        st.plotly_chart(fig_pc, width='stretch')

    # ── AQI Reference Scale ──────────────────────────────────────────
    st.markdown(section_title("📋 AQI Reference Scale"), unsafe_allow_html=True)
    ref_cols = st.columns(6)
    for i, scale in enumerate(AQI_SCALE):
        color  = scale["color"]
        tcolor = "#111" if scale["range"] in ["0–50","51–100"] else "#fff"
        with ref_cols[i]:
            st.markdown(f"""
            <div style="background:{color}22;border:1px solid {color}66;border-radius:12px;
                        padding:12px 8px;text-align:center">
                <div style="font-size:0.75em;font-weight:700;color:{color}">{scale['range']}</div>
                <div style="font-size:0.65em;color:#a0aec0;margin-top:4px;line-height:1.3">{scale['category']}</div>
            </div>
            """, unsafe_allow_html=True)
