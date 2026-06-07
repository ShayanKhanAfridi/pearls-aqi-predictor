# 🌫️ Pearls AQI Predictor

> **End-to-End MLOps System** — 3-Day Air Quality Index Forecasting for Karachi, Pakistan

## 🚀 Live Dashboard

# 🌐 [pearls-aqi-predictor-by-shayan.streamlit.app](https://pearls-aqi-predictor-by-shayan.streamlit.app/)

> Click above to open the live, interactive AQI forecasting dashboard for Karachi — updated every hour, no login required.

---

[![Hourly Feature Pipeline](https://github.com/ShayanKhanAfridi/pearls-aqi-predictor/actions/workflows/hourly_pipeline.yml/badge.svg)](https://github.com/ShayanKhanAfridi/pearls-aqi-predictor/actions/workflows/hourly_pipeline.yml)
[![Daily Training Pipeline](https://github.com/ShayanKhanAfridi/pearls-aqi-predictor/actions/workflows/daily_training.yml/badge.svg)](https://github.com/ShayanKhanAfridi/pearls-aqi-predictor/actions/workflows/daily_training.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Hopsworks](https://img.shields.io/badge/Hopsworks-4.7.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)

---

## What is this?

Pearls AQI Predictor is a fully automated, production-grade MLOps pipeline that forecasts the Air Quality Index (AQI) for Karachi for the **next 3 days (72 hours)**. It collects live environmental data every hour, trains and compares five machine learning models every day, and surfaces predictions through an interactive dark-themed Streamlit dashboard — all without any manual intervention.

The system uses a **multi-output direct regression** architecture: a single model simultaneously predicts Day 1, Day 2, and Day 3 AQI as three separate outputs. This eliminates the error-compounding problem of recursive/chained forecasting approaches.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    Open-Meteo APIs (Free, No Key)                  │
│         Weather Forecast API   ·   Air Quality API                 │
└────────────────────┬──────────────────────────┬───────────────────┘
                     │                          │
          ┌──────────▼──────────┐   ┌──────────▼──────────┐
          │  Backfill Pipeline  │   │   Feature Pipeline  │
          │  (3-Year History)   │   │  (Live · Every Hour)│
          │  [Run Once, Colab]  │   │  [GitHub Actions]   │
          └──────────┬──────────┘   └──────────┬──────────┘
                     │                          │
                     └────────────┬─────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   Hopsworks Feature Store   │
                    │    Feature Group: v1        │
                    │   ~42 engineered features   │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │     Training Pipeline       │
                    │  Runs Daily @ 02:00 UTC     │
                    │  5 candidates → best model  │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  Hopsworks Model Registry   │
                    │  aqi_predictor_multioutput  │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │     Streamlit Dashboard     │
                    │  Live inference on load     │
                    │  Home · Forecast · History  │
                    │  Health Advisory · Pollutants│
                    └─────────────────────────────┘
```

---

## 📁 Project Structure

```
pearls-aqi-predictor/
├── README.md
├── requirements.txt                    ← Python 3.11 only
├── .env.example                        ← Credentials template
├── run_dashboard.ps1                   ← Windows launcher
│
├── dashboard/
│   └── app.py                          ← Streamlit app + inference engine
│
├── pipelines/
│   ├── backfill_pipeline.py            ← 3-year historical ingestion (run once)
│   ├── feature_pipeline.py             ← Hourly live feature computation (v6)
│   └── training_pipeline.py            ← 5-model training + selection + registry (v7)
│
├── notebooks/
│   ├── eda.ipynb                       ← Exploratory Data Analysis
│   ├── shap_analysis.ipynb             ← SHAP feature importance
│   └── inference_debug.py              ← Isolated inference testing
│
├── .github/workflows/
│   ├── hourly_pipeline.yml             ← GitHub Actions: every hour
│   └── daily_training.yml              ← GitHub Actions: daily at 02:00 UTC
│
├── data/
│   ├── raw/                            ← Cached raw API responses
│   └── engineered/                     ← Cached engineered feature CSVs
│
└── model/
    └── aqi_multioutput_model.pkl       ← Local fallback model artifact
```

---

## 🧠 Machine Learning Engine

### Architecture: Multi-Output Direct Regression

Rather than recursively predicting one hour at a time (which compounds errors), the system trains a single model to simultaneously output three targets:

| Target | Horizon | Construction |
|--------|---------|--------------|
| `target_day1` | 0–24h | Simple mean of AQI over hours +1 to +24 |
| `target_day2` | 24–48h | Gaussian-weighted mean over hours +25 to +48 |
| `target_day3` | 48–72h | Gaussian-weighted mean over hours +49 to +72 |

Gaussian weighting (σ=8.0) smooths the Day 2 and Day 3 targets, reducing noise from anomalous individual hours at window edges.

### Five Candidate Algorithms

| Algorithm | Type | Key Hyperparameters |
|-----------|------|---------------------|
| **Ridge Regression** | Linear (baseline) | StandardScaler + MultiOutputRegressor, α=10.0 |
| **Random Forest** | Tree Ensemble | 400 trees, max_depth=6, min_samples_leaf=30 |
| **XGBoost** | Gradient Boosting | 600 estimators, max_depth=4, lr=0.03, λ=5.0 |
| **LightGBM** | Gradient Boosting | 600 estimators, num_leaves=20, lr=0.03, λ=5.0 |
| **Gradient Boosting** | Sequential Boosting | 300 estimators, max_depth=3, lr=0.04 |

### Model Selection: Weighted RMSE

The best model is selected by **Weighted RMSE**, which penalizes long-horizon errors more heavily:

```
Weighted RMSE = 0.20 × Day1_RMSE + 0.35 × Day2_RMSE + 0.45 × Day3_RMSE
```

Day 3 carries the highest weight (0.45) because a 72-hour forecast that fails is less useful than no forecast at all. An **overfitting guard** disqualifies any model where the train-to-validation RMSE gap exceeds 30% on any single day.

### Data Splits

| Split | Size | Purpose |
|-------|------|---------|
| Train | 70% | Fit model parameters |
| Validation | 15% | Model selection + overfitting check |
| Test | 15% | Final unbiased evaluation (registry metrics) |
| Train+Val | 85% | Final model retraining after selection |

---

## 🔧 Feature Engineering (42 Features)

### Lag Features (7)
Historical AQI values at 1h, 3h, 6h, 12h, 24h, 48h, and 168h (1 week) ago.

### Rolling Statistics (8)
Rolling means over 6h, 24h, 72h, 168h windows; rolling std over 24h and 72h; rolling max and min over 24h.

### Cyclical Time Encoding (8)
Sine/Cosine pairs for hour-of-day, month, day-of-week, and day-of-year. Ensures the model sees time as a continuous cycle rather than discrete integers.

### Wind Vectors (2)
`wind_x = wind_speed × cos(direction)` and `wind_y = wind_speed × sin(direction)` — converts polar wind to Cartesian for tree-based models.

### Regime and Residual Features (3) — added in v6
- `aqi_residual_168h`: deviation of current AQI from 7-day rolling mean
- `aqi_residual_72h`: deviation from 3-day rolling mean
- `aqi_regime_range`: rolling_max_24h − rolling_min_24h

### Interaction Features (4)
`wind_x_pm25`, `humidity_x_pm25`, `temp_x_aqi`, `pollution_index` (composite PM2.5+PM10+NO₂ load).

### Pollutant and Weather Inputs (10)
Raw PM2.5, PM10, O₃, NO₂, SO₂, CO; temperature, humidity, pressure, cloud cover, precipitation, current and forecast wind speed.

### Calendar Flags (3)
`is_weekend`, `is_winter` (Nov–Feb), `is_monsoon` (Jul–Sep).

---

## 📊 Dashboard

Five pages accessible from the sidebar:

| Page | What you see |
|------|-------------|
| 🏠 **Home** | Current AQI, health category, meteorological conditions, immediate health advice |
| 📈 **Forecast** | Day 1 / Day 2 / Day 3 AQI prediction cards + Plotly trend charts |
| 📊 **Historical** | Historical AQI trends, distribution histograms, rolling averages |
| ⚠️ **Health Advisory** | Detailed guidance by AQI level for sensitive groups, general population |
| 🌬️ **Pollutant Breakdown** | PM2.5, PM10, O₃, NO₂, SO₂, CO levels vs. WHO guideline thresholds |

Inference runs on every dashboard load and is cached for 1 hour (`st.cache_data ttl=3600`). The dashboard first attempts to load the model from the Hopsworks Model Registry and falls back to the local `model/aqi_multioutput_model.pkl` if unavailable.

---

## ⚙️ Setup — Local Development (Windows)

> **Python 3.11 is required.** Hopsworks is incompatible with Python 3.12+.

### Step 1: Clone

```powershell
git clone https://github.com/ShayanKhanAfridi/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### Step 2: Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If you get an ExecutionPolicy error:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 3: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Credentials

```powershell
copy .env.example .env
```

Edit `.env`:
```env
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
```

### Step 5: Launch Dashboard

```powershell
.\run_dashboard.ps1
```

Open `http://localhost:8501`

---

## 🚀 Running the Backfill Pipeline (Google Colab)

The backfill pipeline ingests 3 years of historical data — this is computationally heavy and should be run on Colab or a remote notebook, not locally.

```python
# Cell 1
!pip install hopsworks==4.7.* requests pandas numpy python-dotenv openmeteo-requests requests-cache retry-requests

# Cell 2
!pip install "hopsworks[python]" confluent-kafka --quiet

# Cell 3
import os
os.environ["HOPSWORKS_API_KEY"] = "your-api-key-here"

# Cell 4 — paste full contents of pipelines/backfill_pipeline.py and run
```

---

## 🏃 Running Pipelines Locally

After the backfill is complete:

```powershell
# Live feature ingestion
python pipelines/feature_pipeline.py

# Model training and selection
python pipelines/training_pipeline.py
```

> **Note:** On Windows, HDFS/Delta Lake writes to Hopsworks are skipped automatically (known limitation). Feature computation runs and is logged to console. Full cloud writes happen in GitHub Actions on Linux runners.

---

## ⏱️ CI/CD Pipeline Schedule

| Workflow | Trigger | Action |
|----------|---------|--------|
| `hourly_pipeline.yml` | Every hour at `:00 UTC` | Fetch live AQI + weather, compute 42 features, push to Feature Store |
| `daily_training.yml` | Every day at `02:00 UTC` | Fetch features, train 5 models, select best by Weighted RMSE, register in Hopsworks |

Both workflows can also be triggered manually from the GitHub Actions UI (`workflow_dispatch`).

---

## 🔍 Explainability

`notebooks/shap_analysis.ipynb` uses SHAP (SHapley Additive exPlanations) to explain predictions. Key findings:

1. `aqi_lag1` — strongest predictor (recent AQI momentum)
2. `aqi_rolling_mean_24h` — current baseline pollution level
3. `aqi_lag168` — weekly seasonality signal (same time last week)
4. `aqi_residual_168h` — anomaly signal vs. recent baseline
5. `pm25` — raw particulate matter concentration
6. `hour_sin` / `hour_cos` — time-of-day cycle
7. `future_wind_24h` — forecasted wind (most impactful for Day 3)

---

## 📋 Requirements

Core dependencies from `requirements.txt`:

```
hopsworks==4.7.5
deltalake>=0.17.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
openmeteo-requests>=1.2.0
requests-cache>=1.1.0
retry-requests>=2.0.0
shap>=0.44.0
plotly>=5.18.0
streamlit>=1.31.0
python-dotenv>=1.0.0
joblib>=1.3.0
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

Please ensure your code runs with Python 3.11 and that any new features added to the feature pipeline are also reflected in `FEATURE_COLS` in the training pipeline.

---

## 👤 Author

**Shayan Khan Afridi**  
GitHub: [@ShayanKhanAfridi](https://github.com/ShayanKhanAfridi)  
Repository: [pearls-aqi-predictor](https://github.com/ShayanKhanAfridi/pearls-aqi-predictor)

---

*Built with ❤️ for Karachi*
