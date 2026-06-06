# 🌫️ Pearls AQI Predictor — Karachi (MLOps v7)

An end-to-end **MLOps forecasting system** that predicts the Air Quality Index (AQI) for **Karachi** for the next **3 days**. It features a serverless architecture utilizing **Hopsworks Feature Store & Model Registry**, live data ingestion via **Open-Meteo**, and a stunning **Streamlit dashboard** running automatic batch inference.

---

## 🏗️ Project Architecture

```
Open-Meteo API  ──►  Feature Pipeline  ──►  Hopsworks Feature Store
                      (hourly, GitHub Actions)        │
                                                      │
Open-Meteo Archive ──► Backfill Pipeline ──►  ────────┘
     (3 years, once)

Hopsworks Feature Store  ──►  Training Pipeline  ──►  Hopsworks Model Registry
                               (daily, GitHub Actions)

Hopsworks Feature Store  ──►  Dashboard (app.py)  ──►  Live 3-Day AQI Forecast
Hopsworks Model Registry ──►  (inference on load, auto-refresh every hour)
```

---

## 📁 File Structure

```
pearls-aqi-predictor/
│
├── README.md                           ← Setup & execution guide
├── requirements.txt                    ← Package dependencies
├── .env.example                        ← Template for Hopsworks API Key
├── .env                                ← Local credentials (ignored by Git)
├── run_dashboard.ps1                   ← Windows PowerShell launch shortcut
│
├── dashboard/
│   └── app.py                          ← Streamlit dashboard (inference engine inside)
│
├── pipelines/
│   ├── __init__.py
│   ├── backfill_pipeline.py            ← 3-year historical data backfill
│   ├── feature_pipeline.py             ← Live hourly data collection + features
│   └── training_pipeline.py            ← Model training, validation, & registry
│
├── notebooks/
│   ├── eda.ipynb                       ← Exploratory Data Analysis
│   ├── shap_analysis.ipynb             ← SHAP explainability analysis
│   └── inference_debug.py              ← Standalone inference debugger
│
├── data/
│   ├── raw/                            ← Raw downloaded historical data
│   └── engineered/                     ← Engineered feature datasets
│
├── model/
│   └── aqi_multioutput_model.pkl       ← Local fallback serialized model
│
└── .github/
    └── workflows/
        ├── hourly_pipeline.yml         ← GitHub Actions: Runs feature_pipeline.py hourly
        └── daily_training.yml          ← GitHub Actions: Runs training_pipeline.py daily
```

---

## ⚙️ Windows Installation & Configuration

Follow these steps exactly to set up and run the project on your Windows machine.

### Step 1: Clone the Repository
Open PowerShell in your desired folder and clone the repository:
```powershell
git clone <repository-url>
cd pearls-aqi-predictor
```

### Step 2: Create a Virtual Environment (Python 3.11 ONLY)
> [!IMPORTANT]
> **Hopsworks has known compatibility issues with Python 3.12+.** You MUST use **Python 3.11** to avoid compilation and runtime import errors.

Create a virtual environment named `.venv`:
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
To activate the virtual environment in PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

> [!NOTE]
> If you get an execution policy error (`UnauthorizedAccess` / script execution is disabled on this system), run the following command first to temporarily allow scripts in this session:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`
> Then try activating again.

### Step 4: Install Dependencies
Once the virtual environment is active (you will see `(.venv)` at the beginning of your terminal prompt), run:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Open `.env` in a text editor and fill in your Hopsworks API Key:
```env
HOPSWORKS_API_KEY=your_actual_hopsworks_api_key
```

---

## 🚀 Step-by-Step Execution Guide

You must run the pipelines in this exact order to populate your Feature Store and train your model.

### 1️⃣ Run the Backfill Pipeline (Once)
Downloads 3 years of historical air quality and weather data from Open-Meteo, engineers 55 features (including cyclical time features, wind vectors, rolling statistics, and v6 residual/regime features), and registers the `aqi_features` feature group in Hopsworks.
```powershell
python pipelines/backfill_pipeline.py
```
*Note: This will take ~5–10 minutes to run and upload all historical data to Hopsworks.*

### 2️⃣ Run the Feature Pipeline (Hourly)
Fetches the latest hour's live air quality and weather conditions, calculates rolling features, and updates Hopsworks.
```powershell
python pipelines/feature_pipeline.py
```
*(In production, GitHub Actions runs this automatically every hour).*

### 3️⃣ Run the Training Pipeline
Fetches historical features from Hopsworks, trains 5 candidate models (Ridge, RF, XGBoost, LightGBM, Gradient Boosting), performs time-series validation, evaluates them on a test set, and uploads the best model (by weighted RMSE) to the Hopsworks Model Registry.
```powershell
python pipelines/training_pipeline.py
```
*(In production, GitHub Actions runs this automatically once a day).*

### 4️⃣ Launch the Streamlit Dashboard
To start the dashboard using the PowerShell shortcut:
```powershell
.\run_dashboard.ps1
```
This runs:
```powershell
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🛠️ Troubleshooting & Windows Gotchas

### ❌ Error: `cannot import name 'connection' from 'hsfs'`
* **Cause:** You ran `streamlit run` using a global or incorrect Python/Streamlit installation instead of the virtual environment. Python 3.12+ or an outdated global library triggers this import failure.
* **Fix:** Always activate the virtual environment first (`.\.venv\Scripts\Activate.ps1`) before running commands, or run the script using the dedicated shortcut `.\run_dashboard.ps1` which automatically uses the correct `.venv` path.

### ⚠️ Warning: `Failed to libgssapi_krb5 ... Loading Kerberos libraries not supported`
* **Cause:** Hopsworks HDFS library is checking for Kerberos authentication libraries which are not standard on Windows client systems.
* **Fix:** **Safe to ignore.** This is a warning only; the system will fall back to token/API key authentication and function normally.

### ⚠️ Warning: `IO error on RPC call, retrying`
* **Cause:** Minor network packet delay between your Windows client and the Hopsworks remote server.
* **Fix:** **Safe to ignore.** The library automatically retries and completes the data transfer.

### ❌ Error: `No delta logs found for featuregroup: .../aqi_predictions_1`
* **Cause:** When you first run the dashboard, the predictions table (`aqi_predictions`) is created, but no predictions have been successfully committed yet. The dashboard's history page tries to read from an empty table, causing Hopsworks to log a `FlightServerError`.
* **Fix:** The dashboard contains a built-in `try-except` handler for this error, meaning **it will not crash the app**. Once the first batch prediction runs and the background write job finishes on Hopsworks (takes a few minutes), this error will stop appearing in the console.
