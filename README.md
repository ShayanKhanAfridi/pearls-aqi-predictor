# 🌫️ Karachi AQI Predictor — End-to-End MLOps System

A production-grade, end-to-end **MLOps forecasting system** that predicts the Air Quality Index (AQI) for **Karachi** for the next **3 days (72 hours)**. 

This system uses a serverless architecture powered by **Hopsworks Feature Store & Model Registry**, live automated pipelines run via **GitHub Actions**, and a modern, high-fidelity **Streamlit Dashboard** running real-time batch inference.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────┐
                       │   Open-Meteo Weather API    │
                       └──────────────┬───────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   ┌──────────────────────┐                       ┌──────────────────────┐
   │  Backfill Pipeline   │                       │   Feature Pipeline   │
   │  (3-Year Historical) │                       │ (Live Hourly Updates)│
   └──────────┬───────────┘                       └───────────┬──────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │  Hopsworks Feature Store  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │     Training Pipeline     │
                        │  (Retrains 5 algorithms)  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Hopsworks Model Registry  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Streamlit Dashboard    │
                        │ (Inference & Analytics)   │
                        └───────────────────────────┘
```

---

## 📁 Project Directory Structure

```
pearls-aqi-predictor/
├── README.md                           ← Project documentation
├── requirements.txt                    ← Core dependency requirements
├── .env.example                        ← Template environment configuration
├── .env                                ← Local credentials (ignored by Git)
├── run_dashboard.ps1                   ← Windows PowerShell dashboard launcher
│
├── dashboard/
│   └── app.py                          ← Streamlit application & inference engine
│
├── pipelines/
│   ├── __init__.py
│   ├── backfill_pipeline.py            ← Ingests & engineers 3 years of historical data
│   ├── feature_pipeline.py             ← Live hourly feature ingestion with REST fallback
│   └── training_pipeline.py            ← Models retraining, evaluation & registration
│
├── notebooks/
│   ├── eda.ipynb                       ← Exploratory Data Analysis & visualisations
│   ├── shap_analysis.ipynb             ← Feature importance & model explainability
│   └── inference_debug.py              ← Isolated testing of the inference lifecycle
│
├── data/
│   ├── raw/                            ← Cached/saved raw CSV data
│   └── engineered/                     ← Cached/saved engineered feature CSV data
│
└── model/
    └── aqi_multioutput_model.pkl       ← Local fallback serialized model artifact
```

---

## ⚙️ Setup and Installation

Follow these steps to set up and run the system locally on Windows.

### Step 1: Clone the Repository
```powershell
git clone <repository-url>
cd pearls-aqi-predictor
```

### Step 2: Create a Virtual Environment (Python 3.11 ONLY)
> [!IMPORTANT]
> **Hopsworks requires Python 3.11.** Do not use Python 3.12+ as some libraries (like `hsfs` and compiled C dependencies) will fail to compile or import.

Create a virtual environment named `.venv`:
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
Activate the environment in your PowerShell console:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then activate).*

### Step 4: Install Dependencies
With the environment active, run the following:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure Credentials
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Open `.env` in a text editor and fill in your Hopsworks API Key:
```env
HOPSWORKS_API_KEY=your_actual_hopsworks_api_key_here
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_CERT_FOLDER=.hopsworks_certs
```

---

## 🚀 Running the Backfill Pipeline in Google Colab / Jupyter

Because processing and uploading 3 years of historical data to Hopsworks contains heavy computations, it is recommended to run the backfill pipeline on **Google Colab** or a remote Jupyter Notebook.

Follow these cell-by-cell steps to run `pipelines/backfill_pipeline.py` in Colab:

### Cell 1: Install core requirements
```python
!pip install hopsworks==4.7.* requests pandas numpy python-dotenv openmeteo-requests requests-cache retry-requests
```

### Cell 2: Install Hopsworks connector with Kafka support
```python
!pip install "hopsworks[python]" confluent-kafka --quiet
```

### Cell 3: Configure your API environment
```python
import os
os.environ["HOPSWORKS_API_KEY"] = "your-api-key-here"
```

### Cell 4: Copy and execute the pipeline
Copy the complete contents of `pipelines/backfill_pipeline.py` into this cell and run it. It will fetch historical weather and air quality datasets from Open-Meteo, calculate engineered features, and push them to your Hopsworks Feature Store.

---

## 🏃‍♂️ Running Pipelines Locally

Once the historical backfill is complete, you can run the live update pipeline and model training pipeline from your virtual environment:

### Live Ingestion (Hourly Pipeline)
```powershell
python pipelines/feature_pipeline.py
```
*Note: On Windows, HDFS client writes are bypassed gracefully, and local computations are logged. In the cloud (GitHub Actions), this script automatically updates the Hopsworks online Feature Store.*

### Model Training & Selection
```powershell
python pipelines/training_pipeline.py
```
*This downloads the features from Hopsworks, trains candidate models, validates them, and registers the best model in the registry.*

### Launch the Streamlit Dashboard
```powershell
.\run_dashboard.ps1
```
Open `http://localhost:8501` to view the live dashboard.

---

## 🧠 Machine Learning Engine & Models

The system evaluates **5 candidate regression models** to solve the multi-output 3-day forecasting task:

1. **Ridge Regression:** L2 regularized linear model, providing a strong baseline.
2. **Random Forest Regressor:** Tree ensemble that models non-linear relationships.
3. **XGBoost Regressor:** High-performance gradient booster optimized for regression.
4. **LightGBM Regressor:** Light, leaf-wise gradient boosting engine.
5. **Gradient Boosting Regressor:** Sequential boosting algorithm for residual minimization.

### Selection Strategy (Weighted RMSE)
The training pipeline automatically selects the best algorithm by calculating a **Weighted Root Mean Squared Error (RMSE)** across the 3 forecast horizons:
$$\text{Weighted RMSE} = 0.20 \times \text{Day 1 RMSE} + 0.35 \times \text{Day 2 RMSE} + 0.45 \times \text{Day 3 RMSE}$$
This puts higher penalty on later errors, ensuring the model remains robust across the entire 72-hour forecast span.

### Feature Engineering Highlights
* **Cyclical Time Encoding:** Sine and Cosine transformations of hour, month, day of week, and day of year.
* **Wind Vectors:** Translation of wind speed and direction into Cartesian coordinate wind vectors (`wind_x`, `wind_y`).
* **Regime & Residual Features:**
  * `aqi_residual_168h`: The difference between current AQI and its 7-day rolling mean.
  * `aqi_residual_72h`: The difference between current AQI and its 3-day rolling mean.
  * `aqi_regime_range`: Max-min AQI range over the last 24 hours.

---

## 📊 Dashboard Modules

* **🏠 Home:** Overview of Karachi's current AQI status, health category (Good, Moderate, Unhealthy, etc.), meteorological parameters, and immediate health advice.
* **📈 Forecast:** Visual representations and tables of the 3-day multi-output AQI predictions.
* **📊 Historical:** Historical AQI trend lines and summary distributions.
* **⚠️ Health Advisory:** Detailed guidelines for vulnerable groups, general population warnings, and protective measures.
* **🌬️ Pollutant Breakdown:** Detailed tracking of PM2.5, PM10, ozone ($O_3$), nitrogen dioxide ($NO_2$), sulfur dioxide ($SO_2$), and carbon monoxide ($CO$) against global standards.
