#  Islamabad Air Quality Index (AQI) Predictor

A serverless MLOps system that collects real-time atmospheric data in Islamabad, manages features in a cloud feature store, and serves 3-day look-ahead forecasts through a web dashboard.

---

##  System Architecture

The platform uses a decoupled design to separate data pipelines from model inference:

*  **Feature Pipeline (`feature_pipeline.py`):** Fetches live data, calculates rolling metrics, and updates the feature store.
*  **Feature Store (Hopsworks):** Manages the online and offline data layers securely.
*  **Training Pipeline (`training_pipeline.py`):** Evaluates models daily (Ridge Regression, Random Forest, XGBoost) and registers the best performer.
*  **User Dashboard (`app.py`):** A Streamlit interface that pulls the latest features and active models for live predictions.

---

##  Data Insights & Feature Analysis

### 1. Feature Correlations
* **Particulate Matter:** `aqi`, `pm25`, and `pm10` share a near-perfect correlation of **0.98 to 1.00**, confirming that particulates drive air quality in Islamabad.
* **Weather Dynamics:** Temperature and humidity show a strong inverse relationship of **-0.96**.

### 2. Target Distribution
* **Data Skewness:** While normal daily baselines sit between **50.0 and 250.0**, seasonal events create a heavy right tail reaching past **2500+**. This skew justifies using robust ensemble models over basic linear regressions.

---

##  Model Performance

The machine learning models maintain reliable predictive power across all forecasting horizons:

| Forecast Horizon | Mean Absolute Error (MAE) ⬇️ | Root Mean Squared Error (RMSE) ⬇️ | $R^2$ Score ⬆️ |
| :--- | :---: | :---: | :---: |
| **1-Day Look-Ahead** | 4.334 | 5.436 | 0.9831 |
| **2-Day Look-Ahead** | 4.812 | 5.921 | 0.9792 |
| **3-Day Look-Ahead** | 5.245 | 6.412 | 0.9741 |

---


