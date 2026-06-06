import streamlit as st
import hopsworks
import pandas as pd
import numpy as np
import joblib
import os
from src.config import HOPSWORKS_API_KEY, FEATURE_GROUP_NAME

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="",
    layout="wide"
)

# --- CACHED HOPSWORKS MODEL FETCHER ---
@st.cache_resource(show_spinner="Connecting to Hopsworks & downloading champion models...")
def load_hopsworks_models():
    """Connects to Hopsworks registry and downloads the best models for Days 1, 2, and 3."""
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()
    
    models = {}
    
    for day in [1, 2, 3]:
        model_name = f"aqi_model_{day}d"
        try:
            hw_model = mr.get_model(name=model_name, version=1) 
            downloaded_model_path = hw_model.download()
            pkl_file = os.path.join(downloaded_model_path, f"{model_name}.pkl")
            models[day] = joblib.load(pkl_file)
        except Exception as e:
            st.error(f"Failed to load model for Day {day}: {e}")
            st.stop()
            
    # Fetch the schema columns from the feature store to align input shapes
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=1)
    sample_df = aqi_fg.read(dataframe_type="pandas").head(1)
    
    drop_cols = ['timestamp', 'date', 'target_aqi_1d', 'target_aqi_2d', 'target_aqi_3d']
    feature_columns = [col.lower() for col in sample_df.columns if col not in drop_cols]
    
    return models, feature_columns

# Load models and column structural map
try:
    models, trained_feature_names = load_hopsworks_models()
except Exception as e:
    st.error(f"Could not connect to Hopsworks. Verify your API key configuration. Error: {e}")
    st.stop()

# --- APP INTERFACE HEADER ---
st.title(" Islamabad Air Quality Index (AQI) Predictor")
st.markdown("""
This interactive simulation lets you tweak real-time pollutant levels to see how future **Day 1, Day 2, and Day 3** AQI metrics will react. Predictions are evaluated by the cloud-optimized machine learning champions trained in your pipeline.
""")

st.divider()

# --- SIDEBAR / INPUT POLLUTANT CONTROLS ---
st.sidebar.header(" Live Pollutant Inputs")
st.sidebar.markdown("Adjust current air particles to simulate upcoming ambient environments.")

# Dynamic input collection based on PM values & base air quality metrics
pm25 = st.sidebar.slider("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=35.0, step=0.1)
pm10 = st.sidebar.slider("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=70.0, step=0.1)
base_aqi = st.sidebar.slider("Current Raw AQI / Rolling AQI Base", min_value=0, max_value=500, value=85, step=1)

# Supplementary features map to cover pipeline requirements
simulated_inputs = {
    'pm25': pm25,
    'pm10': pm10,
    'aqi': base_aqi,
    'aqi_rolling_24h': base_aqi  
}

# --- PROCESS INPUT TO MATCH TRAINING FEATURES ---
input_data = {}
for col in trained_feature_names:
    input_data[col] = simulated_inputs.get(col, 0.0)

input_df = pd.DataFrame([input_data])

# --- GENERATE FORECAST PREDICTIONS ---
predictions = {}
for day, model in models.items():
    aligned_input = input_df[trained_feature_names]
    predictions[day] = model.predict(aligned_input)[0]

# --- UI VISUALIZATION MATRIX ---
st.subheader("🔮 Predictive Intelligence Matrix")
col1, col2, col3 = st.columns(3)

def get_aqi_status(aqi_val):
    """Categorizes AQI value according to standard health brackets."""
    if aqi_val <= 50: return "🟢 Good"
    elif aqi_val <= 100: return "🟡 Moderate"
    elif aqi_val <= 150: return "🟠 Unhealthy for Sensitive Groups"
    elif aqi_val <= 200: return "🔴 Unhealthy"
    else: return "🟣 Hazardous"

with col1:
    status = get_aqi_status(predictions[1])
    st.metric(label="📆 Day 1 Forecast (24 Hours Out)", value=f"{predictions[1]:.1f} AQI")
    st.markdown(f"**Condition:** {status}")

with col2:
    status = get_aqi_status(predictions[2])
    st.metric(label="📆 Day 2 Forecast (48 Hours Out)", value=f"{predictions[2]:.1f} AQI")
    st.markdown(f"**Condition:** {status}")

with col3:
    status = get_aqi_status(predictions[3])
    st.metric(label="📆 Day 3 Forecast (72 Hours Out)", value=f"{predictions[3]:.1f} AQI")
    st.markdown(f"**Condition:** {status}")

st.divider()

# --- FORECAST TREND GRAPH ---
st.subheader(" 3-Day Forecast Trendline")
trend_df = pd.DataFrame({
    "Timeline": ["Current Base", "Day 1 (24h)", "Day 2 (48h)", "Day 3 (72h)"],
    "AQI Forecast Value": [base_aqi, predictions[1], predictions[2], predictions[3]]
}).set_index("Timeline")

st.line_chart(trend_df, y="AQI Forecast Value")

st.divider()

# --- NEW ADDITION: DYNAMIC HEALTH ADVISORY INSIGHTS ---
st.subheader(" Automated Public Health Advisory")

# Compute maximum expected AQI over the next 3 days to determine worst-case scenario advice
max_future_aqi = max(predictions[1], predictions[2], predictions[3])

if max_future_aqi <= 50:
    st.success("### ✅ Air Quality is Excellent\n"
               "The 3-day trend indicates pristine conditions. It is perfect for outdoor sports, "
               "ventilation, and regular activities across Islamabad.")
elif max_future_aqi <= 100:
    st.info("### 🟡 Air Quality is Acceptable\n"
             "Atmospheric health is moderate. Extremely sensitive individuals should consider monitoring "
             "prolonged outdoor exertion, but no major restrictions are necessary.")
elif max_future_aqi <= 150:
    st.warning("### 🟠 Sensitive Groups Precautionary Alert\n"
               "The  upcoming spikes  could impact vulnerable groups (children, elderly, and those "
               "with respiratory illnesses). Consider reducing heavy outdoor exercises over the forecast window.")
elif max_future_aqi <= 200:
    st.error("### 🔴 Unhealthy Air Warning\n"
             "Active forecast trends show high pollutant retention. Everyone may begin to experience adverse health effects. "
             "It is highly recommended to wear particulate masks (N95) outdoors and limit strenuous activity.")
else:
    st.error("### 🟣 Emergency Critical Air Hazard\n"
             "Severe environmental alert! The predictive matrix forecasts hazardous levels. Avoid outdoor exposure entirely, "
             "keep windows sealed, and ensure indoor air purifiers are running at full capacity.")