# app.py
import streamlit as st
import hopsworks
import pandas as pd
import datetime
import os
import joblib
from src.config import HOPSWORKS_API_KEY, FEATURE_GROUP_NAME

# 1. Page Configuration
st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide")

# 2. Inject Custom Responsive CSS to Fix Text Truncation/Cut-offs
st.markdown(
    """
    <style>
    /* target the inner metric component container block */
    [data-testid="stMetricValue"] {
        font-size: clamp(1.8rem, 2.5vw, 2.8rem) !important;
        font-weight: 700 !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.2 !important;
    }
    /* target metric labels to ensure uniform wrapping */
    [data-testid="stMetricLabel"] {
        font-size: clamp(0.9rem, 1.2vw, 1.1rem) !important;
        white-space: normal !important;
        word-break: break-all !important;
    }
    /* container tweaks to stop horizontal squishing */
    div[data-testid="column"] {
        padding: 10px 15px !important;
        background-color: #f8f9fa;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(" Islamabad Air Quality Index Predictor")
st.markdown("This dashboard shows predicted AQI for Islamabad for the next 3 days.")
            
@st.cache_resource
def load_models_and_features():
    """Connects to Hopsworks, downloads all 3 models, and grabs the latest feature row."""
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    aqi_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=1)
    df = aqi_fg.read()
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp', ascending=False)
    elif 'date' in df.columns:
        df = df.sort_values('date', ascending=False)
        
    latest_features = df.iloc[[0]].copy()
    
    drop_cols = ['timestamp', 'date', 'target_aqi_1d', 'target_aqi_2d', 'target_aqi_3d']
    X_live = latest_features.drop(columns=[col for col in drop_cols if col in latest_features.columns], errors='ignore')
    
    models = {}
    os.makedirs("saved_models", exist_ok=True)
    
    for day in [1, 2, 3]:
        model_name = f"aqi_model_{day}d"
        try:
            hw_model = mr.get_model(name=model_name, version=2) 
            model_dir = hw_model.download()
            
            pkl_filename = f"aqi_model_{day}d.pkl"
            local_pkl_path = os.path.join(model_dir, pkl_filename)
            
            models[day] = joblib.load(local_pkl_path)
        except Exception as e:
            st.error(f"⚠️ Error loading {model_name}: {e}")
            
    return X_live, models, latest_features

# Run Data Connection
with st.spinner("🔌 Establishing connections to remote feature stores and downloading champion models..."):
    X_live, models, raw_features = load_models_and_features()

# Display Context Rows
#st.subheader("📊 Latest Registered Air Quality Observation Context")
#col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)

#current_aqi = float(raw_features['aqi'].values[0]) if 'aqi' in raw_features.columns else float(raw_features['aqi_rolling_24h'].values[0])
#current_pm25 = float(raw_features['pm25'].values[0]) if 'pm25' in raw_features.columns else 0.0

# Extract clean formatted metadata strings
#obs_time = str(raw_features['timestamp'].values[0]) if 'timestamp' in raw_features.columns else "Recent"
#location_str = "Islamabad Center"

#with col_meta1:
# st.metric(label="Current Air Quality (AQI Baseline)", value=f"{current_aqi:.1f}")
#with col_meta2:
#st.metric(label="PM2.5 Level (µg/m³)", value=f"{current_pm25:.1f}")
#with col_meta3:
# st.metric(label="Observation Timestamp", value=obs_time)
#with col_meta4:
#st.metric(label="Monitoring Location", value=location_str)

st.markdown("---")
st.subheader(" Multi-Day Look-Ahead Forecasting Matrix")

predictions = {}
for day in [1, 2, 3]:
    if day in models:
        pred_val = models[day].predict(X_live)[0]
        predictions[day] = max(0.0, pred_val)
    else:
        predictions[day] = None

# Dynamically extract Today's Live AQI baseline from the incoming raw feature row
current_aqi_live = float(raw_features['aqi'].values[0]) if 'aqi' in raw_features.columns else (float(raw_features['aqi_rolling_24h'].values[0]) if 'aqi_rolling_24h' in raw_features.columns else 0.0)

# Created a 4-column layout row to cleanly embed Today's live tracking point
col0, col1, col2, col3 = st.columns(4)
today = datetime.date.today()

def get_aqi_status(val):
    if val <= 50: return "🟩 Good", "Minimal Impact"
    elif val <= 100: return "🟨 Moderate", "Acceptable Quality"
    elif val <= 150: return "🟧 Unhealthy for Sensitive Groups", "Action Advised"
    else: return "🔴 Unhealthy", "Active Health Hazard Warning Issued"

with col0:
    st.markdown(f"###  Today's AQI")
    st.caption(f"Observed: {today.strftime('%A, %B %d')}")
    st.metric(label="Live Baseline AQI Score", value=f"{current_aqi_live:.1f}")
    status, health = get_aqi_status(current_aqi_live)
    st.info(f"**Status:** {status}\n\n*{health}*")

with col1:
    st.markdown(f"###  Day 1 AQI")
    st.caption(f"Target: {(today + datetime.timedelta(days=1)).strftime('%A, %B %d')}")
    if predictions[1] is not None:
        st.metric(label="Predicted AQI Index Score", value=f"{predictions[1]:.1f}")
        status, health = get_aqi_status(predictions[1])
        st.info(f"**Status:** {status}\n\n*{health}*")
    else:
        st.warning("Prediction Offline")

with col2:
    st.markdown(f"###  Day 2 AQI")
    st.caption(f"Target: {(today + datetime.timedelta(days=2)).strftime('%A, %B %d')}")
    if predictions[2] is not None:
        st.metric(label="Predicted AQI Index Score", value=f"{predictions[2]:.1f}")
        status, health = get_aqi_status(predictions[2])
        st.info(f"**Status:** {status}\n\n*{health}*")
    else:
        st.warning("Prediction Offline")

with col3:
    st.markdown(f"###  Day 3 AQI")
    st.caption(f"Target: {(today + datetime.timedelta(days=3)).strftime('%A, %B %d')}")
    if predictions[3] is not None:
        st.metric(label="Predicted AQI Index Score", value=f"{predictions[3]:.1f}")
        status, health = get_aqi_status(predictions[3])
        st.info(f"**Status:** {status}\n\n*{health}*")
    else:
        st.warning("Prediction Offline")

st.markdown("---")
st.caption("  This dashboard fetches live context dynamically. Models are automatically updated daily by a continuous automated model selection tournament running via serverless workflows on GitHub Actions.")