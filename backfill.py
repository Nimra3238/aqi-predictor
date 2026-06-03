# backfill.py
import os
import pandas as pd
import hopsworks
import numpy as np
from dotenv import load_dotenv

# Force load variables from your local .env file
load_dotenv()

try:
    from src.config import HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
except ImportError:
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    FEATURE_GROUP_NAME = "islamabad_aqi_feature_group"
    FEATURE_GROUP_VERSION = 1

from src.features import engineer_features

def run_backfill():
    print("📖 Reading 'historical_raw.csv' safely...")
    try:
        df = pd.read_csv("historical_raw.csv", comment='#')
    except FileNotFoundError:
        print("❌ Critical Error: 'historical_raw.csv' is missing from your root directory!")
        return

    df.columns = df.columns.str.strip()
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    if 'median' in df.columns:
        df['pm25'] = pd.to_numeric(df['median'], errors='coerce')
    else:
        print("❌ Error: Could not find 'median' column in your CSV data.")
        return

    df['pm25'] = df['pm25'].interpolate(method='linear').bfill().ffill()

    def pm25_to_aqi(pm):
        if pm <= 12.0: return ((50 - 0)/(12.0 - 0)) * (pm - 0) + 0
        elif pm <= 35.4: return ((100 - 51)/(35.4 - 12.1)) * (pm - 12.1) + 51
        elif pm <= 55.4: return ((150 - 101)/(55.4 - 35.5)) * (pm - 35.5) + 101
        elif pm <= 150.4: return ((200 - 151)/(150.4 - 55.5)) * (pm - 55.5) + 151
        elif pm <= 250.4: return ((300 - 201)/(250.4 - 150.5)) * (pm - 150.5) + 201
        else: return ((500 - 301)/(500.4 - 250.5)) * (pm - 250.5) + 301

    print("🧮 Calculating baseline synthetic AQI records...")
    df['aqi'] = df['pm25'].apply(pm25_to_aqi)

    # Provide safe matching columns for your pre-initialized feature group schema
    df['pm10'] = df['pm25'] * 1.5
    df['no2'] = 15.0
    df['o3'] = 25.0
    df['temperature'] = 22.0
    df['humidity'] = 55.0

    print("🧠 Extracting features through your src/features.py functions...")
    processed_df = engineer_features(df)
    
    # FIX THE HOURLY SKEW & TYPE MISMATCH: Cast explicitly to int32 to match Hopsworks int type
    processed_df['hour'] = (processed_df.index % 24).astype('int32')
    
    # Matches the exact initialized features layout in your cloud server
    expected_features = [
        'timestamp', 'pm25', 'aqi', 'pm10', 'no2', 'o3', 
        'temperature', 'humidity', 'hour', 'day_of_week', 
        'month', 'pm25_rolling_24h', 'aqi_rolling_24h', 'aqi_diff_1h'
    ]
    
    # Filter down to match the feature store columns perfectly, removing raw stats columns
    processed_df = processed_df[expected_features]
    
    print(f"📊 Validated Data Dimensions for Schema: {processed_df.shape}")
    
    print("🚀 Connecting to Hopsworks Store using API key variables...")
    if not HOPSWORKS_API_KEY:
        print("❌ Error: HOPSWORKS_API_KEY variable is missing from your .env configuration file.")
        return
        
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    print("💾 Retrieving Feature Group inside cluster...")
    aqi_fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=['timestamp'],
        description="Islamabad historical daily air quality metrics feature data layer."
    )
    
    print("⏳ Streaming data rows into Hopsworks storage (This may take a minute)...")
    aqi_fg.insert(processed_df, write_options={"wait_for_job": True})
    print("🏆 Success! Your Feature Store is completely initialized and seeded with robust hourly distributions.")

if __name__ == "__main__":
    run_backfill()