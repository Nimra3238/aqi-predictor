# feature_pipeline.py
import os
import sys
import tempfile
import pandas as pd

# Force the environment to use a valid Windows temp directory
os.environ['TMPDIR'] = tempfile.gettempdir()
import hopsworks
from src.config import HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from src.data_fetcher import fetch_live_aqi
from src.features import engineer_features

def run():
    print("⏳ [1/4] Checking environment credentials...")
    if not HOPSWORKS_API_KEY:
        print("❌ Error: HOPSWORKS_API_KEY is empty inside your .env file!")
        sys.exit(1)
        
    print(f"🚀 [2/4] Linking registry with api key prefix: {HOPSWORKS_API_KEY[:5]}...")
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        print("   ✅ Secure connection established with cloud feature store.")
    except Exception as e:
        print(f"   ❌ Authorization failure: {str(e)}")
        sys.exit(1)
        
    print("📡 [3/4] Gathering current metrics from Islamabad API stations...")
    try:
        raw_df = fetch_live_aqi()
        
        # --- FIX 1: FORCE UNIQUE PRIMARY KEY TIMESTAMP ---
        # This guarantees Hopsworks recognizes a NEW row every single day/hour
        raw_df['timestamp'] = pd.Timestamp.now()
        
        current_aqi_val = raw_df['aqi'].values[0]
        print(f"   ✅ Data gathered successfully. Current AQI reading: {current_aqi_val}")
    except Exception as e:
        print(f"   ❌ API endpoint error: {str(e)}")
        sys.exit(1)
        
    print("🧠 [4/4] Executing feature engineering transforms and updating registry...")
    try:
        # Get the feature group reference from Hopsworks
        aqi_fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=['timestamp'],
            description="Hourly Air Quality Index features for Islamabad with time components.",
            online_enabled=True
        )
        
        # --- FIX 2: BLEND HISTORICAL CONTEXT FOR ROLLING WINDOWS ---
        print("📥 Fetching recent historical data from Hopsworks to calculate rolling trends...")
        historical_df = aqi_fg.read()
        
        if not historical_df.empty:
            # Sort chronologically and take the last 23 rows
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
            historical_df = historical_df.sort_values('timestamp').tail(23)
            
            # Combine the 23 historical rows with our 1 fresh live row
            combined_df = pd.concat([historical_df, raw_df], ignore_index=True)
            
            print("🧮 Calculating rolling averages across history + current row...")
            engineered_combined = engineer_features(combined_df)
            
            # Extract ONLY the final row (today's fresh row) which now has a perfect 24h rolling calculation
            feature_df = engineered_combined.tail(1).copy()
        else:
            # Fallback if the feature group is completely empty
            print("⚠️ Feature group is empty. Processing single row baseline.")
            feature_df = engineer_features(raw_df)

    except Exception as e:
        print(f"⚠️ History blending skipped due to error: {str(e)}. Falling back to single row.")
        feature_df = engineer_features(raw_df)
    
    # Drop the target column cleanly if it slips into the dataframe
    if 'target_aqi_3d' in feature_df.columns:
        feature_df = feature_df.drop(columns=['target_aqi_3d'])

    print("--- DATAFRAME CLEANED AND VERIFIED ---")
    print(feature_df.info())
    print(feature_df)
    
    # Push the single, perfectly calculated fresh row to Hopsworks
    aqi_fg.insert(feature_df, write_options={"wait_for_job": False})
    print("🎉 Success! Your fresh, real-time feature point was pushed into the cloud.")


if __name__ == "__main__":
    run()