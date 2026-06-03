# feature_pipeline.py
import os
import sys
import tempfile

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
        # registration=False prevents Hopsworks from locking up to prompt interactive browser requests
        #project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, registration=False)
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        print("   ✅ Secure connection established with cloud feature store.")
    except Exception as e:
        print(f"   ❌ Authorization failure: {str(e)}")
        sys.exit(1)
        
    print("📡 [3/4] Gathering current metrics from Islamabad API stations...")
    try:
        raw_df = fetch_live_aqi()
        current_aqi_val = raw_df['aqi'].values[0]
        print(f"   ✅ Data gathered successfully. Current AQI reading: {current_aqi_val}")
    except Exception as e:
        print(f"   ❌ API endpoint error: {str(e)}")
        sys.exit(1)
        
    print("🧠 [4/4] Executing feature engineering transforms and updating registry...")
    feature_df = engineer_features(raw_df)
    
    # In live streaming pipelines, the future label isn't known yet, so we mark it as None
    feature_df['target_aqi_3d'] = None
    
    aqi_fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=['timestamp'],
        description="Hourly Air Quality Index features for Islamabad with time components.",
        online_enabled=True
    )
  # Drop the empty target column that is crashing Hopsworks
    if 'target_aqi_3d' in feature_df.columns:
        feature_df = feature_df.drop(columns=['target_aqi_3d'])

    # Make sure all print lines and the insert line are perfectly aligned
    print("--- DATAFRAME CLEANED ---")
    print(feature_df.info())
    
    aqi_fg.insert(feature_df, write_options={"wait_for_job": False})
    print("🎉 Success! Your first online streaming feature point was pushed into the cloud.")


if __name__ == "__main__":
    run()