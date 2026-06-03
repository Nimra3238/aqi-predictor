# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Target city coordinates for Islamabad
CITY_NAME = "Islamabad"
LAT = "33.6844"
LON = "73.0479"

FEATURE_GROUP_NAME = "aqi_prediction_fg"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "islamabad_aqi_model"