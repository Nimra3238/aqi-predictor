# src/data_fetcher.py
import requests
import pandas as pd
from src.config import AQICN_TOKEN, LAT, LON

def fetch_live_aqi():
    """Fetches real-time AQI and pollutant data for Islamabad from AQICN."""
    if not AQICN_TOKEN:
        raise ValueError("Missing AQICN_TOKEN! Double check your .env file setup.")
        
    url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
    response = requests.get(url).json()
    
    if response['status'] != 'ok':
        raise ValueError(f"AQICN API Error: {response['data']}")
        
    data = response['data']
    iaqi = data.get('iaqi', {})
    
    # Extract data with safe fallbacks (default to 0.0 if a pollutant isn't reported)
    metrics = {
        "timestamp": pd.to_datetime(data['time']['s']),
        "aqi": float(data.get('aqi', 0)),
        "pm25": float(iaqi.get('pm25', {}).get('v', 0)),
        "pm10": float(iaqi.get('pm10', {}).get('v', 0)),
        "no2": float(iaqi.get('no2', {}).get('v', 0)),
        "o3": float(iaqi.get('o3', {}).get('v', 0)),
        "temperature": float(iaqi.get('t', {}).get('v', 0)),
        "humidity": float(iaqi.get('h', {}).get('v', 0))
    }
    return pd.DataFrame([metrics])