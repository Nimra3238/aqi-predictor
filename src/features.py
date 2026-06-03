# src/features.py
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes time components, 24-hour trends, and differences."""
    df = df.copy()
    
    # 1. Clean whitespace from headers and handle variations in naming
    df.columns = df.columns.str.strip()
    if 'date' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
        
    # Ensure baseline pollutant mapping is forced to numeric types safely
    if 'median' in df.columns and 'pm25' not in df.columns:
        df['pm25'] = pd.to_numeric(df['median'], errors='coerce')

    # Parse timestamp strings or objects cleanly to datetimes
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Structural Time Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # 3. Historical Contextual Trends (Rolling metrics require a sorted index)
    # If the file has fewer than 24 rows initially, min_periods=1 prevents NaN crashes
    df['pm25_rolling_24h'] = df['pm25'].rolling(window=24, min_periods=1).mean()
    df['aqi_rolling_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
    
    # 4. Delta Change feature
    df['aqi_diff_1h'] = df['aqi'].diff(1).fillna(0.0)
    
    # Primary key formatting required by Hopsworks string/timestamp storage rules
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def create_targets(df: pd.DataFrame, forecast_steps: int = 72) -> pd.DataFrame:
    """Aligns the row features with a label 3 days in the future.
    
    Use forecast_steps=72 for hourly live stream data tracking.
    Use forecast_steps=3 for daily aggregated historical backfill data.
    """
    df = df.copy()
    
    # Safely sort by timestamp string format or parse back to datetime temporarily
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp_dt').reset_index(drop=True)
    
    # Shift backwards so current features map to the target value X steps ahead
    df['target_aqi_3d'] = df['aqi'].shift(-forecast_steps)
    
    # Clean up the temporary datetime processing column cleanly
    df = df.drop(columns=['timestamp_dt'])
    return df