# training_pipeline.py
import hopsworks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import joblib
import os
from src.config import HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, MODEL_NAME

def train_and_register():
    print("🔌 Connecting to Hopsworks Project...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    print("📥 Fetching features from the remote Hopsworks Feature Store...")
    aqi_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=1)
    df = aqi_fg.read()

    # 1. Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    print("📋 Features successfully pulled from Hopsworks!")

    # 2. Sort chronologically by timestamp/date before making look-ahead shifts
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    elif 'date' in df.columns:
        df = df.sort_values('date')

    # 3. Target Generation: Compute the 3-day (72 steps) future prediction target
    if 'target_aqi_3d' not in df.columns:
        print("🔮 Generating 3-day future targets (target_aqi_3d) from historical trends...")
        base_aqi_col = 'aqi' if 'aqi' in df.columns else 'aqi_rolling_24h'
        df['target_aqi_3d'] = df[base_aqi_col].shift(-72)
        df = df.dropna(subset=['target_aqi_3d'])

    # 4. Splitting into features (X) and target label (y)
    drop_cols = ['timestamp', 'date', 'target_aqi_3d']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['target_aqi_3d']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Training shape: {X_train.shape} | Evaluation shape: {X_test.shape}")
    
    # --- TOURNAMENT STAGE (TRAINING 3 MODELS) ---
    
    # Model 1: Ridge Regression
    print("🏋️ Training Model 1: Ridge Regression (Baseline)...")
    lr = Ridge()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = root_mean_squared_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)
    
    # Model 2: Random Forest
    print("🏋️ Training Model 2: Random Forest Regressor (Ensemble)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = root_mean_squared_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    
    # Model 3: XGBoost Regressor
    print("🏋️ Training Model 3: XGBoost Regressor (Advanced Boosting)...")
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = root_mean_squared_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)
    
    print("\n================ SYSTEM TOURNAMENT PERFORMANCE ================")
    print(f" 1. Ridge Baseline   -> MAE: {lr_mae:.2f} | RMSE: {lr_rmse:.2f} | R²: {lr_r2:.2f}")
    print(f" 2. Random Forest    -> MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f} | R²: {rf_r2:.2f}")
    print(f" 3. XGBoost Boosting -> MAE: {xgb_mae:.2f} | RMSE: {xgb_rmse:.2f} | R²: {xgb_r2:.2f}")
    print("===============================================================\n")
    
    # 5. DYNAMIC SELECTION MAP: Compare all 3 MAE metrics and choose the absolute lowest
    score_map = {
        lr_mae: (lr, lr_mae, lr_rmse, lr_r2, "Baseline Ridge Regression model predicting Islamabad AQI 3 days out."),
        rf_mae: (rf, rf_mae, rf_rmse, rf_r2, "Optimized Random Forest ensemble model predicting Islamabad AQI 3 days out."),
        xgb_mae: (xgb, xgb_mae, xgb_rmse, xgb_r2, "Advanced XGBoost model predicting Islamabad AQI 3 days out.")
    }
    
    best_mae = min(score_map.keys())
    champion_model, champion_mae, champion_rmse, champion_r2, model_desc = score_map[best_mae]
    
    print(f"🏆 The Champion Model is selected based on best performance! Description: {model_desc}")
    
    # Save the local champion model artifact
    model_dir = "saved_model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "aqi_model.pkl")
    joblib.dump(champion_model, model_path)
    
    # 6. REGISTRY: Push the tournament champion into Hopsworks Model Registry
    print("🏆 Uploading champion model to central Model Registry...")
    mr = project.get_model_registry()
    model_meta = mr.python.create_model(
        name=MODEL_NAME,
        metrics={"mae": champion_mae, "rmse": champion_rmse, "r2": champion_r2},
        description=model_desc
    )
    model_meta.save(model_path)
    print("✅ Best model successfully registered and ready for tomorrow's dashboard!")
    
    return X_train, champion_model

if __name__ == "__main__":
    train_and_register()