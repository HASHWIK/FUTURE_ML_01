from src.data_preprocessing import load_and_merge_data, clean_data, save_processed_data
from src.feature_engineering import add_time_features, add_lag_features, encode_categorical
from src.model_random_forest import train_random_forest
from src.model_xgboost import train_xgboost
from src.model_lightgbm import train_lightgbm
from src.model_prophet import train_prophet
import pandas as pd
from src.forecasting import forecast_next_30_days


def main():

    print("🚀 Starting Sales Forecasting Pipeline")

    # Step 1: Load & merge
    df = load_and_merge_data()

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Feature Engineering
    df = add_time_features(df)
    df = add_lag_features(df)
    df = encode_categorical(df)

    print("Training Models...")

    rf_metrics = train_random_forest(df)
    xgb_metrics = train_xgboost(df)
    lgb_metrics = train_lightgbm(df)
    prophet_metrics = train_prophet(df)

    print("\n🔥 Model Comparison:")
    print("Random Forest:", rf_metrics)
    print("XGBoost:", xgb_metrics)
    print("LightGBM:", lgb_metrics)
    print("Prophet:", prophet_metrics)


    # ✅ Select Best Model (Based on R2)
    models_metrics = {
        "RandomForest": rf_metrics,
        "XGBoost": xgb_metrics,
        "LightGBM": lgb_metrics,
        "Prophet": prophet_metrics
    }

    best_model_name = max(models_metrics, key=lambda x: models_metrics[x]["R2"])

    print(f"\n🏆 Best Model Selected: {best_model_name}")

    # Save Best Model Name
    with open("models/best_model.txt", "w") as f:
        f.write(best_model_name)

    forecast_next_30_days(df)
    print("\n✅ Pipeline Completed Successfully!")
if __name__ == "__main__":
    main()