import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # 🔥 FIX: Use non-GUI backend
import matplotlib.pyplot as plt
from datetime import timedelta
import os

def forecast_next_30_days(df):

    print("\n📈 Generating 30-Day Future Forecast using XGBoost...")

    # -----------------------------
    # Load trained model
    # -----------------------------
    model = joblib.load("models/xgboost.pkl")

    # Load feature column order used during training
    feature_columns = joblib.load("models/xgb_feature_columns.pkl")

    # -----------------------------
    # Get last available row
    # -----------------------------
    df = df.sort_values("Date")
    last_row = df.iloc[-1:].copy()
    last_date = df["Date"].max()

    # -----------------------------
    # Create next 30 dates
    # -----------------------------
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=30
    )

    predictions = []

    # -----------------------------
    # Generate predictions iteratively
    # -----------------------------
    for date in future_dates:

        new_row = last_row.copy()

        # Update date-based features
        new_row["Date"] = date
        new_row["Year"] = date.year
        new_row["Month"] = date.month
        new_row["Day"] = date.day
        new_row["Weekday"] = date.weekday()

        # Drop non-feature columns
        X_future = new_row.drop(
            columns=["Sales", "Date", "Customers"],
            errors="ignore"
        )

        # Ensure same feature order as training
        X_future = X_future[feature_columns]

        # Predict
        pred = model.predict(X_future)[0]
        predictions.append(pred)

        # Update lag features dynamically
        new_row["Lag_1"] = pred
        new_row["Lag_7"] = pred
        new_row["Rolling_Mean_7"] = pred

        last_row = new_row

    # -----------------------------
    # Create forecast dataframe
    # -----------------------------
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Sales": predictions
    })

    # -----------------------------
    # Save forecast CSV
    # -----------------------------
    os.makedirs("reports", exist_ok=True)
    forecast_df.to_csv("reports/future_30_day_forecast.csv", index=False)

    print("✅ Forecast CSV saved to reports/future_30_day_forecast.csv")

    # -----------------------------
    # Plot Forecast
    # -----------------------------
    # -----------------------------
    # Plot Forecast (Improved X-Axis)
    # -----------------------------
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(10, 5))

    dates = pd.to_datetime(forecast_df["Date"])

    ax.plot(dates, forecast_df["Predicted_Sales"])

    # Show every 5th day to avoid overlapping
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    ax.set_title("Next 30-Day Sales Forecast (XGBoost)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Sales")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("reports/forecast_plot.png")
    plt.close()