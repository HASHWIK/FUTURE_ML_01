import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
from config import MODEL_PATH


def train_lightgbm(df):

    # Aggregate total daily sales
    prophet_df = df.groupby("Date")["Sales"].sum().reset_index()

    prophet_df.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)

    # Train-test split
    split_index = int(len(prophet_df) * 0.8)
    train = prophet_df[:split_index]
    test = prophet_df[split_index:]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train)

    # Predict only test period
    forecast = model.predict(test[["ds"]])

    y_true = test["y"].values
    y_pred = forecast["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n📊 Prophet Performance:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    joblib.dump(model, f"{MODEL_PATH}/prophet_model.pkl")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}