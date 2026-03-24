import streamlit as st
import pandas as pd
import joblib
import sys
import os
import matplotlib.pyplot as plt

# -----------------------------
# PATH SETUP
# -----------------------------
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_PATH)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecast & Prediction Dashboard")
st.markdown("### Powered by XGBoost ML Model")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(ROOT_PATH, "models", "xgboost.pkl"))
    feature_columns = joblib.load(os.path.join(ROOT_PATH, "models", "xgb_feature_columns.pkl"))
    return model, feature_columns

model, feature_columns = load_model()

# ==========================================================
# 🔴 REAL-TIME PREDICTION
# ==========================================================
st.markdown("## 🔴 Real-Time Sales Prediction")

col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", min_value=1, step=1)
    date = st.date_input("Select Date")

    promo = 1 if st.selectbox("Is Promotion Running Today?", ["Yes", "No"]) == "Yes" else 0
    promo2 = 1 if st.selectbox("Is Long-Term Promotion Active?", ["Yes", "No"]) == "Yes" else 0
    school_holiday = 1 if st.selectbox("Is It a School Holiday?", ["Yes", "No"]) == "Yes" else 0

with col2:
    open_store = 1 if st.selectbox("Store Status", ["Open", "Closed"]) == "Open" else 0

    store_type_display = st.selectbox(
        "Store Type",
        [
            "A → Basic store",
            "B → Medium store",
            "C → Large store",
            "D → Premium store"
        ]
    )
    store_type = store_type_display[0]

    assortment_display = st.selectbox(
        "Assortment Type",
        [
            "A → Basic",
            "B → Medium",
            "C → Extended"
        ]
    )
    assortment = assortment_display[0]

    lag_1 = st.number_input("Yesterday Sales", min_value=0.0)
    lag_7 = st.number_input("Sales 7 Days Ago", min_value=0.0)

# -----------------------------
# 7 DAY INPUT
# -----------------------------
st.markdown("### 📊 7-Day Average Sales")

avg_option = st.radio(
    "Choose input method:",
    ["Enter Directly", "Calculate from 7 Days"]
)

if avg_option == "Enter Directly":
    rolling_7 = st.number_input("7-Day Average", min_value=0.0)
    daily_values = None
else:
    cols = st.columns(7)
    daily_values = []

    for i in range(7):
        value = cols[i].number_input(f"D{i+1}", min_value=0.0, key=f"d{i}")
        daily_values.append(value)

    rolling_7 = sum(daily_values) / 7 if sum(daily_values) != 0 else 0
    st.success(f"Average: {rolling_7:,.2f}")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Sales", use_container_width=True):

    date = pd.to_datetime(date)

    input_data = {
        "Store": store,
        "DayOfWeek": date.weekday() + 1,
        "Open": open_store,
        "Promo": promo,
        "Promo2": promo2,
        "SchoolHoliday": school_holiday,
        "Year": date.year,
        "Month": date.month,
        "Day": date.day,
        "Weekday": date.weekday(),
        "Lag_1": lag_1,
        "Lag_7": lag_7,
        "Rolling_Mean_7": rolling_7
    }

    for stype in ["a", "b", "c", "d"]:
        input_data[f"StoreType_{stype}"] = 1 if store_type.lower() == stype else 0

    for atype in ["a", "b", "c"]:
        input_data[f"Assortment_{atype}"] = 1 if assortment.lower() == atype else 0

    df_input = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[feature_columns]

    prediction = model.predict(df_input)[0]

    st.markdown("## 📊 Prediction Result")
    st.metric("Predicted Sales", f"₹ {prediction:,.2f}")

    if lag_1 > 0:
        percent_change = ((prediction - lag_1) / lag_1) * 100
        st.metric("Change vs Yesterday", f"{percent_change:.2f}%")

# ==========================================================
# 📈 30 DAY FORECAST
# ==========================================================
st.markdown("---")
st.markdown("## 📈 30-Day Forecast")

store_id = st.number_input("Store ID for Forecast", min_value=1, key="forecast")

if st.button("Generate Forecast", use_container_width=True):

    today = pd.Timestamp.today()
    future_dates = pd.date_range(start=today, periods=30)

    predictions = []

    for date in future_dates:
        input_data = {
            "Store": store_id,
            "DayOfWeek": date.weekday() + 1,
            "Open": 1,
            "Promo": 0,
            "SchoolHoliday": 0,
            "Year": date.year,
            "Month": date.month,
            "Day": date.day,
            "Weekday": date.weekday(),
            "Lag_1": 5000,
            "Lag_7": 5000,
            "Rolling_Mean_7": 5000
        }

        df_input = pd.DataFrame([input_data])

        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[feature_columns]

        prediction = model.predict(df_input)[0]
        predictions.append(prediction)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Sales": predictions
    })

    st.line_chart(forecast_df.set_index("Date"))
    st.dataframe(forecast_df)

# ==========================================================
# 📊 FEATURE IMPORTANCE
# ==========================================================
st.markdown("---")
st.markdown("## 📊 Feature Importance")

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(importance_df.set_index("Feature"))
    st.dataframe(importance_df)

else:
    st.info("Feature importance not available.")