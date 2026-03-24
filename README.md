# 🚀 Sales Demand Forecast & Prediction System
## 📊 End-to-End Machine Learning Project for Retail Sales Prediction and Demand Forecasting

This project builds a complete machine learning pipeline to predict retail store sales and forecast future demand using historical business data.

The system includes data preprocessing, feature engineering, model training, model comparison, and deployment using an interactive Streamlit dashboard.

## 🌐 Live Application

🔗 Try the dashboard here:
https://futureml01-4ktpobgypu384nsffaaxat.streamlit.app/

## 📌 Project Overview

Sales forecasting helps businesses optimize inventory management, marketing strategies, and revenue planning.

This project develops a machine learning-based forecasting system that:

- Analyzes historical retail sales data
- Builds predictive machine learning models
- Selects the best-performing model
- Deploys the solution through an interactive dashboard

The final system allows users to:

- Predict future store sales
- Generate 30-day demand forecasts
- Visualize sales trends and feature importance

## 🎯 Problem Statement

Businesses face major challenges in predicting product demand accurately.

Poor forecasting can lead to:

- Overstocking inventory
- Product shortages
- Increased operational costs
- Reduced business profitability

This project aims to build an intelligent forecasting system that predicts future sales using historical store data and machine learning models.

## 📂 Dataset Information

The dataset includes historical retail store data with the following attributes:

- Store ID
- Date
- Promotion indicators
- School holidays
- Store type
- Assortment type
- Competition distance
- Historical sales
- Customer information

These features help train machine learning models to understand sales patterns and seasonal trends.

## 🧠 Machine Learning Models Used

Multiple models were trained and evaluated:

- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Prophet (Time Series Forecasting)

Model performance was compared using evaluation metrics to select the best predictive model.

## ⚙️ Machine Learning Pipeline

The project follows a structured ML workflow:

- Data Collection
- Data Cleaning & Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model Comparison
- Best Model Selection
- Sales Forecast Generation
- Dashboard Deployment

## 📊 Dashboard Features

The deployed dashboard provides three key functionalities:

### 🔴 Real-Time Sales Prediction

Users can predict store sales by entering:

- Store ID
- Promotion status
- Holiday indicators
- Store type
- Assortment type
- Previous sales values

The system instantly predicts expected sales for the selected date.

### 📈 30-Day Sales Forecast

The system generates a 30-day forecast visualization including:

- Forecast sales trend
- Average predicted sales
- Maximum predicted sales
- Minimum predicted sales
- Target revenue line

This helps businesses understand future demand patterns.

### 📊 Feature Importance Analysis

Displays the top factors affecting sales predictions, helping businesses understand:

- Which features influence revenue the most
- How promotions affect sales
- Seasonal trends and store performance

## 📁 Project Structure
```
Sales_Demand_Forecast
│
├── dashboard
│   └── app.py
│
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── model_random_forest.py
│   ├── model_xgboost.py
│   ├── model_lightgbm.py
│   └── model_prophet.py
│
├── models
│   ├── xgboost.pkl
│   └── xgb_feature_columns.pkl
│
├── reports
│
├── main.py
├── requirements.txt
└── README.md
```

## 📊 Model Evaluation

Models were evaluated using standard regression metrics:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Example performance comparison:

| Model         | R² Score        |
| ------------- | --------------- |
| Random Forest | 0.81            |
| XGBoost       | **0.83 (Best)** |
| LightGBM      | 0.79            |
| Prophet       | 0.19            |

The XGBoost model achieved the best performance and was selected for deployment.

## 💻 Technologies Used

### Programming Language

 - Python

### Libraries

 - Pandas
 - NumPy
 - Scikit-learn
 - XGBoost
 - LightGBM
 - Prophet
 - Matplotlib

### Framework

 - Streamlit

### Deployment

 - Streamlit Cloud

## ▶️ How to Run the Project Locally
Clone the repository
```
git clone https://github.com/Deepakchakra/FUTURE_ML_01.git
```
Navigate to the project folder
```
cd FUTURE_ML_01
```
Install dependencies
```
pip install -r requirements.txt
```
Run the dashboard
```
streamlit run dashboard/app.py
```

## 📷 Dashboard Preview 

- Real-Time Prediction Interface
- Sales Forecast Graph
- Feature Importance Visualization

## 📈 Future Improvements

Possible improvements for the system:

- Integrate real-time sales APIs
- Add deep learning models (LSTM / Transformer)
- Deploy using Docker or cloud platforms
- Add automated model retraining pipeline

## 👨‍💻 Author

### Deepak Chakrasali

Machine Learning & AI Enthusiast

### GitHub
https://github.com/Deepakchakra
