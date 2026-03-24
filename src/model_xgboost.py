import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
from config import MODEL_PATH
from .evaluation import evaluate_model

def train_xgboost(df):

    columns_to_drop = ["Sales", "Date", "Customers"]

    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    y = df["Sales"]

    joblib.dump(X.columns.tolist(), "models/xgb_feature_columns.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred, "XGBoost")

    joblib.dump(model, f"{MODEL_PATH}/xgboost.pkl")

    return metrics