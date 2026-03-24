from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from config import MODEL_PATH
from .evaluation import evaluate_model

def train_random_forest(df):

    columns_to_drop = ["Sales", "Date", "Customers"]

    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred, "Random Forest")

    joblib.dump(model, f"{MODEL_PATH}/random_forest.pkl")

    return metrics