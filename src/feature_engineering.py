import pandas as pd

def add_time_features(df):

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    return df


def add_lag_features(df):

    df = df.sort_values(by=["Store", "Date"])

    df["Lag_1"] = df.groupby("Store")["Sales"].shift(1)
    df["Lag_7"] = df.groupby("Store")["Sales"].shift(7)

    df["Rolling_Mean_7"] = df.groupby("Store")["Sales"].shift(1).rolling(7).mean()

    df.dropna(inplace=True)

    return df


def encode_categorical(df):

    categorical_cols = [
        "StoreType",
        "Assortment",
        "StateHoliday",
        "PromoInterval"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df