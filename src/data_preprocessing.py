import pandas as pd
from config import TRAIN_FILE, STORE_FILE, PROCESSED_FILE

def load_and_merge_data():
    print("Loading datasets...")

    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    store = pd.read_csv(STORE_FILE)

    print("Merging datasets...")
    df = train.merge(store, on="Store", how="left")

    return df


def clean_data(df):
    print("Cleaning data...")

    # Remove closed stores
    df = df[df["Open"] == 1]

    # Remove zero sales
    df = df[df["Sales"] > 0]

    # Fill missing competition distance
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
    df["CompetitionDistance"].median()
    )

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def save_processed_data(df):
    df.to_csv(PROCESSED_FILE, index=False)
    print("Processed dataset saved.")