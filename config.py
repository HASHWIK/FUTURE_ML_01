import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "figures")

# File names
TRAIN_FILE = os.path.join(DATA_RAW_PATH, "train.csv")
STORE_FILE = os.path.join(DATA_RAW_PATH, "store.csv")

PROCESSED_FILE = os.path.join(DATA_PROCESSED_PATH, "final_dataset.csv")