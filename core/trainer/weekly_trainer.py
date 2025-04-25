# core/trainer/weekly_trainer.py

import os
import pandas as pd
from core.utils.load_multiple_bhavcopies import load_multiple_bhavcopies
from core.utils.aggregate_weekly import aggregate_weekly_data
from core.features.weekly_feature_engineer import create_weekly_features
from core.trainer.common import train_and_save_model
from core.config import DATA_DIR, WEEKLY_MODEL_PATH

PROCESSED_WEEKLY_PATH = "data/weekly_processed.csv"

def run_weekly_training():
    print("[INFO] Loading daily bhavcopies...")
    df = load_multiple_bhavcopies(DATA_DIR)
    print(f"[INFO] Loaded {len(df)} rows of raw data")

    print("[INFO] Aggregating to weekly format...")
    weekly_df = aggregate_weekly_data(df)
    print(f"[INFO] Weekly dataset has {len(weekly_df)} rows")

    print("[INFO] Creating features...")
    features = create_weekly_features(weekly_df, predict_mode=False)
    print(f"[INFO] Feature dataset has {len(features)} rows")

    os.makedirs(os.path.dirname(PROCESSED_WEEKLY_PATH), exist_ok=True)
    features.to_csv(PROCESSED_WEEKLY_PATH, index=False)
    print(f"[INFO] Weekly data saved to {PROCESSED_WEEKLY_PATH}")

    X = features.drop(columns=["symbol", "date", "target"])
    y = features["target"]

    train_and_save_model(X, y, model_path=WEEKLY_MODEL_PATH)

if __name__ == "__main__":
    run_weekly_training()
