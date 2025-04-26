# core/trainer/trainer.py

import os
import pandas as pd
from core.utils.load_multiple_bhavcopies import load_multiple_bhavcopies
from core.features.feature_engineer import create_features
from core.trainer.common import train_and_save_model
from core.config import DATA_DIR, DAILY_PROCESSED_PATH, DAILY_MODEL_PATH

PROCESSED_PATH = "data/processed_data.csv"

def run_daily_training():
    df = load_multiple_bhavcopies(DATA_DIR)
    print(f"[INFO] Loaded {len(df)} rows of raw bhavcopy data")
    
    features = create_features(df, predict_mode=False)
    print(f"[INFO] Processed dataset has {len(features)} rows")

    os.makedirs(os.path.dirname(DAILY_PROCESSED_PATH), exist_ok=True)
    features.to_csv(DAILY_PROCESSED_PATH, index=False)
    print(f"[INFO] Saved processed data to {DAILY_PROCESSED_PATH}")

    X = features.drop(columns=["symbol", "date", "target"])
    y = features["target"]

    train_and_save_model(X, y, model_path=DAILY_MODEL_PATH)

if __name__ == "__main__":
    run_daily_training()
