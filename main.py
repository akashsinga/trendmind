# main.py
import pandas as pd
from features.create_features import create_features
from utils.load_multiple_bhavcopies import load_multiple_bhavcopies

OUTPUT_FILE = "data/processed_data.csv"
DATA_DIR = "data/bhavcopies"
DAYS_TO_LOAD = None  # Load all available files if None

def main():
    raw_df = load_multiple_bhavcopies(DATA_DIR, days=DAYS_TO_LOAD)
    print(f"[INFO] Loaded {len(raw_df)} rows of combined raw data")

    processed_df = create_features(raw_df)
    print(f"[INFO] Processed dataset has {len(processed_df)} rows")

    processed_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved processed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
