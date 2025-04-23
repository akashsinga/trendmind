# weekly_main.py
import pandas as pd
from utils.load_multiple_bhavcopies import load_multiple_bhavcopies
from utils.aggregate_weekly import aggregate_weekly

OUTPUT_FILE = "data/weekly_processed.csv"
DATA_DIR = "data/bhavcopies"


def main():
    print("[INFO] Loading daily bhavcopies...")
    raw_df = load_multiple_bhavcopies(DATA_DIR)
    print(f"[INFO] Loaded {len(raw_df)} rows of raw data")

    print("[INFO] Aggregating to weekly format...")
    weekly_df = aggregate_weekly(raw_df)
    print(f"[INFO] Weekly dataset has {len(weekly_df)} rows")

    weekly_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Weekly data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
