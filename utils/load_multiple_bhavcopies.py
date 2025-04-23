# utils/load_multiple_bhavcopies.py
import os
import pandas as pd
from utils.load_bhavcopy import load_bhavcopy
from datetime import datetime

def load_multiple_bhavcopies(data_dir, days=None):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if days:
        files = files[-days:]  # Pick last N files

    print(f"[INFO] Using {len(files)} bhavcopy files:")
    for f in files:
        try:
            date_str = f.replace(".csv", "")
            date_obj = datetime.strptime(date_str, "%d%m%Y")
            if date_obj.weekday() >= 5:
                print(f"[WARNING] {f} falls on a weekend — skipping.")
        except ValueError:
            print(f"[WARNING] Could not parse date from filename: {f}")

    data_frames = []
    for file in files:
        path = os.path.join(data_dir, file)
        df = load_bhavcopy(path)
        df["date"] = file.replace(".csv", "")
        data_frames.append(df)

    combined = pd.concat(data_frames, ignore_index=True)

    # Keep only symbols that appear on all selected days
    symbol_counts = combined['symbol'].value_counts()
    valid_symbols = symbol_counts[symbol_counts == len(files)].index.tolist()
    combined = combined[combined['symbol'].isin(valid_symbols)]

    return combined
