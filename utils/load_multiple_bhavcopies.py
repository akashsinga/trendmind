# utils/load_multiple_bhavcopies.py
import os
import pandas as pd
from utils.load_bhavcopy import load_bhavcopy
from datetime import datetime

def load_multiple_bhavcopies(data_dir, days=None):
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".csv")],
        key=lambda x: pd.to_datetime(x.replace(".csv", ""), format="%d%m%Y")
    )
    if days:
        files = files[-days:]  # Pick last N files

    print(f"[INFO] Loading {len(files)} bhavcopy files (sorted by date):")

    data_frames = []
    for file in files:
        parsed_date = pd.to_datetime(file.replace(".csv", ""), format="%d%m%Y").strftime("%Y-%m-%d")
        print(f" - {file} → {parsed_date}")
        path = os.path.join(data_dir, file)
        df = load_bhavcopy(path)
        df["date"] = file.replace(".csv", "")
        data_frames.append(df)

    combined = pd.concat(data_frames, ignore_index=True)

    # Removed strict filtering of symbols appearing in all days
    return combined
