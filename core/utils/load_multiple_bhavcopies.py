import os
import pandas as pd
from core.utils.load_bhavcopy import load_bhavcopy

def load_multiple_bhavcopies(data_dir, days=None, verbose=True):
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".csv")],
        key=lambda x: pd.to_datetime(x.replace(".csv", ""), format="%d%m%Y")
    )

    if days:
        files = files[-days:]

    if verbose:
        print(f"[INFO] Loading {len(files)} bhavcopy files (sorted by date):")

    data_frames = []
    for file in files:
        parsed_date = pd.to_datetime(file.replace(".csv", ""), format="%d%m%Y")
        if verbose:
            print(f" - {file} → {parsed_date.strftime('%Y-%m-%d')}")

        path = os.path.join(data_dir, file)
        df = load_bhavcopy(path)
        df["date"] = parsed_date.strftime("%Y-%m-%d")
        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)
