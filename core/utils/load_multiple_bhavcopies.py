from datetime import datetime
import os
import pandas as pd
from core.utils.load_bhavcopy import load_bhavcopy
from core.config import DATA_DIR

def load_multiple_bhavcopies(data_dir = DATA_DIR, days=None, verbose=True):
    
    # --- 🛠️ Precheck Step: Auto-rename messy bhavcopies first ---
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    for filename in files:
        if filename.startswith("sec_bhavdata_full_") and filename.endswith(".csv"):
            date_str = filename.replace("sec_bhavdata_full_", "").replace(".csv", "")
            try:
                parsed_date = datetime.strptime(date_str, "%d%m%Y")
                clean_name = parsed_date.strftime("%d%m%Y") + ".csv"
                raw_path = os.path.join(data_dir, filename)
                clean_path = os.path.join(data_dir, clean_name)

                if not os.path.exists(clean_path):
                    os.rename(raw_path, clean_path)
                    print(f"[RENAME] {filename} ➔ {clean_name}")
                else:
                    print(f"[SKIP] Already exists: {clean_name}")
            except Exception as e:
                print(f"[ERROR] Failed to rename {filename}: {e}")

    # --- 🧠 Now safe to sort clean files by date ---
    
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
