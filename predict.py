# predict.py
from datetime import datetime, timedelta
import os
import pandas as pd
import joblib
from features.create_features import create_features
from utils.load_bhavcopy import load_bhavcopy

MODEL_PATH = "models/random_forest_model.pkl"
DATA_DIR = "data/bhavcopies"
OUTPUT_FILE = "prediction_results.csv"
JOURNAL_DIR = "logs"
CONFIDENCE_BUCKETS = [(0.9, 1.0), (0.7, 0.9), (0.5, 0.7)]


def load_all_bhavcopies(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if len(files) < 1:
        raise ValueError("Need at least one bhavcopy file to make a prediction.")

    latest_file = files[-1]
    latest_date = latest_file.replace(".csv", "")

    dfs = []
    for file in files:
        path = os.path.join(data_dir, file)
        df = load_bhavcopy(path)
        df["date"] = file.replace(".csv", "")
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True), latest_date


def predict():
    # Auto-skip weekends
    today = datetime.strptime(datetime.now().strftime("%d%m%Y"), "%d%m%Y")
    if today.weekday() >= 5:  # Saturday or Sunday
        print("[INFO] Weekend detected. No prediction will be made.")
        return
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Trained model not found. Please run train.py first.")
        return

    print("[INFO] Loading bhavcopies...")
    df, latest_date = load_all_bhavcopies(DATA_DIR)
    processed_df = create_features(df)

    if processed_df.empty:
        print("[WARNING] No data after feature creation.")
        return

    X = processed_df.drop(columns=["symbol", "date", "target"])
    model = joblib.load(MODEL_PATH)

    probas = model.predict_proba(X)
    predictions = model.predict(X)
    confidence = probas[:, 1]

    processed_df["prediction"] = predictions
    processed_df["confidence"] = confidence

    pred_df = processed_df[processed_df["prediction"] == 1]
    pred_df = pred_df[pred_df["date"] == latest_date]  # Predicting for tomorrow using today's data

    if pred_df.empty:
        print(f"[INFO] No predicted upward moves for trading day: {latest_date}.")
        return

    pred_df = pred_df[["symbol", "date", "close_price", "confidence"]]
    pred_df.rename(columns={"close_price": "last_close_price"}, inplace=True)
    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    try:
        predicted_date = (datetime.strptime(latest_date, "%d%m%Y") + timedelta(days=1)).strftime("%d%m%Y")
    except ValueError:
        predicted_date = "Next Day"

    print(f"\n[PREDICTION RESULTS for {predicted_date} (based on data from {latest_date})]")
    print(pred_df)

    top_symbols = pred_df["symbol"].tolist()
    print("\n[SUMMARY]")
    print(f"Next trading day is likely to be strong for: {', '.join(top_symbols)}")

    # Print confidence buckets
    print("\n[CONFIDENCE ZONES]")
    for low, high in CONFIDENCE_BUCKETS:
        bucket_df = pred_df[(pred_df["confidence"] > low) & (pred_df["confidence"] <= high)]
        symbols = bucket_df["symbol"].tolist()
        if symbols:
            print(f"{low:.1f}–{high:.1f}: {', '.join(symbols)}")

    # Save general prediction file
    pred_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Prediction results saved to {OUTPUT_FILE}")

    # Save journal entry
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    journal_file = os.path.join(JOURNAL_DIR, f"predictions_{predicted_date}.csv")
    pred_df.to_csv(journal_file, index=False)
    print(f"[INFO] Daily journal saved to {journal_file}")


if __name__ == "__main__":
    predict()
