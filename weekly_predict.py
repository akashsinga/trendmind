# weekly_predict.py
import os
import pandas as pd
import joblib
from datetime import datetime, timedelta
from features.create_weekly_features import create_weekly_features

MODEL_PATH = "models/weekly_model.pkl"
DATA_PATH = "data/weekly_processed.csv"
OUTPUT_FILE = "weekly_prediction_results.csv"
JOURNAL_DIR = "logs"
CONFIDENCE_BUCKETS = [(0.9, 1.0), (0.7, 0.9), (0.5, 0.7)]


def get_latest_week(df):
    df["week_dt"] = pd.to_datetime(df["week"])  # handles "2025-01-27" correctly
    latest = df["week_dt"].max()
    return latest.strftime("%Y-%m-%d")  # match format in CSV


def predict_weekly():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Weekly model not found. Run weekly_train.py first.")
        return

    print("[INFO] Loading weekly dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded weekly data with shape: {df.shape}")

    print("[INFO] Creating weekly features...")
    df = create_weekly_features(df)
    if df.empty:
        print("[WARNING] No data available after feature creation.")
        return

    latest_week = get_latest_week(df)

    # 🔧 FIX: Remove helper column before filtering
    df.drop(columns=["week_dt"], inplace=True, errors="ignore")

    latest_df = df[df["week"] == latest_week]

    if latest_df.empty:
        print(f"[WARNING] No data found for latest week: {latest_week}")
        return

    X = latest_df.drop(columns=["symbol", "week", "target"])
    model = joblib.load(MODEL_PATH)

    print(f"[INFO] Making predictions for next week based on week: {latest_week}")
    probas = model.predict_proba(X)
    predictions = model.predict(X)
    confidence = probas[:, 1]

    latest_df["prediction"] = predictions
    latest_df["confidence"] = confidence

    pred_df = latest_df[latest_df["prediction"] == 1].copy()
    if pred_df.empty:
        print(f"[INFO] No bullish predictions for week starting {latest_week}.")
        return

    pred_df = pred_df[["symbol", "week", "close_price", "confidence"]]
    pred_df.rename(columns={"close_price": "last_week_close"}, inplace=True)
    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    # Predicting for the *next* week
    try:
        predicted_week = (
            datetime.strptime(latest_week, "%Y-%m-%d") + timedelta(days=7)
        ).strftime("%Y-%m-%d")
    except ValueError:
        predicted_week = "NextWeek"

    print(f"\n[WEEKLY PREDICTION for {predicted_week} (based on {latest_week})]")
    print(pred_df)

    print("\n[SUMMARY]")
    print("Likely bullish stocks for next week:", ", ".join(pred_df["symbol"]))

    print("\n[CONFIDENCE ZONES]")
    for low, high in CONFIDENCE_BUCKETS:
        bucket = pred_df[(pred_df["confidence"] > low) & (pred_df["confidence"] <= high)]
        if not bucket.empty:
            print(f"{low:.1f}–{high:.1f}: {', '.join(bucket['symbol'])}")

    # Save predictions
    pred_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Weekly prediction results saved to {OUTPUT_FILE}")

    os.makedirs(JOURNAL_DIR, exist_ok=True)
    journal_path = os.path.join(JOURNAL_DIR, f"weekly_predictions_{predicted_week}.csv")
    pred_df.to_csv(journal_path, index=False)
    print(f"[INFO] Journal saved to {journal_path}")


if __name__ == "__main__":
    predict_weekly()
