# core/predictor/weekly_predictor.py

import os
from datetime import datetime, timedelta
import pandas as pd
import joblib

from core.config import (DATA_DIR, WEEKLY_MODEL_PATH,WEEKLY_PREDICTIONS_DIR, WEEKLY_PREDICTIONS_LATEST_PATH,CONFIDENCE_THRESHOLD, CONFIDENCE_BUCKETS)
from core.utils.load_multiple_bhavcopies import load_multiple_bhavcopies
from core.utils.aggregate_weekly import aggregate_weekly_data
from core.utils.dates import get_next_trading_day
from core.features.weekly_feature_engineer import create_weekly_features

def run_weekly_prediction(prediction_threshold = CONFIDENCE_THRESHOLD):
    if not os.path.exists(WEEKLY_MODEL_PATH):
        print("[ERROR] Trained weekly model not found.")
        return

    print("[INFO] Loading daily bhavcopies...")
    df = load_multiple_bhavcopies(DATA_DIR)
    weekly_df = aggregate_weekly_data(df)
    features = create_weekly_features(weekly_df, predict_mode=True)

    if features.empty:
        print("[WARNING] No data after weekly feature creation.")
        return

    model = joblib.load(WEEKLY_MODEL_PATH)
    X = features.drop(columns=["symbol", "date", "target"], errors="ignore")
    predictions = model.predict(X)
    confidences = model.predict_proba(X)[:, 1]

    features["prediction"] = predictions
    features["confidence"] = confidences

    latest_date = features["date"].iloc[0]
    predicted_date = get_next_trading_day(latest_date)

    pred_df = features[
        (features["prediction"] == 1) &
        (features["confidence"] >= prediction_threshold)
    ].copy()

    if pred_df.empty:
        print(f"[INFO] No strong bullish weekly predictions for {predicted_date} (based on {latest_date}).")
        return

    pred_df = pred_df[["symbol", "date", "close", "confidence"]]
    pred_df.rename(columns={"close": "last_close_price"}, inplace=True)
    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    print(f"\n[WEEKLY PREDICTION RESULTS for {predicted_date} (based on {latest_date})]")
    print(pred_df)

    print("\n[SUMMARY]")
    top_5 = pred_df.head(5)["symbol"].tolist()
    print(f"Top 5 weekly bullish candidates: {', '.join(top_5)}")

    print("\n[CONFIDENCE ZONES]")
    for low, high in CONFIDENCE_BUCKETS:
        bucket_df = pred_df[(pred_df["confidence"] > low) & (pred_df["confidence"] <= high)]
        symbols = bucket_df["symbol"].tolist()
        if symbols:
            print(f"{low:.1f}–{high:.1f}: {', '.join(symbols)}")
            
    dated_path = os.path.join(WEEKLY_PREDICTIONS_DIR, f"{predicted_date}.csv")
    
    os.makedirs(os.path.dirname(dated_path), exist_ok=True)
    pred_df.to_csv(dated_path, index=False)
    pred_df.to_csv(WEEKLY_PREDICTIONS_LATEST_PATH, index=False)
    print(f"[INFO] Weekly predictions saved to {dated_path} and {WEEKLY_PREDICTIONS_LATEST_PATH}")

if __name__ == "__main__":
    run_weekly_prediction()
