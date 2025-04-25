# core/predictor/predictor.py

import os
from datetime import datetime, timedelta
import pandas as pd
import joblib

from core.config import (DATA_DIR, DAILY_MODEL_PATH,DAILY_PREDICTIONS_DIR, DAILY_PREDICTIONS_LATEST_PATH,CONFIDENCE_THRESHOLD, CONFIDENCE_BUCKETS)
from core.utils.load_multiple_bhavcopies import load_multiple_bhavcopies
from core.features.feature_engineer import create_features

def run_daily_prediction(prediction_threshold = CONFIDENCE_THRESHOLD):
    today = datetime.now()
    if today.weekday() >= 5:
        print("[INFO] Weekend detected. No prediction will be made.")
        return

    if not os.path.exists(DAILY_MODEL_PATH):
        print("[ERROR] Trained model not found. Please run trainer.py first.")
        return

    print("[INFO] Loading bhavcopies...")
    df = load_multiple_bhavcopies(DATA_DIR)
    features = create_features(df, predict_mode=True)

    if features.empty:
        print("[WARNING] No data after feature creation.")
        return

    model = joblib.load(DAILY_MODEL_PATH)
    X = features.drop(columns=["symbol", "date", "target"], errors="ignore")
    predictions = model.predict(X)
    confidences = model.predict_proba(X)[:, 1]

    features["prediction"] = predictions
    features["confidence"] = confidences

    latest_date = features["date"].iloc[0]
    predicted_date = (datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%d%m%Y")

    # Filter by confidence and prediction
    pred_df = features[
        (features["prediction"] == 1) &
        (features["confidence"] >= prediction_threshold)
    ].copy()

    if pred_df.empty:
        print(f"[INFO] No strong bullish predictions for {predicted_date} (based on {latest_date}).")
        return

    pred_df = pred_df[["symbol", "date", "close","prediction","confidence"]]
    pred_df.rename(columns={"close": "last_close_price"}, inplace=True)
    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    # 🖨️ Print summary
    print(f"\n[PREDICTION RESULTS for {predicted_date} (based on data from {latest_date})]")
    print(pred_df)

    print("\n[SUMMARY]")
    top_5 = pred_df.head(5)["symbol"].tolist()
    print(f"Top 5 bullish candidates: {', '.join(top_5)}")

    print("\n[CONFIDENCE ZONES]")
    for low, high in CONFIDENCE_BUCKETS:
        bucket_df = pred_df[(pred_df["confidence"] > low) & (pred_df["confidence"] <= high)]
        symbols = bucket_df["symbol"].tolist()
        if symbols:
            print(f"{low:.1f}–{high:.1f}: {', '.join(symbols)}")
            
    
    dated_path = os.path.join(DAILY_PREDICTIONS_DIR, f"{predicted_date}.csv")
    
    # 💾 Save to outputs/daily/
    os.makedirs(os.path.dirname(dated_path), exist_ok=True)
    pred_df.to_csv(dated_path, index=False)
    pred_df.to_csv(DAILY_PREDICTIONS_LATEST_PATH, index=False)
    print(f"[INFO] Predictions saved to {dated_path} and {DAILY_PREDICTIONS_LATEST_PATH}")

if __name__ == "__main__":
    run_daily_prediction()
