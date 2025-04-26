import os
import pandas as pd
import joblib
from core.config import (CONFIDENCE_THRESHOLD,DAILY_PREDICTIONS_DIR,DAILY_PREDICTIONS_LATEST_PATH,MODEL_DIR)
from core.utils.dates import get_next_trading_day
from core.utils.top_signals import print_top_signals

ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_model.pkl")
PROCESSED_DATA_PATH = "data/processed_data.csv"

def run_ensemble_prediction(prediction_threshold=CONFIDENCE_THRESHOLD):
    if not os.path.exists(ENSEMBLE_MODEL_PATH):
        print("[ERROR] Ensemble model not found. Please run ensemble_trainer.py first.")
        return

    print("[INFO] Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    if df.empty:
        print("[WARNING] Processed data is empty.")
        return

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()

    X = latest_df.drop(columns=["symbol", "date", "target"], errors="ignore")
    X = X.fillna(0)

    model = joblib.load(ENSEMBLE_MODEL_PATH)

    latest_df["prediction_class"] = model.predict(X)
    latest_df["confidence"] = model.predict_proba(X).max(axis=1)
    latest_df["prediction"] = latest_df["prediction_class"].map({1: "bullish", 0: "bearish"})

    final_pred_df = latest_df[latest_df["confidence"] >= prediction_threshold].copy()
    final_pred_df = final_pred_df[["symbol", "date", "close", "prediction", "confidence"]]
    final_pred_df.rename(columns={"close": "last_close_price"}, inplace=True)
    final_pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    if final_pred_df.empty:
        print("[INFO] No strong predictions.")
        return

    predicted_date = get_next_trading_day(latest_date)

    dated_path = os.path.join(DAILY_PREDICTIONS_DIR, f"{predicted_date}.csv")
    os.makedirs(os.path.dirname(dated_path), exist_ok=True)

    final_pred_df.to_csv(dated_path, index=False)
    final_pred_df.to_csv(DAILY_PREDICTIONS_LATEST_PATH, index=False)

    print_top_signals(final_pred_df)

    print(f"\n[PREDICTION RESULTS for {predicted_date} (based on {latest_date})]")
    print(final_pred_df)

if __name__ == "__main__":
    run_ensemble_prediction()
