import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from core.utils.load_multiple_bhavcopies import load_multiple_bhavcopies
from core.config import DATA_DIR, DAILY_PREDICTIONS_LATEST_PATH

def run_ensemble_backtest(predictions_file=DAILY_PREDICTIONS_LATEST_PATH):
    print("[INFO] Loading predictions...")
    pred_df = pd.read_csv(predictions_file)
    if pred_df.empty:
        print("[ERROR] Predictions file is empty.")
        return

    prediction_date = pred_df["date"].iloc[0]

    # Load bhavcopies
    print("[INFO] Loading bhavcopy data...")
    bhav_df = load_multiple_bhavcopies(DATA_DIR)
    bhav_df["date"] = pd.to_datetime(bhav_df["date"])
    pred_date = pd.to_datetime(prediction_date)
    next_trading_df = bhav_df[bhav_df["date"] > pred_date]
    
    if next_trading_df.empty:
        print("[ERROR] No bhavcopy data available for next trading day after prediction date.")
        return

    next_date = next_trading_df["date"].min()
    next_day_df = next_trading_df[next_trading_df["date"] == next_date]

    # Merge predictions with actual next day prices
    merged = pred_df.merge(next_day_df, on="symbol", how="inner", suffixes=("_pred", "_actual"))

    if merged.empty:
        print("[ERROR] No matching symbols found between predictions and bhavcopy.")
        return

    # Define actual movement
    merged["actual_movement"] = merged["close"] > merged["last_close_price"]

    # Calculate prediction correctness
    merged["predicted_bullish"] = merged["prediction"] == "bullish"
    merged["correct_prediction"] = merged["predicted_bullish"] == merged["actual_movement"]

    # Metrics
    y_true = merged["actual_movement"].astype(int)
    y_pred = merged["predicted_bullish"].astype(int)

    print("\n[RESULTS]")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

    # Separate bullish and bearish analysis
    print("\n[DETAILED BREAKDOWN]")
    bullish = merged[merged["prediction"] == "bullish"]
    bearish = merged[merged["prediction"] == "bearish"]

    print(f"\nBullish Predictions: {len(bullish)}")
    if not bullish.empty:
        bull_acc = bullish["correct_prediction"].mean()
        print(f"Bullish Correctness: {bull_acc:.4f}")

    print(f"\nBearish Predictions: {len(bearish)}")
    if not bearish.empty:
        bear_acc = bearish["correct_prediction"].mean()
        print(f"Bearish Correctness: {bear_acc:.4f}")

if __name__ == "__main__":
    run_ensemble_backtest()
