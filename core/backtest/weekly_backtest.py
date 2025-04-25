# core/backtest/weekly_backtest.py

import os
import pandas as pd
from datetime import timedelta

from core.config import (WEEKLY_PREDICTIONS_DIR,WEEKLY_BACKTEST_OUTPUT,WEEKLY_PROCESSED_PATH,CONFIDENCE_BUCKETS,PERCENT_MOVE_THRESHOLD)

def get_latest_weekly_prediction_file():
    files = sorted(
        (f for f in os.listdir(WEEKLY_PREDICTIONS_DIR) if f.endswith(".csv") and f != "latest.csv"),
        key=lambda x: pd.to_datetime(x.replace(".csv", ""))
    )
    if not files:
        raise FileNotFoundError("[ERROR] No weekly prediction files found.")
    return files[-1]

def run_weekly_backtest():
    latest_file = get_latest_weekly_prediction_file()
    predicted_for = latest_file.replace(".csv", "")
    print(f"[INFO] Backtesting weekly predictions for: {predicted_for}")

    pred_df = pd.read_csv(os.path.join(WEEKLY_PREDICTIONS_DIR, latest_file))
    actual_df = pd.read_csv(WEEKLY_PROCESSED_PATH)

    next_week_dt = pd.to_datetime(predicted_for) + timedelta(days=7)
    next_week_str = next_week_dt.strftime("%Y-%m-%d")

    actual_next_week = actual_df[actual_df["date"] == next_week_str]
    actual_close_map = actual_next_week.set_index("symbol")["close"].to_dict()

    pred_df["actual_close"] = pred_df["symbol"].map(actual_close_map)
    pred_df.dropna(subset=["actual_close"], inplace=True)

    pred_df["correct_prediction"] = (pred_df["actual_close"] > pred_df["last_close_price"]).astype(int)
    pred_df["percent_move"] = (
        (pred_df["actual_close"] - pred_df["last_close_price"]) / pred_df["last_close_price"] * 100
    ).round(2)

    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    print(f"\n[BACKTEST RESULTS for {predicted_for} → actual week: {next_week_str}]")
    print(pred_df[["symbol", "last_close_price", "actual_close", "percent_move", "confidence", "correct_prediction"]])

    total = len(pred_df)
    correct = pred_df["correct_prediction"].sum()
    accuracy = correct / total if total else 0

    print("\n[SUMMARY]")
    print(f"Total Predictions: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\n[CONFIDENCE ZONES]")
    for low, high in CONFIDENCE_BUCKETS:
        bucket = pred_df[(pred_df["confidence"] > low) & (pred_df["confidence"] <= high)]
        if not bucket.empty:
            print(f"{low:.1f}–{high:.1f}: {', '.join(bucket['symbol'])}")

    qualified = pred_df[pred_df["percent_move"] > PERCENT_MOVE_THRESHOLD]
    if not qualified.empty:
        min_conf = qualified["confidence"].min()
        print(f"\n[INFO] Minimum confidence among >{PERCENT_MOVE_THRESHOLD}% movers: {min_conf:.4f}")
        print(qualified[["symbol", "percent_move", "confidence"]])
    else:
        print(f"\n[INFO] No predictions had a percent move greater than {PERCENT_MOVE_THRESHOLD}%")

    os.makedirs(os.path.dirname(WEEKLY_BACKTEST_OUTPUT), exist_ok=True)
    pred_df.to_csv(WEEKLY_BACKTEST_OUTPUT, index=False)
    print(f"[INFO] Weekly backtest results saved to {WEEKLY_BACKTEST_OUTPUT}")

if __name__ == "__main__":
    run_weekly_backtest()
