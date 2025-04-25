# core/backtest/backtest.py

import os
import pandas as pd
from core.config import (DATA_DIR,DAILY_PREDICTIONS_DIR,DAILY_BACKTEST_OUTPUT,PERCENT_MOVE_THRESHOLD)
from core.utils.load_bhavcopy import load_bhavcopy

def get_latest_prediction_file():
    files = sorted(
        (f for f in os.listdir(DAILY_PREDICTIONS_DIR) if f.endswith(".csv") and f != "latest.csv"),
        key=lambda x: pd.to_datetime(x.replace(".csv", ""))
    )
    if not files:
        raise FileNotFoundError("[ERROR] No daily prediction files found.")
    return files[-1]

def run_daily_backtest():
    latest_pred_file = get_latest_prediction_file()
    prediction_date = latest_pred_file.replace(".csv", "")

    print(f"[INFO] Using predictions from {latest_pred_file}")
    pred_df = pd.read_csv(os.path.join(DAILY_PREDICTIONS_DIR, latest_pred_file))

    bhavcopy_path = os.path.join(DATA_DIR, f"{prediction_date.replace('-', '')}.csv")
    if not os.path.exists(bhavcopy_path):
        print(f"[ERROR] Bhavcopy not found for predicted day: {prediction_date}")
        return

    print(f"[INFO] Loading actual bhavcopy from {bhavcopy_path}...")
    actual_df = load_bhavcopy(bhavcopy_path)

    actual_close_map = actual_df.set_index("symbol")["close"].to_dict()
    pred_df["actual_close"] = pred_df["symbol"].map(actual_close_map)
    pred_df.dropna(subset=["actual_close"], inplace=True)

    pred_df["correct_prediction"] = (pred_df["actual_close"] > pred_df["last_close_price"]).astype(int)
    pred_df["percent_move"] = (
        (pred_df["actual_close"] - pred_df["last_close_price"]) / pred_df["last_close_price"] * 100
    ).round(2)

    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    print(f"\n[BACKTEST RESULTS for {prediction_date}]")
    print(pred_df[["symbol", "last_close_price", "actual_close", "percent_move", "confidence", "correct_prediction"]])

    total = len(pred_df)
    correct = pred_df["correct_prediction"].sum()
    accuracy = correct / total if total else 0

    print("\n[SUMMARY]")
    print(f"Total Predictions Made:    {total}")
    print(f"Correct Predictions:       {correct}")
    print(f"Accuracy:                  {accuracy:.2%}")

    movers = pred_df[pred_df["percent_move"] > PERCENT_MOVE_THRESHOLD]
    if not movers.empty:
        min_conf = movers["confidence"].min()
        print(f"\n[INFO] Stocks with >{PERCENT_MOVE_THRESHOLD}% move:")
        print(f"Minimum Confidence Among Them: {min_conf:.4f}")
        print(movers[["symbol", "percent_move", "confidence"]])
    else:
        print(f"\n[INFO] No predicted stocks moved more than {PERCENT_MOVE_THRESHOLD}% on {prediction_date}")

    os.makedirs(os.path.dirname(DAILY_BACKTEST_OUTPUT), exist_ok=True)
    pred_df.to_csv(DAILY_BACKTEST_OUTPUT, index=False)
    print(f"[INFO] Backtest results saved to {DAILY_BACKTEST_OUTPUT}")

if __name__ == "__main__":
    run_daily_backtest()
