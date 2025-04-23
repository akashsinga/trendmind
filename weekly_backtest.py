# weekly_backtest.py
import os
import pandas as pd

LOG_DIR = "logs"
DATA_PATH = "data/weekly_processed.csv"
OUTPUT_FILE = "weekly_backtest_results.csv"


def get_latest_prediction_file():
    files = sorted(
        (f for f in os.listdir(LOG_DIR) if f.startswith("weekly_predictions_") and f.endswith(".csv")),
        key=lambda x: pd.to_datetime(x.replace("weekly_predictions_", "").replace(".csv", ""))
    )
    if not files:
        raise FileNotFoundError("[ERROR] No weekly prediction logs found.")
    return files[-1]


def backtest_weekly():
    latest_pred_file = get_latest_prediction_file()
    predicted_for = latest_pred_file.replace("weekly_predictions_", "").replace(".csv", "")

    print(f"[INFO] Backtesting predictions made for week: {predicted_for}")
    pred_df = pd.read_csv(os.path.join(LOG_DIR, latest_pred_file))

    actual_df = pd.read_csv(DATA_PATH)

    # Get actuals for the *next* week
    next_week_dt = pd.to_datetime(predicted_for) + pd.Timedelta(days=7)
    next_week_str = next_week_dt.strftime("%Y-%m-%d")

    next_week_df = actual_df[actual_df["week"] == next_week_str]
    actual_close_map = next_week_df.set_index("symbol")["close_price"].to_dict()

    pred_df["actual_close"] = pred_df["symbol"].map(actual_close_map)
    pred_df.dropna(subset=["actual_close"], inplace=True)

    pred_df["correct_prediction"] = (
        pred_df["actual_close"] > pred_df["last_week_close"]
    ).astype(int)

    pred_df["percent_move"] = (
        (pred_df["actual_close"] - pred_df["last_week_close"]) / pred_df["last_week_close"] * 100
    ).round(2)

    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    print(f"\n[BACKTEST RESULTS for prediction week → {predicted_for}, actual week → {next_week_str}]")
    print(pred_df[["symbol", "last_week_close", "actual_close", "percent_move", "confidence", "correct_prediction"]])

    total = len(pred_df)
    correct = pred_df["correct_prediction"].sum()
    accuracy = correct / total if total else 0

    print(f"\nTotal Predictions: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    percent_threshold = 4
    qualified = pred_df[pred_df["percent_move"] > percent_threshold]
    if not qualified.empty:
        min_conf = qualified["confidence"].min()
        print(f"\n[INFO] Minimum confidence among >{percent_threshold}% movers: {min_conf:.4f}")
        print(qualified[["symbol", "percent_move", "confidence"]])
    else:
        print(f"\n[INFO] No predictions had a percent move greater than {percent_threshold}%")

    pred_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Weekly backtest results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    backtest_weekly()
