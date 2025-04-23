import os
import pandas as pd
from utils.load_bhavcopy import load_bhavcopy

LOG_DIR = "logs"
DATA_DIR = "data/bhavcopies"
OUTPUT_FILE = "backtest_results.csv"


def get_latest_prediction_file():
    files = sorted(
        (f for f in os.listdir(LOG_DIR) if f.startswith("predictions_") and f.endswith(".csv")),
        key=lambda x: pd.to_datetime(x.replace("predictions_", "").replace(".csv", ""), format="%d%m%Y")
    )
    if not files:
        raise FileNotFoundError("[ERROR] No prediction logs found.")
    return files[-1]


def backtest():
    latest_pred_file = get_latest_prediction_file()
    predict_for = latest_pred_file.replace("predictions_", "").replace(".csv", "")

    print(f"[INFO] Using predictions from {latest_pred_file}")
    pred_df = pd.read_csv(os.path.join(LOG_DIR, latest_pred_file))

    actual_path = os.path.join(DATA_DIR, f"{predict_for}.csv")
    if not os.path.exists(actual_path):
        print(f"[ERROR] Bhavcopy not found for predicted day: {predict_for}")
        return

    print(f"[INFO] Loading actual bhavcopy from {predict_for}.csv...")
    actual_df = load_bhavcopy(actual_path)

    actual_close_map = actual_df.set_index("symbol")["close_price"].to_dict()
    pred_df["actual_close"] = pred_df["symbol"].map(actual_close_map)
    pred_df.dropna(subset=["actual_close"], inplace=True)

    pred_df["correct_prediction"] = (pred_df["actual_close"] > pred_df["last_close_price"]).astype(int)
    pred_df["percent_move"] = (
        (pred_df["actual_close"] - pred_df["last_close_price"]) / pred_df["last_close_price"] * 100
    ).round(2)

    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    print(f"\n[BACKTEST RESULTS for {predict_for}]")
    print(pred_df[["symbol", "last_close_price", "actual_close", "percent_move", "confidence", "correct_prediction"]])

    total = len(pred_df)
    correct = pred_df["correct_prediction"].sum()
    accuracy = correct / total if total else 0

    print("\n[SUMMARY]")
    print(f"Total Predictions Made:    {total}")
    print(f"Correct Predictions:       {correct}")
    print(f"Accuracy:                  {accuracy:.2%}")

    percent_threshold = 4
    qualified = pred_df[pred_df["percent_move"] > percent_threshold]
    if not qualified.empty:
        min_conf = qualified["confidence"].min()
        print(f"\n[INFO] Stocks with >{percent_threshold}% move:")
        print(f"Minimum Confidence Among Them: {min_conf:.4f}")
        print(qualified[["symbol", "percent_move", "confidence"]])
    else:
        print(f"\n[INFO] No predicted stocks moved more than {percent_threshold}% on {predict_for}")

    pred_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Backtest results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    backtest()
