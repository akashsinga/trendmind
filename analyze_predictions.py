# analyze_predictions.py
import os
import pandas as pd
from utils.load_bhavcopy import load_bhavcopy

LOG_DIR = "logs"
BHAVCOPY_DIR = "data/bhavcopies"


def extract_date_from_filename(filename):
    return filename.replace("predictions_", "").replace(".csv", "")


def analyze_predictions():
    prediction_files = [f for f in os.listdir(LOG_DIR) if f.startswith("predictions_") and f.endswith(".csv")]
    prediction_files.sort()

    all_results = []

    for file in prediction_files:
        predicted_date = extract_date_from_filename(file)
        prediction_path = os.path.join(LOG_DIR, file)
        bhavcopy_path = os.path.join(BHAVCOPY_DIR, f"{predicted_date}.csv")

        if not os.path.exists(bhavcopy_path):
            print(f"[SKIP] Missing bhavcopy for {predicted_date}")
            continue

        pred_df = pd.read_csv(prediction_path)
        actual_df = load_bhavcopy(bhavcopy_path)
        actual_close_map = actual_df.set_index("symbol")["close_price"].to_dict()

        pred_df["actual_close"] = pred_df["symbol"].map(actual_close_map)
        pred_df.dropna(subset=["actual_close"], inplace=True)

        pred_df["correct"] = (pred_df["actual_close"] > pred_df["last_close_price"]).astype(int)
        pred_df["percent_move"] = (
            (pred_df["actual_close"] - pred_df["last_close_price"]) / pred_df["last_close_price"] * 100
        ).round(2)

        total = len(pred_df)
        correct = pred_df["correct"].sum()
        accuracy = correct / total if total else 0
        avg_conf = pred_df["confidence"].mean()
        avg_move = pred_df["percent_move"].mean()

        all_results.append({
            "date": predicted_date,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy * 100, 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_percent_move": round(avg_move, 2),
        })

    summary_df = pd.DataFrame(all_results)
    print("\n[PERFORMANCE SUMMARY]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    analyze_predictions()
