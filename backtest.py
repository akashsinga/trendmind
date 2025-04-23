# backtest.py
import os
import pandas as pd
import joblib
from features.create_features import create_features
from utils.load_bhavcopy import load_bhavcopy
from utils.load_multiple_bhavcopies import load_multiple_bhavcopies

MODEL_PATH = "models/random_forest_model.pkl"
DATA_DIR = "data/bhavcopies"
OUTPUT_FILE = "backtest_results.csv"
PREDICT_FOR = None  # Will be set based on the latest file


def load_custom_days_exclude_latest(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if len(files) < 2:
        raise ValueError("Need at least two bhavcopy files to perform backtest.")

    *train_files, actual_file = files
    global PREDICT_FOR
    PREDICT_FOR = actual_file.replace(".csv", "")

    dfs = []
    for file in train_files:
        path = os.path.join(data_dir, file)
        df = load_bhavcopy(path)
        df["date"] = file.replace(".csv", "")
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True), actual_file


def backtest():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found. Please train it first.")
        return

    print("[INFO] Loading bhavcopies...")
    df, actual_file = load_custom_days_exclude_latest(DATA_DIR)
    processed_df = create_features(df)

    if processed_df.empty:
        print("[WARNING] No data after feature creation.")
        return

    print("[DEBUG] Processed DataFrame shape:", processed_df.shape)
    print("[DEBUG] Dates available:", processed_df["date"].unique())

    X = processed_df.drop(columns=["symbol", "date", "target"])
    model = joblib.load(MODEL_PATH)

    probas = model.predict_proba(X)
    predictions = model.predict(X)
    confidence = probas[:, 1]

    processed_df["prediction"] = predictions
    processed_df["confidence"] = confidence

    last_training_day = processed_df["date"].max()
    pred_df = processed_df[processed_df["prediction"] == 1]
    pred_df = pred_df[pred_df["date"] == last_training_day]  # Predicting for PREDICT_FOR

    if pred_df.empty:
        print(f"[INFO] No predicted upward moves for {PREDICT_FOR}.")
        return

    # Load actual close values for the day being predicted
    actual_path = os.path.join(DATA_DIR, actual_file)
    if os.path.exists(actual_path):
        actual_df = load_bhavcopy(actual_path)
        actual_close_map = actual_df.set_index("symbol")["close_price"].to_dict()
        pred_df["actual_close"] = pred_df["symbol"].map(actual_close_map)
        pred_df["correct_prediction"] = (
            pred_df["actual_close"] > pred_df["close_price"]
        ).astype(int)
        pred_df["percent_move"] = (
            (pred_df["actual_close"] - pred_df["close_price"]) / pred_df["close_price"] * 100
        ).round(2)

    pred_df = pred_df[["symbol", "date", "close_price", "confidence"] + (["actual_close", "percent_move", "correct_prediction"] if "actual_close" in pred_df else [])]
    pred_df.rename(columns={"close_price": "last_close_price"}, inplace=True)
    pred_df.sort_values(by="confidence", ascending=False, inplace=True)

    print(f"\n[BACKTEST RESULTS for {PREDICT_FOR}]")
    print(pred_df)

    if "correct_prediction" in pred_df:
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
            print(qualified[["symbol", "last_close_price", "actual_close", "percent_move", "confidence", "correct_prediction"]])
        else:
            print(f"\n[INFO] No predictions had a percent move greater than {percent_threshold}%")

    pred_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Backtest results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    backtest()
