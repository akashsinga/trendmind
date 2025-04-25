# core/config.py

from datetime import datetime

# === Data Source ===
DATA_DIR = "data/bhavcopies"

# === Daily Paths ===
DAILY_MODEL_PATH = "models/random_forest_model.pkl"
DAILY_OUTPUT_DIR = "outputs/daily"
DAILY_OUTPUT_PATH = f"{DAILY_OUTPUT_DIR}/{datetime.now().date()}.csv"
DAILY_LATEST_PATH = f"{DAILY_OUTPUT_DIR}/latest.csv"
DAILY_PROCESSED_PATH = "data/processed_data.csv"

# === Weekly Paths ===
WEEKLY_MODEL_PATH = "models/weekly_random_forest.pkl"
WEEKLY_OUTPUT_DIR = "outputs/weekly"
WEEKLY_OUTPUT_PATH = f"{WEEKLY_OUTPUT_DIR}/{datetime.now().date()}.csv"
WEEKLY_LATEST_PATH = f"{WEEKLY_OUTPUT_DIR}/latest.csv"
WEEKLY_PROCESSED_PATH = "data/weekly_processed.csv"

# === Prediction Filtering ===
CONFIDENCE_THRESHOLD = 0.6
CONFIDENCE_BUCKETS = [(0.9, 1.0), (0.7, 0.9), (0.5, 0.7)]

# === Backtest Config ===
DAILY_PREDICTIONS_DIR = "outputs/daily"
DAILY_BACKTEST_OUTPUT = "outputs/backtest_results.csv"
WEEKLY_PREDICTIONS_DIR = "outputs/weekly"
WEEKLY_BACKTEST_OUTPUT = "outputs/weekly_backtest_results.csv"
WEEKLY_PROCESSED_PATH = "data/weekly_processed.csv"
PERCENT_MOVE_THRESHOLD = 4

