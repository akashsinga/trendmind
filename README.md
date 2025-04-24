# 🧠 TrendMind: Stock Prediction Engine (Daily & Weekly)

TrendMind is a machine learning pipeline designed to predict high-confidence bullish moves in the Indian stock market using historical bhavcopy data. It is tailored for **BTST and swing trading** setups, focusing on **technical patterns** and **price-action behavior**.

---

## 🚀 Features

- 📅 Supports both **Daily & Weekly prediction models**
- 📊 Feature engineering includes:
  - Price momentum (returns, SMA)
  - Volume spikes and compression
  - Range compression and volatility (ATR)
  - Gap detection and candle strength
- 🔍 Powered by **RandomForestClassifier**
- 📈 Confidence-scored predictions for ranking trade ideas
- 🗂️ Prediction journal saved automatically by date
- 🧪 Backtesting available for both daily and weekly moves
- ✅ Fully offline — uses bhavcopy CSVs

---

## 🛠️ Folder Structure

.
├── data/
│   ├── bhavcopies/             # Raw bhavcopy CSVs (e.g., 01012025.csv)
│   ├── processed_data.csv      # Output from daily pipeline
│   └── weekly_processed.csv    # Output from weekly pipeline
│
├── models/                     # Trained model files (RandomForest .pkl)
│
├── logs/                       # Prediction logs saved by date
│
├── features/
│   ├── create_features.py          # Daily feature engineering
│   └── create_weekly_features.py   # Weekly feature engineering
│
├── utils/
│   ├── load_bhavcopy.py
│   ├── load_multiple_bhavcopies.py
│   └── aggregate_weekly.py
│
├── main.py                     # Create daily processed data
├── train.py                    # Train RandomForest model (daily)
├── predict.py                  # Predict top daily bullish stocks
│
├── weekly_main.py              # Create weekly OHLCV from daily
├── weekly_train.py             # Train RandomForest model (weekly)
├── weekly_predict.py           # Predict top bullish stocks for next week
│
├── backtest.py                 # Backtest daily predictions using next-day bhavcopy
└── weekly_backtest.py          # Backtest weekly predictions using next-week bhavcopy

---

## ✅ Usage

### 1. Drop Bhavcopies
Save daily bhavcopies in the `data/bhavcopies/` folder. File names should follow this format: `DDMMYYYY.csv`


---

### 2. Run Daily Pipeline
```bash
python main.py             # Create features from daily bhavcopies
python train.py            # Train RandomForestClassifier
python predict.py          # Predict for the next trading day

python weekly_main.py      # Aggregate daily data into weekly OHLCV
python weekly_train.py     # Train on weekly feature set
python weekly_predict.py   # Predict next week’s likely bullish moves

python backtest.py         # Check how accurate daily predictions were
python weekly_backtest.py  # Check accuracy of weekly predictions