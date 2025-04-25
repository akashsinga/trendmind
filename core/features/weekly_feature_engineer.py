# core/features/weekly_feature_engineer.py

import pandas as pd

def create_weekly_features(df: pd.DataFrame, predict_mode: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["symbol", "date"], inplace=True)

    if predict_mode:
        latest_week = df["date"].max()
        df = df[df["date"] == latest_week]

    # Weekly features
    df["close_t-1"] = df.groupby("symbol")["close"].shift(1)
    df["sma_3"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(3).mean())
    df["price_change_t-1"] = df.groupby("symbol")["close"].pct_change()

    df["volume_t-1"] = df.groupby("symbol")["volume"].shift(1)
    df["volume_2w_avg"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(2).mean())
    df["volume_spike_ratio"] = df["volume"] / df["volume_2w_avg"]

    df["hl_range"] = df["high"] - df["low"]
    df["range_2w_avg"] = df.groupby("symbol")["hl_range"].transform(lambda x: x.rolling(2).mean())
    df["range_compression_ratio"] = df["hl_range"] / df["range_2w_avg"]

    df["atr_3"] = df.groupby("symbol")["hl_range"].transform(lambda x: x.rolling(3).mean())
    df["gap_pct"] = df["open"] / df.groupby("symbol")["close"].shift(1) - 1
    df["body_to_range_ratio"] = (df["close"] - df["open"]).abs() / df["hl_range"]

    if not predict_mode:
        df["next_close"] = df.groupby("symbol")["close"].shift(-1)
        df["target"] = (df["next_close"] > df["close"]).astype(int)
        df.drop(columns=["next_close"], inplace=True)
        df.dropna(subset=["target"], inplace=True)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df
