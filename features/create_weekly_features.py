# features/create_weekly_features.py
import pandas as pd

def create_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.sort_values(by=["symbol", "week"], inplace=True)

    # Weekly return
    df["weekly_pct_change"] = df.groupby("symbol")["close_price"].pct_change()

    # Lag features
    df["close_t-1"] = df.groupby("symbol")["close_price"].shift(1)
    df["close_t-2"] = df.groupby("symbol")["close_price"].shift(2)
    df["volume_t-1"] = df.groupby("symbol")["ttl_trd_qnty"].shift(1)

    # 3-week average volume
    df["volume_3w_avg"] = df.groupby("symbol")["ttl_trd_qnty"].transform(lambda x: x.rolling(3).mean())
    df["volume_spike_ratio"] = df["ttl_trd_qnty"] / df["volume_3w_avg"]

    # Weekly range compression
    df["hl_range"] = df["high_price"] - df["low_price"]
    df["range_3w_avg"] = df.groupby("symbol")["hl_range"].transform(lambda x: x.rolling(3).mean())
    df["range_compression_ratio"] = df["hl_range"] / df["range_3w_avg"]

    # Phase 2 features
    df["atr_5"] = df.groupby("symbol")["hl_range"].transform(lambda x: x.rolling(5).mean())
    df["gap_pct"] = df["open_price"] / df.groupby("symbol")["close_price"].shift(1) - 1
    df["body_to_range_ratio"] = (df["close_price"] - df["open_price"]).abs() / df["hl_range"]


    # # Target: next week's close higher?
    df["next_close"] = df.groupby("symbol")["close_price"].shift(-1)
    df["target"] = (df["next_close"] > df["close_price"]).astype(int)

    df.drop(columns=["next_close"], inplace=True)
    df.dropna(inplace=True)

    return df
