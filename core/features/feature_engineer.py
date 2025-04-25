# core/features/feature_engineer.py

import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame, predict_mode: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.sort_values(by=["symbol", "date"], inplace=True)

    if predict_mode:
        latest_day = df["date"].max()
        df = df[df["date"] == latest_day]

    # ──────────────────────────────────────────────
    # 📦 PHASE 1 – PRICE & VOLUME CORE FEATURES
    # ──────────────────────────────────────────────

    df["price_change_t-1"] = df.groupby("symbol")["close"].pct_change()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["deliverable_qty"] = pd.to_numeric(df["deliverable_qty"], errors="coerce")
    df["deliv_ratio"] = df["deliverable_qty"] / df["volume"]

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["range_compression_ratio"] = df.groupby("symbol")["hl_range"].transform(lambda x: x / x.rolling(3).mean())
    df["gap_pct"] = df["open"] / df.groupby("symbol")["close"].shift(1) - 1
    df["body_to_range_ratio"] = (df["close"] - df["open"]).abs() / df["hl_range"]
    df["atr_5"] = df.groupby("symbol")["hl_range"].transform(lambda x: x.rolling(5).mean())
    df["atr_volatility"] = df.groupby("symbol")["hl_range"].transform(lambda x: x.rolling(5).std())

    # ──────────────────────────────────────────────
    # 📈 PHASE 2 – TREND, MOMENTUM & POSITIONING
    # ──────────────────────────────────────────────

    df["return_3d"] = df.groupby("symbol")["close"].pct_change(periods=3)
    df["return_5d"] = df.groupby("symbol")["close"].pct_change(periods=5)
    df["distance_from_ema_5"] = df["close"] - df.groupby("symbol")["close"].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    high_5d = df.groupby("symbol")["high"].transform(lambda x: x.rolling(5).max())
    low_5d = df.groupby("symbol")["low"].transform(lambda x: x.rolling(5).min())
    df["position_in_range_5d"] = (df["close"] - low_5d) / (high_5d - low_5d)

    # ──────────────────────────────────────────────
    # 🕯️ PHASE 3 – CANDLE ANATOMY & TREND STRENGTH
    # ──────────────────────────────────────────────

    df["lower_wick_pct"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["close"]
    df["upper_wick_pct"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["close"]

    def calc_slope(x):
        if len(x) < 5 or x.isna().any():
            return np.nan
        y = x.values
        x_idx = np.arange(5)
        slope, _ = np.polyfit(x_idx, y, 1)
        return slope
    df["slope_close_5d"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(5).apply(calc_slope, raw=False))

    # ──────────────────────────────────────────────
    # 🔀 PHASE 4 – COMPOSITE & DIVERGENCE SIGNALS
    # ──────────────────────────────────────────────

    df["volume_spike_ratio"] = df["volume"] / df.groupby("symbol")["volume"].transform(lambda x: x.rolling(3).mean())
    df["momentum_divergence"] = df["return_3d"] / df["volume_spike_ratio"]

    # ──────────────────────────────────────────────
    # 🧠 PHASE 5 – CONVICTION SIGNALS
    # ──────────────────────────────────────────────

    df["closing_strength"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

    # ──────────────────────────────────────────────
    # 🚀 PHASE 6 – VOLATILITY & TREND ZONE SIGNALS
    # ──────────────────────────────────────────────

    # Bollinger Band Width normalized
    rolling_mean = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20).mean())
    rolling_std = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20).std())
    upper_bb = rolling_mean + (2 * rolling_std)
    lower_bb = rolling_mean - (2 * rolling_std)
    bb_width = (upper_bb - lower_bb) / rolling_mean
    df["volatility_squeeze"] = bb_width / bb_width.rolling(20).mean()

    # ADX-like trend strength approximation (directional movement range)
    up_move = df.groupby("symbol")["high"].diff()
    down_move = df.groupby("symbol")["low"].diff().abs()
    dm = np.where(up_move > down_move, up_move, 0)
    df["trend_zone_strength"] = pd.Series(dm).rolling(14).mean()

    # ──────────────────────────────────────────────
    # 🎯 FINAL – LABEL ASSIGNMENT
    # ──────────────────────────────────────────────

    if not predict_mode:
        df["next_close"] = df.groupby("symbol")["close"].shift(-1)
        df["target"] = (df["next_close"] > df["close"]).astype(int)
        df.drop(columns=["next_close"], inplace=True)
        df.dropna(subset=["target"], inplace=True)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df
