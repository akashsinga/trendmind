#core/utils/aggregate_weekly.py
import pandas as pd

def aggregate_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    df_weekly = (
        df.groupby(["symbol", pd.Grouper(freq="W-MON")])
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .reset_index()
        .dropna()
    )
    return df_weekly
