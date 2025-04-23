# utils/aggregate_weekly.py
import pandas as pd

def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], format="%d%m%Y")
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly = df.groupby(['symbol', 'week']).agg({
        'open_price': 'first',
        'high_price': 'max',
        'low_price': 'min',
        'close_price': 'last',
        'ttl_trd_qnty': 'sum'
    }).reset_index()

    return weekly
