def add_features(df):
    # Calculate previous day's percentage change
    df['price_change_t-1'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # Previous 2 days' close data
    df['close_t-2'] = df.groupby('symbol')['close'].shift(2)
    
    # Calculate rolling averages (e.g., 5-day moving average)
    df['sma_5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())

    # Drop NaN rows after adding new features
    df = df.dropna()

    return df
