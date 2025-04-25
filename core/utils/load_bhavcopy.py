import pandas as pd

def load_bhavcopy(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]

    # 🧠 Optional debug lines
    # print("Available columns:", df.columns.tolist())

    if 'series' in df.columns:
        df = df[df['series'].str.strip() == 'EQ']

    required = ['symbol', 'open_price', 'high_price', 'low_price', 'close_price', 'ttl_trd_qnty']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in bhavcopy: {missing}")

    df = df[required]

    # Rename columns to consistent internal format
    rename_map = {
        'open_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'close_price': 'close',
        'ttl_trd_qnty': 'volume'
    }
    df.rename(columns=rename_map, inplace=True)

    return df