import pandas as pd

def load_bhavcopy(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]

    # print("Available columns:", df.columns.tolist())  # 🧠 Debug line

    if 'series' in df.columns:
        df = df[df['series'] == ' EQ']

    required = ['symbol', 'open_price', 'high_price', 'low_price', 'close_price', 'ttl_trd_qnty']
    selected = [col for col in required if col in df.columns]

    # print("Using columns:", selected)  # 🧠 Debug line

    df = df[selected]
    return df
