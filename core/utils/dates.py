from datetime import datetime, timedelta
import os
import json

# Constants
STATIC_DIR = "core/static"

def load_holidays(year):
    file_path = os.path.join(STATIC_DIR, f"holidays_{year}.json")

    if not os.path.exists(file_path):
        print(f"[ERROR] Holidays file not found for {year}.")
        return []

    with open(file_path, "r") as f:
        holidays = json.load(f)

    return holidays

def get_next_trading_day(current_date_str):
    current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
    year = current_date.year
    holidays = load_holidays(year)

    next_day = current_date + timedelta(days=1)

    while next_day.weekday() >= 5 or next_day.strftime("%Y-%m-%d") in holidays:
        next_day += timedelta(days=1)

        # If we cross into the next year, reload holidays
        if next_day.year != year:
            year = next_day.year
            holidays = load_holidays(year)

    return next_day.strftime("%d%m%Y")
