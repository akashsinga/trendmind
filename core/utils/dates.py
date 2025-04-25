from datetime import datetime, timedelta
from core.utils.market_calendar import load_nse_holidays

def get_next_trading_day(current_date_str):
    current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
    year = current_date.year
    holidays = load_nse_holidays(year)

    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.strftime("%Y-%m-%d") in holidays:
        next_day += timedelta(days=1)

        # If we cross into the next year, reload holidays
        if next_day.year != year:
            year = next_day.year
            holidays = load_nse_holidays(year)

    return next_day.strftime("%d%m%Y")