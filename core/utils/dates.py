from datetime import datetime, timedelta

def get_next_trading_day(current_date_str):
  current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
  next_day = current_date + timedelta(days=1)
  
  while next_day.weekday() >=5:
    next_day += timedelta(days=1)
    
  
  return next_day.strftime("%d%m%Y")