import os
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup

DATA_FOLDER = "data"

def get_holiday_file_path(year:int) -> str:
  return os.path.join(DATA_FOLDER, f"nse_holidays_{year}.json")

def fetch_and_save_nse_holidays(year:int):
  url = f"https://www.nseindia.com/resources/exchange-communication-holidays"
  headers = {"User-Agent": "Mozilla/5.0"}
    
  session = requests.Session()
  session.headers.update(headers)
  response = session.get(url)
  
  if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch NSE holidays: {response.status_code}")
    
  soup = BeautifulSoup(response.text, "html.parser")
  holiday_table = soup.find("table")
  
  if not holiday_table:
        raise RuntimeError("Could not find holidays table in NSE response")

  holidays = []
  for row in holiday_table.find_all("tr")[1:]:
      cols = row.find_all("td")
      if len(cols) >= 2:
          try:
              raw_date = cols[0].text.strip()
              parsed_date = datetime.strptime(raw_date, "%d-%b-%Y").strftime("%Y-%m-%d")
              if parsed_date.startswith(str(year)):
                  holidays.append(parsed_date)
          except Exception:
              continue

  # Save holidays to JSON
  os.makedirs(DATA_FOLDER, exist_ok=True)
  path = get_holiday_file_path(year)
  with open(path, "w") as f:
      json.dump(holidays, f, indent=2)
  
  return set(holidays)

def load_nse_holidays(year: int) -> set:
    path = get_holiday_file_path(year)
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    else:
        return fetch_and_save_nse_holidays(year)