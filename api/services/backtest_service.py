# api/services/backtest_service.py

from core.backtest.backtest_ensemble import run_ensemble_backtest

def run_daily_backtest_service():
  try:
    run_ensemble_backtest()
    return {'status': 'success', 'message': 'Daily backtest completed'}
  except Exception as e:
    return {'status': 'error', 'message': str(e)}