# api/services/predict_service.py

from core.predictor.predict_ensemble import run_ensemble_prediction

def run_daily_prediction_service():
  try:
    run_ensemble_prediction()
    return {'status': 'success', 'message': 'Daily prediction completed'}
  except Exception as e:
    return {'status': 'error', 'message': str(e)}