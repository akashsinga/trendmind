# api/services/predict_service.py

from core.predictor.predict_ensemble import run_ensemble_prediction
from sqlalchemy.orm import Session
from api.db.database import SessionLocal
from api.db.prediction_model import PredictionDaily
from datetime import date

def run_daily_prediction_service():
  try:
    run_ensemble_prediction()
    return {'status': 'success', 'message': 'Daily prediction completed'}
  except Exception as e:
    return {'status': 'error', 'message': str(e)}
  

def get_latest_predictions():
  session: Session = SessionLocal()
  
  try:
    today = date.today()
    preds = session.query(PredictionDaily).filter(PredictionDaily.predicted_for_date == today).all()

    result = [
        {
            "symbol": pred.symbol,
            "date": pred.date.strftime("%Y-%m-%d"),
            "prediction": pred.prediction,
            "confidence": pred.confidence,
            "predicted_for_date": pred.predicted_for_date.strftime("%Y-%m-%d")
        }
        for pred in preds
    ]
    return {"status": "success", "data": result}
  except Exception as e:
    return {'status': 'error', 'message': str(e)}
  finally:
    session.close()