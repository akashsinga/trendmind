# api/routes/predict.py

from fastapi import APIRouter
from api.services.predict_service import run_daily_prediction_service, get_latest_predictions

router = APIRouter()

@router.post("/predictions/trigger/daily")
async def predict_daily():
    response = run_daily_prediction_service()
    return response

@router.get("/predictions/latest")
async def latest_predictions():
    response = get_latest_predictions()
    return response
