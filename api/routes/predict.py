# api/routes/predict.py

from fastapi import APIRouter
from api.services.predict_service import run_daily_prediction_service

router = APIRouter()

@router.post("/predict/daily")
async def predict_daily():
  response = run_daily_prediction_service()
  return response