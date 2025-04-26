#api/routes/train.py

from fastapi import APIRouter
from api.services.train_service import run_ensemble_training_service

router = APIRouter()

@router.post("/train/ensemble")
async def train_ensemble():
  response = run_ensemble_training_service()
  return response