# api/routes/backtest.py

from fastapi import APIRouter
from api.services.backtest_service import run_daily_backtest_service

router = APIRouter()

@router.post("/backtest/daily")
async def backtest_daily():
  response = run_daily_backtest_service()
  return response