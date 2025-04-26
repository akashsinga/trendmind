# api/db/backtest_model.py

from sqlalchemy import Column, Integer, String, Float, Date, Boolean
from api.db.base_class import Base

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    prediction_date = Column(Date, index=True)   # Date on which prediction was made
    predicted_direction = Column(String)         # "bullish" or "bearish"
    actual_direction = Column(String)            # "bullish" or "bearish"
    success = Column(Boolean)                    # 1 if prediction correct, 0 otherwise
