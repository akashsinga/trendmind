# api/db/prediction_model.py

from sqlalchemy import Column, Integer, String, Float, Date
from api.db.base_class import Base

class PredictionDaily(Base):
    __tablename__ = "predictions_daily"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)            # Bhavcopy date
    prediction = Column(String)                 # "bullish" or "bearish"
    confidence = Column(Float)
    predicted_for_date = Column(Date, index=True)   # For which day prediction was made
