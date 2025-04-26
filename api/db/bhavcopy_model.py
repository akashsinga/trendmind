# api/db/bhavcopy_model.py

from sqlalchemy import Column, Integer, String, Float, Date
from api.db.database import Base

class Bhavcopy(Base):
    __tablename__ = "bhavcopies"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    deliverable_qty = Column(Integer)
