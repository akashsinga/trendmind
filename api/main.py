# api/main.py

from fastapi import FastAPI
from api.routes import train
from api.db.database import create_db_and_tables

app = FastAPI(title="TrendMind API",description="Backend API for stock movement prediction system",version="1.0.0")

app.include_router(train.router)

# Basic Health Check
@app.get("/status")
async def status():
    return {"status": "OK", "message": "Server running successfully"}
