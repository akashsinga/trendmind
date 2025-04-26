# api/main.py

from fastapi import FastAPI
from api.db.database import create_db_and_tables

app = FastAPI(title="TrendMind API",description="Backend API for stock movement prediction system",version="1.0.0")

# Initialize DB tables
@app.on_event("startup")
async def startup_event():
    pass

# Basic Health Check
@app.get("/status")
async def status():
    return {"status": "OK", "message": "Server running successfully"}
