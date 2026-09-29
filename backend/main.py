"""
FastAPI app.

    uvicorn backend.main:app --reload --port 8000
    open http://127.0.0.1:8000
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend import predictor

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

app = FastAPI(title="Crop Predictor", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    coordinates: Optional[str] = None       # "lat, lon"
    ref_date: dt.date
    ground_truth: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
@app.post("/api/predict")
def predict(req: PredictRequest):
    lat, lon = req.lat, req.lon
    if (lat is None or lon is None) and req.coordinates:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", req.coordinates)
        if len(nums) >= 2:
            lat, lon = float(nums[0]), float(nums[1])
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return JSONResponse({"error": "Enter valid coordinates as: lat, lon"}, status_code=422)

    try:
        from backend.gee_fetch import fetch_ndvi
        df = fetch_ndvi(lat, lon, req.ref_date)
    except Exception as exc:                 # auth / network / GEE errors -> shown in the UI
        return JSONResponse({"error": f"Earth Engine error: {exc}"}, status_code=503)

    try:
        return predictor.predict_from_series(df, req.ref_date, req.ground_truth)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
