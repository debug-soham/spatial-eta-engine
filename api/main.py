import os
import sys
# Add parent directory to path to enable importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any

from src.dataset import haversine_distance, save_synthetic_data
from src.spatial_features import coord_to_h3
from src.explainability import explain_prediction
from src.train import train_and_evaluate

app = FastAPI(
    title="Spatial-Temporal Delivery Delay & ETA Engine",
    description="Production-grade API for food/courier delivery arrival time estimation and delay risk forecasting",
    version="1.0.0"
)

# Paths
MODEL_PATH = "models/eta_pipeline.joblib"
DATA_PATH = "data/delivery_records.csv"

# Global model state
pipeline_payload = None

def load_pipeline():
    global pipeline_payload
    if not os.path.exists(MODEL_PATH):
        print(f"Model pipeline not found at {MODEL_PATH}. Initiating automatic training...")
        if not os.path.exists(DATA_PATH):
            save_synthetic_data(num_records=15000)
        train_and_evaluate()
    pipeline_payload = joblib.load(MODEL_PATH)
    print("Model pipeline successfully loaded into memory.")

@app.on_event("startup")
def startup_event():
    load_pipeline()

class ETARequest(BaseModel):
    pickup_latitude: float = Field(..., description="Latitude of pickup location", ge=-90, le=90)
    pickup_longitude: float = Field(..., description="Longitude of pickup location", ge=-180, le=180)
    dropoff_latitude: float = Field(..., description="Latitude of dropoff location", ge=-90, le=90)
    dropoff_longitude: float = Field(..., description="Longitude of dropoff location", ge=-180, le=180)
    order_timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        description="Timestamp of order placement (YYYY-MM-DD HH:MM:SS)"
    )
    merchant_prep_time_min: float = Field(default=15.0, description="Estimated merchant preparation time in minutes", ge=0)
    active_driver_ratio: float = Field(default=1.0, description="Active driver density ratio (drivers / pending orders)", ge=0)
    rain_level_mm_hr: float = Field(default=0.0, description="Rainfall level in mm/hr", ge=0)

class ETAResponse(BaseModel):
    pickup_h3_cell: str
    dropoff_h3_cell: str
    distance_km: float
    predicted_eta_min: float
    delay_probability: float
    is_high_delay_risk: bool
    explainability: Dict[str, Any]

@app.get("/health")
def health():
    if pipeline_payload is None:
        return {"status": "unhealthy", "message": "Model pipeline not loaded"}
    
    return {
        "status": "healthy",
        "model_metadata": {
            "delay_threshold_min": pipeline_payload.get("delay_threshold_min"),
            "features_used": pipeline_payload.get("feature_names"),
            "metrics": pipeline_payload.get("metrics")
        }
    }

@app.post("/predict_eta", response_model=ETAResponse)
def predict_eta(req: ETARequest):
    if pipeline_payload is None:
        raise HTTPException(status_code=503, detail="Model pipeline is currently unavailable.")
    
    try:
        # 1. Parse timestamps and calculate spatial properties
        dist_km = haversine_distance(
            req.pickup_latitude, req.pickup_longitude,
            req.dropoff_latitude, req.dropoff_longitude
        )
        # Clip distance as in training
        dist_km = float(np.clip(dist_km, 0.2, 15.0))
        
        pickup_cell = coord_to_h3(req.pickup_latitude, req.pickup_longitude, 8)
        dropoff_cell = coord_to_h3(req.dropoff_latitude, req.dropoff_longitude, 8)
        
        # 2. Re-create dataframe row matching the features expected by the training pipeline
        input_data = pd.DataFrame([{
            'pickup_latitude': req.pickup_latitude,
            'pickup_longitude': req.pickup_longitude,
            'dropoff_latitude': req.dropoff_latitude,
            'dropoff_longitude': req.dropoff_longitude,
            'order_timestamp': req.order_timestamp,
            'distance_km': dist_km,
            'merchant_prep_time_min': req.merchant_prep_time_min,
            'active_driver_ratio': req.active_driver_ratio,
            'rain_level_mm_hr': req.rain_level_mm_hr
        }])
        
        # Extract components from payload
        preprocessor = pipeline_payload['preprocessor']
        regressor = pipeline_payload['regressor']
        classifier = pipeline_payload['classifier']
        
        # 3. Process features through sklearn pipeline
        input_processed = preprocessor.transform(input_data)
        
        # 4. Predict
        eta_pred = float(regressor.predict(input_processed)[0])
        # Ensure predicted ETA is physically reasonable
        eta_pred = max(eta_pred, 5.0)
        
        delay_prob = float(classifier.predict_proba(input_processed)[0, 1])
        
        # High delay risk indicator
        is_high_risk = delay_prob >= 0.50 or eta_pred > 35.0
        
        # 5. Local explainability
        exp = explain_prediction(pipeline_payload, input_data)
        
        return ETAResponse(
            pickup_h3_cell=pickup_cell,
            dropoff_h3_cell=dropoff_cell,
            distance_km=round(dist_km, 3),
            predicted_eta_min=round(eta_pred, 2),
            delay_probability=round(delay_prob, 4),
            is_high_delay_risk=is_high_risk,
            explainability=exp
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
