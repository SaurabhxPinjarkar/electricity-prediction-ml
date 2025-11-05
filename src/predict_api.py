"""
FastAPI Prediction Endpoint

Provides a REST API for electricity consumption predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model_utils import load_model, prepare_input_features, validate_input_ranges


# Initialize FastAPI app
app = FastAPI(
    title="Electricity Consumption Prediction API",
    description="API for predicting electricity consumption based on time and weather features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model_data = None


@app.on_event("startup")
async def load_model_startup():
    """Load model when API starts."""
    global model_data
    model_path = Path(__file__).parent / 'model.pkl'
    
    if model_path.exists():
        model_data = load_model(model_path)
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  Warning: No model found. Please train a model first.")


class PredictionInput(BaseModel):
    """Input schema for prediction."""
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    temperature_celsius: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(default=3.0, ge=0, le=50, description="Wind speed in m/s")
    pressure: float = Field(default=1013.0, ge=950, le=1050, description="Atmospheric pressure in hPa")
    is_weekend: bool = Field(default=False, description="Is it a weekend?")
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "hour": 14,
                "temperature_celsius": 22.5,
                "humidity": 65.0,
                "wind_speed": 4.5,
                "pressure": 1015.0,
                "is_weekend": False,
                "month": 6,
                "day_of_week": 2
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    predicted_consumption_MW: float = Field(..., description="Predicted electricity consumption in MW")
    input_features: dict = Field(..., description="Input features used for prediction")
    model_info: dict = Field(..., description="Information about the model")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Electricity Consumption Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_data is not None,
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "model_info": "/model-info (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None
    }


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    return {
        "model_type": "RandomForestRegressor",
        "n_features": len(model_data.get('feature_columns', [])),
        "feature_columns": model_data.get('feature_columns', []),
        "target_column": model_data.get('target_column', 'unknown')
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a prediction for electricity consumption.
    
    Args:
        input_data: Input features for prediction
        
    Returns:
        Predicted consumption and metadata
    """
    # Check if model is loaded
    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        # Validate input ranges
        validate_input_ranges(
            hour=input_data.hour,
            temp_celsius=input_data.temperature_celsius,
            humidity=input_data.humidity,
            wind_speed=input_data.wind_speed
        )
        
        # Prepare features
        feature_array = prepare_input_features(
            hour=input_data.hour,
            temp_celsius=input_data.temperature_celsius,
            humidity=input_data.humidity,
            wind_speed=input_data.wind_speed,
            pressure=input_data.pressure,
            is_weekend=input_data.is_weekend,
            month=input_data.month,
            day_of_week=input_data.day_of_week,
            feature_columns=model_data['feature_columns'],
            scaler=model_data['scaler']
        )
        
        # Make prediction
        prediction = float(model_data['model'].predict(feature_array)[0])
        
        # Prepare response
        return PredictionOutput(
            predicted_consumption_MW=prediction,
            input_features=input_data.dict(),
            model_info={
                "model_type": "RandomForestRegressor",
                "n_features": len(model_data['feature_columns'])
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(inputs: list[PredictionInput]):
    """
    Make predictions for multiple inputs.
    
    Args:
        inputs: List of input features
        
    Returns:
        List of predictions
    """
    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    predictions = []
    
    for input_data in inputs:
        try:
            result = await predict(input_data)
            predictions.append(result.dict())
        except Exception as e:
            predictions.append({"error": str(e)})
    
    return {"predictions": predictions, "count": len(predictions)}


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting Electricity Consumption Prediction API")
    print("="*60)
    print("\n📡 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
