from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZIPMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
from llm_service import LLMService
from typing import List
from pathlib import Path
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="ML and LLM API",
    description="Secure and performant ML and LLM inference API",
    version="2.0.0"
)

# Add GZIP compression middleware
app.add_middleware(GZIPMiddleware, minimum_size=1000)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Model path configuration
MODEL_PATH = Path(__file__).parent / 'model.joblib'

# Load ML model with error handling
ml_model = None
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    ml_model = joblib.load(MODEL_PATH)
    logger.info(f"ML model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading ML model: {e}")
    ml_model = None

# Initialize LLM service
llm_service = LLMService()

# Pydantic models with validation
class PredictionInput(BaseModel):
    """Input model for iris flower prediction.
    
    Features should be iris flower measurements:
    - sepal_length: 4.3 to 7.9
    - sepal_width: 2.0 to 4.4
    - petal_length: 1.0 to 6.9
    - petal_width: 0.1 to 2.5
    """
    features: List[float] = Field(
        ..., 
        min_items=4, 
        max_items=4,
        description="List of 4 float values representing iris flower features"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that features are within expected ranges."""
        if not all(0 <= f <= 10 for f in v):
            raise ValueError("All features must be between 0 and 10")
        return v

class BatchPredictionInput(BaseModel):
    """Input model for batch predictions."""
    features_list: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of feature arrays (max 1000 samples)"
    )
    
    @validator('features_list')
    def validate_batch(cls, v):
        """Validate each feature array in the batch."""
        for features in v:
            if len(features) != 4:
                raise ValueError("Each feature array must have exactly 4 elements")
            if not all(0 <= f <= 10 for f in features):
                raise ValueError("All features must be between 0 and 10")
        return v

class SentimentInput(BaseModel):
    """Input model for sentiment analysis."""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="Text to analyze for sentiment (max 5000 characters)"
    )

# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"message": "Welcome to the ML and LLM API", "version": "2.0.0"}

@app.post("/predict", tags=["ML"])
@limiter.limit("100/minute")
async def predict(request: Request, input_data: PredictionInput):
    """Predict iris flower class from features.
    
    Args:
        input_data: PredictionInput containing 4 float features
        
    Returns:
        Prediction result with class index (0, 1, or 2)
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if ml_model is None:
        logger.error("ML model not available")
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        features = np.array(input_data.features).reshape(1, -1)
        prediction = ml_model.predict(features)
        confidence = ml_model.predict_proba(features).max()
        
        logger.info(f"Prediction made: class {int(prediction[0])} with confidence {confidence:.3f}")
        
        return {
            "prediction": int(prediction[0]),
            "confidence": float(confidence),
            "classes": ["setosa", "versicolor", "virginica"]
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Invalid features or prediction failed")

@app.post("/predict-batch", tags=["ML"])
@limiter.limit("50/minute")
async def predict_batch(request: Request, input_data: BatchPredictionInput):
    """Batch predict iris flower classes.
    
    Args:
        input_data: BatchPredictionInput containing list of feature arrays
        
    Returns:
        List of predictions with confidences
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if ml_model is None:
        logger.error("ML model not available")
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        features = np.array(input_data.features_list)
        predictions = ml_model.predict(features)
        confidences = ml_model.predict_proba(features).max(axis=1)
        
        logger.info(f"Batch prediction: {len(predictions)} samples processed")
        
        return {
            "count": len(predictions),
            "predictions": predictions.tolist(),
            "confidences": confidences.tolist(),
            "classes": ["setosa", "versicolor", "virginica"]
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail="Batch prediction failed")

@app.post("/sentiment", tags=["LLM"])
@limiter.limit("200/minute")
async def sentiment(request: Request, input_data: SentimentInput):
    """Analyze sentiment of input text.
    
    Args:
        input_data: SentimentInput containing text to analyze
        
    Returns:
        Sentiment analysis result with label and score
        
    Raises:
        HTTPException: If sentiment analysis fails
    """
    try:
        result = llm_service.analyze_sentiment(input_data.text)
        logger.info(f"Sentiment analysis: {result['label']}")
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")

if __name__ == "__main__":
    import uvicorn
    # Bind to localhost only - change to 0.0.0.0 only if behind a reverse proxy
    uvicorn.run(app, host="127.0.0.1", port=8000)
