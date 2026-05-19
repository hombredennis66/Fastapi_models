from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZIPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
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

# Model path configuration
MODEL_PATH = Path(__file__).parent / 'model.joblib'

# Global dictionary to hold our models safely
models = {}
llm_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to load models on startup and clean up on shutdown."""
    global llm_service
    # Load ML model
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        models["iris_model"] = joblib.load(MODEL_PATH)
        logger.info(f"ML model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading ML model: {e}")
        models["iris_model"] = None

    # Load LLM Service
    llm_service = LLMService()
    
    yield # The app runs while in this state
    
    # Clean up when the server shuts down
    models.clear()
    logger.info("Models unloaded securely.")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ML and LLM API",
    description="Secure and performant ML and LLM inference API",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS Middleware for frontend security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change "*" to your specific frontend URL in production!
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add GZIP compression middleware
app.add_middleware(GZIPMiddleware, minimum_size=1000)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# --- Pydantic Models (Unchanged, they were perfect!) ---

class PredictionInput(BaseModel):
    features: List[float] = Field(
        ..., min_items=4, max_items=4,
        description="List of 4 float values representing iris flower features"
    )
    
    @validator('features')
    def validate_features(cls, v):
        if not all(0 <= f <= 10 for f in v):
            raise ValueError("All features must be between 0 and 10")
        return v

class BatchPredictionInput(BaseModel):
    features_list: List[List[float]] = Field(
        ..., min_items=1, max_items=1000,
        description="List of feature arrays (max 1000 samples)"
    )
    
    @validator('features_list')
    def validate_batch(cls, v):
        for features in v:
            if len(features) != 4:
                raise ValueError("Each feature array must have exactly 4 elements")
            if not all(0 <= f <= 10 for f in features):
                raise ValueError("All features must be between 0 and 10")
        return v

class SentimentInput(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=5000,
        description="Text to analyze for sentiment (max 5000 characters)"
    )

# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Welcome to the ML and LLM API", "version": "2.0.0"}

@app.post("/predict", tags=["ML"])
@limiter.limit("100/minute")
async def predict(request: Request, input_data: PredictionInput):
    ml_model = models.get("iris_model")
    if ml_model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        features = np.array(input_data.features).reshape(1, -1)
        
        # Performance improvement: Run blocking prediction in a background thread
        prediction = await run_in_threadpool(ml_model.predict, features)
        probabilities = await run_in_threadpool(ml_model.predict_proba, features)
        confidence = probabilities.max()
        
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
    ml_model = models.get("iris_model")
    if ml_model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        features = np.array(input_data.features_list)
        
        # Run blocking batch prediction in a background thread
        predictions = await run_in_threadpool(ml_model.predict, features)
        probabilities = await run_in_threadpool(ml_model.predict_proba, features)
        confidences = probabilities.max(axis=1)
        
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
    try:
        # Assuming llm_service is also synchronous, run it in a threadpool!
        result = await run_in_threadpool(llm_service.analyze_sentiment, input_data.text)
        logger.info(f"Sentiment analysis: {result['label']}")
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
