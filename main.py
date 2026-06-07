from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from llm_service import LLMService
from typing import List

app = FastAPI(title="ML and LLM API")

# Load ML model
try:
    ml_model = joblib.load('model.joblib')
except Exception as e:
    print(f"Error loading ML model: {e}")
    ml_model = None

# Initialize LLM service
llm_service = LLMService()

class PredictionInput(BaseModel):
    features: List[float]

class SentimentInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Welcome to the ML and LLM API"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if ml_model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        features = np.array(input_data.features).reshape(1, -1)
        prediction = ml_model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/sentiment")
def sentiment(input_data: SentimentInput):
    try:
        result = llm_service.analyze_sentiment(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
