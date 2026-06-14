from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_service import LLMService
from ml_service import MLService
from typing import List

app = FastAPI(title="ML and LLM API")

# Initialize services (lazy loaded)
ml_service = MLService()
llm_service = LLMService()

class PredictionInput(BaseModel):
    features: List[float]

class SentimentInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the ML and LLM API"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        prediction = ml_service.predict(input_data.features)
        return {"prediction": prediction}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
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
