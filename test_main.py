import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML and LLM API"}

def test_predict_endpoint():
    # Iris flower features (sepal length, sepal width, petal length, petal width)
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)

@patch("llm_service.LLMService.analyze_sentiment")
def test_sentiment_endpoint(mock_analyze):
    mock_analyze.return_value = {"label": "POSITIVE", "score": 0.99}
    payload = {"text": "I really enjoy learning about artificial intelligence."}
    response = client.post("/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert data["label"] == "POSITIVE"

@patch("llm_service.LLMService.analyze_sentiment")
def test_sentiment_negative_endpoint(mock_analyze):
    mock_analyze.return_value = {"label": "NEGATIVE", "score": 0.99}
    payload = {"text": "I am so sad today."}
    response = client.post("/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] == "NEGATIVE"
