import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Mock the LLM service at the module level to avoid any network calls during import or test execution
with patch("llm_service.LLMService.analyze_sentiment") as mock_sentiment:
    # Set a default return value for the mock
    mock_sentiment.return_value = {"label": "POSITIVE", "score": 0.99}
    from main import app

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

def test_sentiment_endpoint():
    with patch("main.llm_service.analyze_sentiment") as mock_sentiment:
        mock_sentiment.return_value = {"label": "POSITIVE", "score": 0.999}
        payload = {"text": "I really enjoy learning about artificial intelligence."}
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] == "POSITIVE"
        mock_sentiment.assert_called_once_with(payload["text"])

def test_sentiment_negative_endpoint():
    with patch("main.llm_service.analyze_sentiment") as mock_sentiment:
        mock_sentiment.return_value = {"label": "NEGATIVE", "score": 0.999}
        payload = {"text": "I am so sad today."}
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert data["label"] == "NEGATIVE"
        mock_sentiment.assert_called_once_with(payload["text"])
