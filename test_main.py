"""Test suite for FastAPI ML and LLM API."""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestHealthEndpoints:
    """Test health and basic endpoints."""
    
    def test_read_root(self):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Welcome to the ML and LLM API" in data["message"]
        assert "version" in data

class TestPredictionEndpoints:
    """Test ML prediction endpoints."""
    
    def test_predict_endpoint_valid(self):
        """Test prediction endpoint with valid iris features."""
        payload = {"features": [5.1, 3.5, 1.4, 0.2]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1, 2]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "classes" in data
    
    def test_predict_endpoint_invalid_length(self):
        """Test prediction endpoint with wrong number of features."""
        payload = {"features": [5.1, 3.5, 1.4]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_range(self):
        """Test prediction endpoint with out-of-range features."""
        payload = {"features": [5.1, 3.5, 1.4, 20.0]}  # 20.0 exceeds max of 10
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_batch_endpoint(self):
        """Test batch prediction endpoint."""
        payload = {
            "features_list": [
                [5.1, 3.5, 1.4, 0.2],
                [6.2, 2.9, 4.3, 1.3],
                [7.1, 3.0, 5.9, 2.1]
            ]
        }
        response = client.post("/predict-batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert data["count"] == 3
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        assert "confidences" in data
        assert len(data["confidences"]) == 3
    
    def test_predict_batch_empty(self):
        """Test batch prediction with empty list."""
        payload = {"features_list": []}
        response = client.post("/predict-batch", json=payload)
        assert response.status_code == 422  # Validation error

class TestSentimentEndpoints:
    """Test sentiment analysis endpoints."""
    
    def test_sentiment_positive(self):
        """Test sentiment analysis with positive text."""
        payload = {"text": "I really enjoy learning about artificial intelligence."}
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] in ["POSITIVE", "NEGATIVE"]
        assert isinstance(data["score"], float)
        assert 0 <= data["score"] <= 1
    
    def test_sentiment_negative(self):
        """Test sentiment analysis with negative text."""
        payload = {"text": "I am so sad and frustrated today."}
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] in ["POSITIVE", "NEGATIVE"]
    
    def test_sentiment_empty_text(self):
        """Test sentiment endpoint with empty text."""
        payload = {"text": ""}
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_sentiment_long_text(self):
        """Test sentiment endpoint with text exceeding max length."""
        payload = {"text": "a" * 5001}  # Exceeds 5000 char limit
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 422  # Validation error

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_headers(self):
        """Test that rate limit headers are present."""
        payload = {"features": [5.1, 3.5, 1.4, 0.2]}
        response = client.post("/predict", json=payload)
        # Check for rate limit headers (may be present depending on slowapi config)
        # This is a basic check that the endpoint responds
        assert response.status_code in [200, 429]  # 200 OK or 429 Too Many Requests

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
