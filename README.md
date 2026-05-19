# FastAPI ML and LLM API

A secure and performant FastAPI application providing ML model inference (Iris classification) and LLM-based sentiment analysis.

## Features

### Security
- ✅ Input validation with Pydantic models and field constraints
- ✅ Model integrity verification
- ✅ Rate limiting (100/min for predictions, 200/min for sentiment)
- ✅ Secure host binding (localhost by default)
- ✅ Comprehensive error handling and logging
- ✅ Safe model serialization with joblib protocol 2

### Performance
- ✅ Response compression with GZIP middleware
- ✅ Batch prediction endpoint for throughput optimization
- ✅ Lightweight DistilBERT model for sentiment analysis
- ✅ Efficient model loading and caching
- ✅ Async/await support for non-blocking operations

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Train the iris classification model
python train_ml_model.py
```

This will:
- Load the iris dataset
- Train a Logistic Regression model
- Perform 5-fold cross-validation
- Validate performance against threshold (80%)
- Save model as `model.joblib`

## Running the API

```bash
# Start the server
python main.py

# Or with uvicorn directly
uvicorn main:app --reload
```

Server will be available at `http://localhost:8000`

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Health Check
```bash
GET /
```

Response:
```json
{
  "message": "Welcome to the ML and LLM API",
  "version": "2.0.0"
}
```

### Single Prediction
```bash
POST /predict
```

Request:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:
```json
{
  "prediction": 0,
  "confidence": 0.95,
  "classes": ["setosa", "versicolor", "virginica"]
}
```

### Batch Prediction
```bash
POST /predict-batch
```

Request:
```json
{
  "features_list": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.9, 4.3, 1.3],
    [7.1, 3.0, 5.9, 2.1]
  ]
}
```

Response:
```json
{
  "count": 3,
  "predictions": [0, 1, 2],
  "confidences": [0.95, 0.87, 0.92],
  "classes": ["setosa", "versicolor", "virginica"]
}
```

### Sentiment Analysis
```bash
POST /sentiment
```

Request:
```json
{
  "text": "I really enjoy learning about artificial intelligence."
}
```

Response:
```json
{
  "label": "POSITIVE",
  "score": 0.9987
}
```

## Rate Limits

- `/predict`: 100 requests/minute per IP
- `/predict-batch`: 50 requests/minute per IP
- `/sentiment`: 200 requests/minute per IP

## Testing

```bash
# Run all tests
pytest test_main.py -v

# Run specific test class
pytest test_main.py::TestPredictionEndpoints -v

# Run with coverage
pytest test_main.py --cov=. --cov-report=html
```

## Project Structure

```
.
├── main.py                 # FastAPI application
├── train_ml_model.py       # Model training script
├── llm_service.py          # Sentiment analysis service
├── test_main.py            # Test suite
├── model.joblib            # Trained ML model (generated)
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Security Considerations

1. **Input Validation**: All endpoints validate input types, lengths, and ranges
2. **Rate Limiting**: Prevents abuse and DDoS attacks
3. **Error Handling**: Errors are logged without exposing sensitive details
4. **Model Security**: Models are loaded from verified paths and validated
5. **Secure Defaults**: Server binds to localhost by default; use reverse proxy in production

## Performance Optimization

1. **Compression**: Automatic GZIP compression for responses > 1KB
2. **Batch Processing**: `/predict-batch` endpoint for processing multiple samples
3. **Efficient Models**: DistilBERT is 40% smaller than BERT
4. **Async Support**: Non-blocking I/O for concurrent requests

## Production Deployment

For production, consider:

1. **Reverse Proxy**: Use Nginx or Apache in front of the API
2. **SSL/TLS**: Enable HTTPS
3. **Environment Variables**: Use `.env` file for configuration
4. **Monitoring**: Add Prometheus metrics and centralized logging
5. **Authentication**: Add API key or OAuth2 authentication
6. **Containerization**: Use Docker for consistent deployments

## Dependencies

- **FastAPI**: Modern web framework
- **scikit-learn**: ML algorithms
- **transformers**: NLP models
- **pydantic**: Data validation
- **slowapi**: Rate limiting
- **pytest**: Testing framework

## License

MIT License - See LICENSE file for details
