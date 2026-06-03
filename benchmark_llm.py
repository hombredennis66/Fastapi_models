import time
import json
from main import app
from fastapi.testclient import TestClient

def benchmark_sentiment():
    with TestClient(app) as client:
        payload = {"text": "FastAPI is amazing and very fast!"}

        # Warm up
        client.post("/sentiment", json=payload)

        iterations = 20
        start_time = time.time()
        for i in range(iterations):
            client.post("/sentiment", json=payload)
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        print(f"Average time for /sentiment (repeated, iterations={iterations}): {avg_time*1000:.2f}ms")

if __name__ == "__main__":
    benchmark_sentiment()
