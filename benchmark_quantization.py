import time
import torch
from transformers import pipeline
import numpy as np

def benchmark_pipeline(pipe, iterations=10):
    texts = [
        "This is a great movie!",
        "I hated the ending of that book.",
        "The weather is okay today.",
        "I love coding in Python.",
        "Performance optimization is fun!"
    ]
    # Warm up
    pipe(texts[0])

    start = time.perf_counter()
    for _ in range(iterations):
        for text in texts:
            pipe(text)
    end = time.perf_counter()

    avg_time = (end - start) / (iterations * len(texts)) * 1000
    return avg_time

print("Loading baseline model...")
pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)
baseline_time = benchmark_pipeline(pipe)
print(f"Baseline average latency: {baseline_time:.2f} ms")

print("Quantizing model...")
quantized_model = torch.quantization.quantize_dynamic(
    pipe.model, {torch.nn.Linear}, dtype=torch.qint8
)
pipe.model = quantized_model

quantized_time = benchmark_pipeline(pipe)
print(f"Quantized average latency: {quantized_time:.2f} ms")

improvement = (baseline_time - quantized_time) / baseline_time * 100
print(f"Improvement: {improvement:.2f}%")
