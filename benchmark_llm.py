import time
import httpx
import statistics

def benchmark_sentiment(url="http://localhost:8000/sentiment", iterations=20):
    payloads = [
        {"text": "This is a fantastic day and I am feeling great!"},
        {"text": "I am so tired and everything is going wrong."},
        {"text": "The movie was okay, not great but not terrible either."},
        {"text": "Artificial intelligence is a fascinating field with many possibilities."},
        {"text": "I hate waiting in long lines at the grocery store."}
    ]

    latencies = []

    with httpx.Client(timeout=60.0) as client:
        # Warm up
        client.post(url, json=payloads[0])

        for i in range(iterations):
            payload = payloads[i % len(payloads)]
            start_time = time.perf_counter()
            response = client.post(url, json=payload)
            end_time = time.perf_counter()

            if response.status_code == 200:
                latencies.append((end_time - start_time) * 1000)
            else:
                print(f"Error: {response.status_code}")

    if latencies:
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"Median Latency: {median_latency:.2f} ms")
        return avg_latency
    return None

if __name__ == "__main__":
    # Note: Assumes the server is running
    benchmark_sentiment()
