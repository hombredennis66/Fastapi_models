## 2025-05-15 - FastAPI CPU-bound Concurrency
**Learning:** Defining CPU-bound endpoints (like ML/LLM inference) as `async def` in FastAPI blocks the main event loop, as Python's global interpreter lock (GIL) and the single-threaded nature of the event loop cannot handle heavy computation without blocking.
**Action:** Use standard `def` for CPU-bound routes to leverage FastAPI's internal thread pool, ensuring the event loop remains free to handle other concurrent requests.

## 2025-05-15 - LLM Service Loading and Caching
**Learning:** Loading a transformer pipeline during class initialization can significantly slow down application startup and test collection. Additionally, redundant inference on identical strings is a major waste of resources.
**Action:** Use `functools.cached_property` for lazy loading of the model and `functools.lru_cache` for memoizing inference results. This reduces repeated request latency from ~30ms to <1ms.
