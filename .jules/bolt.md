## 2026-06-08 - [FastAPI & LLM Optimization]
**Learning:** Initializing heavy LLM pipelines during startup can significantly increase cold start time. Using `cached_property` for lazy loading and `lru_cache` for result caching provides immediate performance wins. Furthermore, in FastAPI, synchronous CPU-bound tasks like ML inference should use `def` instead of `async def` to avoid blocking the event loop.
**Action:** Always verify if heavy dependencies can be lazy-loaded and ensure CPU-intensive route handlers are not blocking the main event loop by using standard `def` in FastAPI.

## 2026-06-09 - [Lazy Loading Transformers]
**Learning:** Top-level imports of heavy ML libraries like `transformers` can add seconds to the application startup time. In this environment, it accounted for over 70% of the cold-start duration.
**Action:** Move heavy library imports inside the specific methods or properties that use them to ensure they are only loaded when necessary.

## 2026-06-10 - [Lazy Loading Scikit-learn & NumPy]
**Learning:** Top-level imports of `numpy` and `joblib` plus loading a serialized model file (`.joblib`) during FastAPI startup adds significant overhead (e.g., ~3 seconds). Refactoring this into a service with `cached_property` and local imports deferred the cost until the first request.
**Action:** Move all heavy ML model loading and their dependencies into lazy-loaded properties to ensure near-instant application startup.

## 2026-06-11 - [ML Prediction Caching & Overhead Reduction]
**Learning:** For single-sample ML predictions using Scikit-learn, the overhead of creating NumPy arrays and reshaping can be significant compared to the inference time itself. Implementing an instance-level `lru_cache` (via `cached_property` and a closure) provides a ~350x speedup for repeated requests.
**Action:** Use instance-scoped `lru_cache` for repetitive ML tasks and pass list of tuples directly to `model.predict()` to bypass redundant array allocations.
