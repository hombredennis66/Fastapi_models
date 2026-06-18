## 2026-06-08 - [FastAPI & LLM Optimization]
**Learning:** Initializing heavy LLM pipelines during startup can significantly increase cold start time. Using `cached_property` for lazy loading and `lru_cache` for result caching provides immediate performance wins. Furthermore, in FastAPI, synchronous CPU-bound tasks like ML inference should use `def` instead of `async def` to avoid blocking the event loop.
**Action:** Always verify if heavy dependencies can be lazy-loaded and ensure CPU-intensive route handlers are not blocking the main event loop by using standard `def` in FastAPI.

## 2026-06-09 - [Lazy Loading Transformers]
**Learning:** Top-level imports of heavy ML libraries like `transformers` can add seconds to the application startup time. In this environment, it accounted for over 70% of the cold-start duration.
**Action:** Move heavy library imports inside the specific methods or properties that use them to ensure they are only loaded when necessary.

## 2026-06-10 - [Lazy Loading Scikit-learn & NumPy]
**Learning:** Top-level imports of `numpy` and `joblib` plus loading a serialized model file (`.joblib`) during FastAPI startup adds significant overhead (e.g., ~3 seconds). Refactoring this into a service with `cached_property` and local imports deferred the cost until the first request.
**Action:** Move all heavy ML model loading and their dependencies into lazy-loaded properties to ensure near-instant application startup.

## 2026-06-11 - [ML Prediction Caching & NumPy Overhead Removal]
**Learning:** For ML models serving single-sample predictions (common in real-time APIs), `numpy` array allocation and reshaping can be a significant portion of the total latency. Furthermore, implementing an `lru_cache` for repeated requests provides a massive speedup (~300x). Using a `cached_property` to return a decorated function avoids memory leaks and the unhashable `self` issue in standard `lru_cache` on instance methods.
**Action:** Optimize single-sample ML predictions by passing data directly in expected shapes (e.g., list of tuples) and use per-instance caching for frequent identical inputs.
