## 2026-06-08 - [FastAPI & LLM Optimization]
**Learning:** Initializing heavy LLM pipelines during startup can significantly increase cold start time. Using `cached_property` for lazy loading and `lru_cache` for result caching provides immediate performance wins. Furthermore, in FastAPI, synchronous CPU-bound tasks like ML inference should use `def` instead of `async def` to avoid blocking the event loop.
**Action:** Always verify if heavy dependencies can be lazy-loaded and ensure CPU-intensive route handlers are not blocking the main event loop by using standard `def` in FastAPI.

## 2026-06-09 - [Lazy Loading Transformers]
**Learning:** Top-level imports of heavy ML libraries like `transformers` can add seconds to the application startup time. In this environment, it accounted for over 70% of the cold-start duration.
**Action:** Move heavy library imports inside the specific methods or properties that use them to ensure they are only loaded when necessary.

## 2026-06-10 - [Lazy Loading ML Models and Heavy Dependencies]
**Learning:** Top-level imports and eager loading of Scikit-learn models using `joblib` can significantly delay FastAPI application startup. In this project, `joblib.load` alone took ~1.87s, contributing to a ~3.78s total startup time.
**Action:** Implement a service class (e.g., `MLService`) with a `cached_property` to lazily load the model and its heavy dependencies (`joblib`, `numpy`) only when the first inference request is received. This can reduce startup time by over 80%.
