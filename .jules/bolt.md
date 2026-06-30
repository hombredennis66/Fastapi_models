## 2026-06-08 - [FastAPI & LLM Optimization]
**Learning:** Initializing heavy LLM pipelines during startup can significantly increase cold start time. Using `cached_property` for lazy loading and `lru_cache` for result caching provides immediate performance wins. Furthermore, in FastAPI, synchronous CPU-bound tasks like ML inference should use `def` instead of `async def` to avoid blocking the event loop.
**Action:** Always verify if heavy dependencies can be lazy-loaded and ensure CPU-intensive route handlers are not blocking the main event loop by using standard `def` in FastAPI.

## 2026-06-09 - [Lazy Loading Transformers]
**Learning:** Top-level imports of heavy ML libraries like `transformers` can add seconds to the application startup time. In this environment, it accounted for over 70% of the cold-start duration.
**Action:** Move heavy library imports inside the specific methods or properties that use them to ensure they are only loaded when necessary.

## 2026-06-10 - [Lazy Loading Scikit-learn & NumPy]
**Learning:** Top-level imports of `numpy` and `joblib` plus loading a serialized model file (`.joblib`) during FastAPI startup adds significant overhead (e.g., ~3 seconds). Refactoring this into a service with `cached_property` and local imports deferred the cost until the first request.
**Action:** Move all heavy ML model loading and their dependencies into lazy-loaded properties to ensure near-instant application startup.

## 2026-06-11 - [ML Prediction Path Optimization]
**Learning:** For low-latency ML services, the overhead of creating NumPy arrays and importing NumPy in the hot path can be significant (~10% of execution time for simple models). Furthermore, instance-level caching using `lru_cache` inside a `cached_property` provides massive speedups for repeated requests without the memory risks of global caches or unhashable `self` issues.
**Action:** Use `tuple` conversion and per-instance `lru_cache` for numerical feature caching. Pass lists directly to scikit-learn's `predict` to avoid unnecessary NumPy allocations in the hot path.

## 2026-06-12 - [LLM Pipeline Caching & Truncation]
**Learning:** Using `@lru_cache` on instance methods leads to memory leaks and hashability issues. Combining `@cached_property` for the heavy pipeline object with an internal cached function for results ensures both fast loading and efficient inference without reloading the model. Enabling `truncation=True` is critical for robustness against long inputs.
**Action:** Use the per-instance caching pattern (cached property returning an inner decorated function) for all model inference services. Always enable truncation for LLM pipelines unless full context is strictly required.

## 2026-06-13 - [LLM Dynamic Quantization]
**Learning:** Applying `torch.quantization.quantize_dynamic` to the linear layers of a DistilBERT sentiment-analysis pipeline reduces CPU inference latency by approximately 48% (from ~90ms to ~47ms) in this environment, with negligible impact on accuracy.
**Action:** For CPU-bound LLM inference, always consider 8-bit dynamic quantization of linear layers as a high-impact, low-risk optimization.
