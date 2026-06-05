## 2025-05-14 - Optimized LLM Sentiment Analysis with Caching and Lazy Loading

**Learning:** Implementing `functools.lru_cache` on LLM inference methods can drastically reduce latency for repeated requests (from ~28ms to ~0.0004ms). Combining this with `functools.cached_property` for model initialization ensures the heavy model pipeline is only loaded when actually needed, improving app startup time and resource usage in environments where the LLM service might not be called.

**Action:** Always consider caching strategies for expensive deterministic outputs like sentiment analysis. Use lazy loading for large model weights to keep the initial process footprint light.
