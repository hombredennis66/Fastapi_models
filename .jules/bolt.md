## 2025-05-22 - LLM Sentiment Analysis Optimization

**Learning:** Implementing an LRU cache for LLM sentiment analysis reduced average response time for repeated requests from ~21ms to ~3.5ms (in the test client environment). Lazy loading with `cached_property` helps avoid unnecessary initialization overhead when the model is not immediately needed.
**Action:** Use `functools.lru_cache` for deterministic ML inference tasks and `cached_property` for heavy resource initialization.
