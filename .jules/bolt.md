## 2025-05-22 - [LLM Sentiment Analysis Optimization]
**Learning:** Sentiment analysis with LLMs is computationally expensive. Implementing LRU caching significantly reduces response time for repeated inputs. Additionally, lazy loading the transformer pipeline improves application startup time as the model is only loaded when first needed.
**Action:** Use `functools.lru_cache` for expensive ML/LLM inference methods and `functools.cached_property` for heavy object initialization (like model pipelines) in future optimizations.
