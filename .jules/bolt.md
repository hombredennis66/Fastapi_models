## 2025-05-22 - [LLM Sentiment Caching]
**Learning:** Sentiment analysis with DistilBERT takes ~20ms even for small inputs. Using `lru_cache` on repeated strings reduces latency to <1ms.
**Action:** Always consider input caching for deterministic NLP tasks like sentiment analysis or tokenization where the input domain (natural language) often contains repeats in production traffic.

**Learning:** `lru_cache` on instance methods can prevent garbage collection of the instance. However, for model services intended as singletons, this is an acceptable trade-off for the performance win. Always use `.copy()` when returning cached dicts to prevent cache pollution.
**Action:** Use a private cached method and return a copy in the public method to maintain cache integrity.
