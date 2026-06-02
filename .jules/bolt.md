# BOLT'S JOURNAL - CRITICAL LEARNINGS ONLY

## 2025-05-22 - LLM Sentiment Analysis Cache
**Learning:** Repetitive sentiment analysis requests for the same text are a common pattern in LLM-powered APIs. Using `functools.lru_cache` provides a simple but effective way to eliminate redundant model inference, which is significantly more expensive than a cache lookup.
**Action:** Implement `lru_cache` on the `LLMService.analyze_sentiment` method.
