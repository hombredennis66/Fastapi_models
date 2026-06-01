from transformers import pipeline
from functools import lru_cache

class LLMService:
    def __init__(self):
        # Using a small, efficient model for sentiment analysis
        self.classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    @lru_cache(maxsize=100)
    def _get_cached_sentiment(self, text: str):
        # Internal method to leverage lru_cache for sentiment analysis
        result = self.classifier(text)
        return result[0]

    def analyze_sentiment(self, text: str):
        # Return a copy of the cached result to prevent mutation of the cache entry
        return self._get_cached_sentiment(text).copy()

if __name__ == "__main__":
    service = LLMService()
    test_text = "I love machine learning!"
    print(f"Text: {test_text}")
    print(f"Sentiment: {service.analyze_sentiment(test_text)}")
