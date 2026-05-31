import functools
from transformers import pipeline

class LLMService:
    def __init__(self):
        # Using a small, efficient model for sentiment analysis
        self.classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    @functools.lru_cache(maxsize=100)
    def _get_cached_sentiment(self, text: str):
        """Internal method to handle caching of sentiment analysis results."""
        result = self.classifier(text)
        return result[0]

    def analyze_sentiment(self, text: str):
        """
        Analyzes the sentiment of the input text.
        Uses LRU cache to speed up repeated requests.
        """
        result = self._get_cached_sentiment(text)
        # Return a copy to prevent mutation of the cached object
        return result.copy()

if __name__ == "__main__":
    service = LLMService()
    test_text = "I love machine learning!"
    print(f"Text: {test_text}")
    print(f"Sentiment: {service.analyze_sentiment(test_text)}")
