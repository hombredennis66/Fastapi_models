from transformers import pipeline
from functools import lru_cache, cached_property

class LLMService:
    @cached_property
    def classifier(self):
        # Lazy load the model only when first accessed
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    @lru_cache(maxsize=100)
    def analyze_sentiment(self, text: str):
        """
        Analyzes the sentiment of the given text.
        Uses LRU cache to avoid redundant inference for identical inputs.
        """
        result = self.classifier(text)
        return result[0]

if __name__ == "__main__":
    service = LLMService()
    test_text = "I love machine learning!"
    print(f"Text: {test_text}")
    print(f"Sentiment: {service.analyze_sentiment(test_text)}")
