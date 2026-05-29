import functools
from transformers import pipeline

class LLMService:
    def __init__(self):
        # Using a small, efficient model for sentiment analysis
        self.classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    @functools.lru_cache(maxsize=100)
    def analyze_sentiment(self, text: str):
        result = self.classifier(text)
        return result[0]

if __name__ == "__main__":
    service = LLMService()
    test_text = "I love machine learning!"
    print(f"Text: {test_text}")
    print(f"Sentiment: {service.analyze_sentiment(test_text)}")
